#!/usr/bin/env python3
"""v52: real CMU human relative pre-impact pose -> bounded G1 29DoF reference.

A source is a 120 Hz CMU soccer pose with an *inferred optical ball-motion*
anchor, not verified foot-ball contact, impact force, or professional strike.
This is a frozen offline IK/projection study; a separately executed IsaacLab
G1/PhysX test is required before any physical performance claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

from cmu_g1_semantic_bridge_v051 import (ROOT, STUDY, HUMAN_FEATURE_INDEX,
    SIX_FEATURE_NAMES, CMU, human_six_timed, g1_six_semantic_relations, vector_cos)
from g1_contact_coordination_projection_v040 import (
    TorchG1Relations, load_robot_source, interpolate_source)
from g1_coordination_bridge_v039 import BODY

CONFIG = ROOT / "configs/cmu_g1_phase_projection_v052.json"
URDF = ROOT / "configs/paid_original/paid_main_resolved.urdf"
V51 = ROOT / "data/research_human_kicks/cmu_g1_cross_morph_v051/cross_morphology_audit.json"
OUTPUT = ROOT / "experiments/cmu_real_120hz_phase_v052"
ROBOT = ROOT / "reports/v048_preimpact_new01_original_v32.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def g1_six_differentiable(fk: TorchG1Relations, root: torch.Tensor,
                          q: torch.Tensor) -> torch.Tensor:
    """Same six named geometry features as v51, but keep autograd."""
    _, links = fk(root, q, with_links=True)
    p = {name: links[link][:, :3, 3] for name, link in BODY.items()}
    pelvis = (p["left_hip"] + p["right_hip"]) * .5
    shoulder = (p["left_shoulder"] + p["right_shoulder"]) * .5
    trunk = shoulder - pelvis
    return torch.stack((
        vector_cos(p["left_hip"]-p["left_knee"], p["left_ankle"]-p["left_knee"]),
        vector_cos(p["right_hip"]-p["right_knee"], p["right_ankle"]-p["right_knee"]),
        vector_cos(p["left_shoulder"]-p["left_elbow"], p["left_wrist"]-p["left_elbow"]),
        vector_cos(p["right_shoulder"]-p["right_elbow"], p["right_wrist"]-p["right_elbow"]),
        vector_cos(trunk, p["left_knee"]-p["left_hip"]),
        vector_cos(trunk, p["right_knee"]-p["right_hip"]),
    ), dim=-1)


def phase_window(relative):
    """A zero-ended pre-contact gate. Does not overwrite original postimpact."""
    relative = np.asarray(relative, dtype=np.float64)
    smooth = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))
    gate = smooth((relative + .40)/.10) * smooth((-.06 - relative)/.10)
    gate[relative >= -.06 - 1e-9] = 0.
    return gate


def joint_limits(names):
    tree = ET.parse(URDF).getroot()
    limits = {}
    for j in tree.findall("joint"):
        if j.attrib["type"] == "revolute":
            x = j.find("limit")
            limits[j.attrib["name"]] = (float(x.attrib["lower"]), float(x.attrib["upper"]))
    if len(names) != 29 or any(name not in limits for name in names):
        raise ValueError("G1 29DoF verified URDF joint ordering required")
    return np.asarray([limits[n] for n in names], dtype=np.float64)


def verified_inputs(cfg):
    if cfg["version"] != "0.1.52" or cfg["verified_human_contact"]:
        raise ValueError("Human impact label is never established")
    if cfg["human_clips"] != ["10_02", "10_05"] or cfg["robot_variant"] != "v32_frozen":
        raise ValueError("Clip/control selection is frozen before outcomes")
    for relative, expected in cfg["frozen_sha256"].items():
        path = ROOT / relative
        if digest(path) != expected:
            raise ValueError("Frozen source changed: " + relative)
    v51 = json.loads(V51.read_text())
    if v51["version"] != "0.1.51" or v51["contact_conditioned_true_human_kinetics_qualified"]:
        raise ValueError("v51 qualification changed")
    human = json.loads((ROOT / CMU / "quality_audit.json").read_text())
    return human, load_robot_source(ROBOT)


def optimize_candidate(fk, root, base_q, human, relative, cfg):
    """One fixed 20x29 differentiable morphology-aware projection."""
    if human.shape != (20, 6) or base_q.shape != (20, 29):
        raise ValueError("Measured time/channel counts differ from fixed protocol")
    if not np.isfinite(human).all():
        raise ValueError("CMU source contains missing pose data")
    gate = phase_window(relative)
    X = torch.as_tensor(root, dtype=torch.float64)
    Q = torch.as_tensor(base_q, dtype=torch.float64)
    G = torch.as_tensor(gate[:, None], dtype=torch.float64)
    with torch.no_grad():
        baseline = g1_six_differentiable(fk, X, Q)
        _, bodies = fk(X, Q, with_links=True)
        left = bodies["left_ankle_roll_link"][:, :3, 3].detach()
        right = bodies["right_ankle_roll_link"][:, :3, 3].detach()
    human_relative = np.clip(human - human[:1], -.30, .30)
    target = torch.as_tensor(np.clip(
        baseline.numpy() + cfg["relative_gain"] * gate[:, None] * human_relative, -1, 1),
        dtype=torch.float64)
    u = torch.nn.Parameter(torch.zeros_like(Q))
    optim = torch.optim.Adam([u], lr=cfg["optimizer_lr"])
    history = []
    for it in range(cfg["optimizer_iterations"]):
        delta = cfg["optimizer_rad"] * torch.tanh(u) * G
        pred = g1_six_differentiable(fk, X, Q+delta)
        _, links = fk(X, Q+delta, with_links=True)
        mismatch = ((pred-target).square() * G).sum()/G.sum().clamp_min(1)/6.
        stance = (links["left_ankle_roll_link"][:, :3, 3]-left).square().sum(dim=-1).mean()
        swing = (links["right_ankle_roll_link"][:, :3, 3]-right).square().sum(dim=-1).mean()
        temporal = (delta[1:]-delta[:-1]).square().mean()
        accel = (delta[2:]-2*delta[1:-1]+delta[:-2]).square().mean()
        loss = (mismatch + cfg["support_penalty"]*stance +
                cfg["swing_penalty"]*swing + cfg["joint_penalty"]*delta.square().mean() +
                cfg["temporal_penalty"]*(temporal+accel))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if not bool(torch.isfinite(u.grad).all()):
            raise FloatingPointError("Nonfinite differentiable G1 29DoF optimizer gradient")
        torch.nn.utils.clip_grad_norm_([u], 2.)
        optim.step()
        if it in (0, cfg["optimizer_iterations"]-1):
            history.append({"iteration": it+1, "loss": float(loss.detach()),
                            "proxy_feature_loss": float(mismatch.detach())})
    with torch.no_grad():
        delta = (cfg["optimizer_rad"] * torch.tanh(u)*G).numpy()
        after = g1_six_differentiable(fk, X, Q+torch.as_tensor(delta)).numpy()
        before = baseline.numpy()
        wanted = target.numpy()
        mask = gate > 0.05
        err0 = float(np.mean((before[mask]-wanted[mask])**2))
        err1 = float(np.mean((after[mask]-wanted[mask])**2))
        if not (np.isfinite(err1) and err1 < err0):
            raise ValueError("Targeted real human phase projection failed to reduce proxy objective")
    return delta, dict(proxy_MSE_before=err0, proxy_MSE_after=err1,
        relative_proxy_reduction=(err0-err1)/max(1e-12, err0),
        per_channel_baseline_proxy_MSE=np.mean((before[mask]-wanted[mask])**2,axis=0).tolist(),
        per_channel_projected_proxy_MSE=np.mean((after[mask]-wanted[mask])**2,axis=0).tolist(),
        history=history, phase_window_nonzero_samples=int(np.count_nonzero(gate>0)),
        gate_start_end=[float(gate[0]), float(gate[-1])],
        observed_human_change_rms=float(np.sqrt(np.mean((human-human[:1])**2))))


def full_reference(delta, relative, robot, cfg):
    anchor = float(robot["contact_time"])
    near = anchor + relative
    ts = robot["times"]
    out = np.stack([np.interp(ts, near, delta[:, i], left=0., right=0.)
                    for i in range(29)], axis=-1)
    out[(ts < near[0]) | (ts >= anchor-.06-1e-8)] = 0.
    peak = float(np.abs(out).max())
    slew = float(np.abs(np.diff(out, axis=0)).max())
    scale = min(1., cfg["max_joint_rad"]/max(peak,1e-15),
                cfg["max_slew_per_20ms_rad"]/max(slew,1e-15))
    out *= scale
    if not np.isfinite(out).all() or out.shape != (700,29):
        raise ValueError("Bad bounded output trajectory")
    bounds = joint_limits(robot["joint_names"])
    low, high = bounds[:,0], bounds[:,1]
    src_viol = (robot["q"] < low-1e-5) | (robot["q"] > high+1e-5)
    new_viol = (robot["q"]+out < low-1e-5) | (robot["q"]+out > high+1e-5)
    if (new_viol & ~src_viol).any():
        raise ValueError("Human-guided output adds a G1 URDF joint limit breach")
    if np.count_nonzero(np.abs(out)>1e-7) < 3:
        raise ValueError("Generated real-CMU reference was physically unobservable")
    return out.astype(np.float32), dict(
        source_URDF_joint_limit_breaches=int(src_viol.sum()),
        candidate_URDF_joint_limit_breaches=int(new_viol.sum()),
        reference_peak_rad=float(np.abs(out).max()),
        reference_max_step_rad=float(np.abs(np.diff(out,axis=0)).max()),
        active_controller_frames=int(np.count_nonzero(np.max(np.abs(out),axis=1)>1e-7)),
        nonzero_joints=int(np.count_nonzero(np.max(np.abs(out),axis=0)>1e-7)),
        global_reference_scale=scale,
        note="Static/reference bounds are NOT actual PhysX joint speed, torque or stability gates")


def build(clip, cfg):
    out_dir = OUTPUT / clip
    path = out_dir / "g1_joint_delta_50hz.npz"
    audit = out_dir / "candidate_audit.json"
    phys_cfg = ROOT / ("configs/cmu_real_120hz_" + clip + "_physx_v052.json")
    if any(z.exists() for z in (path, audit, phys_cfg)):
        raise FileExistsError("Never overwrite an existing v52 trial artifact")
    human_source, robot = verified_inputs(cfg)
    relative = np.arange(-.40, 0., .02).round(8)
    human, human_anchor = human_six_timed(clip, relative, human_source)
    root, q = interpolate_source(robot, robot["contact_time"] + relative)
    fk = TorchG1Relations(robot["joint_names"])
    with torch.no_grad():
        ref = g1_six_semantic_relations(fk, torch.tensor(root,dtype=torch.float64),
                                       torch.tensor(q,dtype=torch.float64))
        grad = g1_six_differentiable(fk,torch.tensor(root,dtype=torch.float64),
                                    torch.tensor(q,dtype=torch.float64)).numpy()
        np.testing.assert_allclose(ref,grad,atol=2e-12,rtol=0)
    d, score = optimize_candidate(fk,root,q,human,relative,cfg)
    correction, safety = full_reference(d,relative,robot,cfg)
    out_dir.mkdir(parents=True)
    np.savez_compressed(path,delta_joint_position_rad=correction,
         controller_time_s=robot["times"].astype(np.float64),
         joint_names=np.asarray(robot["joint_names"]),
         source_clip_id=np.asarray(clip),
         human_ball_marker_inferred_launch_s=np.asarray(human_anchor),
         source_is_verified_instep=np.asarray(False),
         human_impact_time_verified=np.asarray(False),
         robot_PhysX_impact_time_s=np.asarray(robot["contact_time"]),
         reference_source=np.asarray("original_v32_G1_PhysX"),
         six_named_geometry_channels=np.asarray(SIX_FEATURE_NAMES),
         human_source_sha256=np.asarray(cfg["frozen_sha256"]["data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050/"+clip+".npz"]))
    old = json.loads((ROOT/"configs/state_conditioned_strike_v15.json").read_text())
    old["version"] = "0.1.52"
    old["research_scope"] = ("Prespecified real 120Hz optical CMU ball-marker motion onset, "
        "NOT true human instep/contact power. Single original v32 robot; v41 safety gate unchanged.")
    old["human_coordination_reference"]["trajectory"] = str(path.relative_to(ROOT))
    old["human_coordination_reference"]["trajectory_sha256"] = digest(path)
    phys_cfg.write_text(json.dumps(old,ensure_ascii=False,indent=2)+"\n")
    result = {
      "version":"0.1.52","human_clip":clip,"human_subjects_in_training":1,
      "robot_reference":"v32_frozen","human_optical_launch_is_NOT_contact":True,
      "actual_PhysX_policy_tested":False,"independent_heldout_human_evaluation":False,
      "source_relative_phase_s":relative.tolist(),"human_optical_ball_launch_s":human_anchor,
      "G1_real_PhysX_impact_s":float(robot["contact_time"]),
      "cross_species_skeleton_or_force_equivalence":False,
      "g1_urdf_differentiable_six_feature_FK_matches_v51":True,
      "optimization":score,"reference_safety":safety,
      "human_pose_to_robot_joint_action_method":"Regularized differentiable G1 29DoF IK for six semantic relative pose changes; original recorded floating root fixed ONLY offline",
      "human_kick_force_or_instep_class_supervised":False,
      "output_file":str(path.relative_to(ROOT)),
      "output_sha256":digest(path),
      "physics_config":str(phys_cfg.relative_to(ROOT)),
      "physics_config_sha256":digest(phys_cfg),
      "guard":"Actual qdot, foot/ball contact, shot, delayed support and 14-second postshot dynamics require original PhysX rollout; not implied by offline optimizer"}
    audit.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("V052_REAL_CMU_CANDIDATE",json.dumps({"clip":clip,"reference_sha":digest(path),
         "proxy_improvement":round(score["relative_proxy_reduction"],6),
         "peak_rad":round(safety["reference_peak_rad"],6),
         "nonzero_frames":safety["active_controller_frames"]}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip",choices=["10_02","10_05"],required=True)
    args=p.parse_args()
    cfg=json.loads(CONFIG.read_text())
    build(args.clip,cfg)


if __name__=="__main__":
    main()
