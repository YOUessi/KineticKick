#!/usr/bin/env python3
"""Frozen full-horizon 29DoF shared-latent *engineering* candidates.

CPU-only. Not professional human force imitation, not inverse dynamics,
not an Isaac PhysX experiment and NEVER an actuator interface.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

import v127_collider_proximity as geom
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/v135_frozen_chain29_offline_ablation.json"
FROZEN_SHA = "8c4e56dce15a1fffe424e1e0cf1b2ec2dc0f278d"
PHASES = ("approach", "stance", "backswing", "hip_trunk_accel",
          "instep_contact", "follow_through", "landing_recovery")
KNOTS = np.array([
    [0., 0., 0., 0.],
    [.35, .10, -.20, .04],
    [.85, .25, -.50, -.10],
    [.90, -.35, -.45, -.80],
    [.98, .95, 1.00, .65],
    [.95, .90, 1.00, 1.00],
    [.65, .55, .35, .60],
    [0., 0., 0., 0.],
], dtype=np.float64)

# Predeclared group coupling, manually reconstructed from tracked v131 *protocol*;
# these numerical weights are an engineering hypothesis, NOT learned from CMU.
SYNERGY = {
    "left_hip_pitch_joint": (.8, -.2, 0., .1),
    "right_hip_pitch_joint": (-.35, .1, 0., -.85),
    "left_hip_roll_joint": (.7, .2, 0., 0.),
    "right_hip_roll_joint": (-.6, -.2, 0., .2),
    "left_hip_yaw_joint": (.2, -.8, 0., .1),
    "right_hip_yaw_joint": (-.2, .8, 0., -.2),
    "left_knee_joint": (.7, 0., 0., -.15),
    "right_knee_joint": (-.4, 0., 0., 1.),
    "left_ankle_pitch_joint": (-.7, 0., 0., .1),
    "right_ankle_pitch_joint": (.3, 0., 0., -.7),
    "left_ankle_roll_joint": (-.35, 0., 0., 0.),
    "right_ankle_roll_joint": (.25, 0., 0., .1),
    "waist_yaw_joint": (0., 1., -.2, .2),
    "waist_roll_joint": (-.3, .6, .1, 0.),
    "waist_pitch_joint": (.15, .75, -.15, .4),
    "left_shoulder_pitch_joint": (0., -.3, -1., .4),
    "right_shoulder_pitch_joint": (0., .3, 1., -.5),
    "left_shoulder_roll_joint": (0., -.2, -.7, .2),
    "right_shoulder_roll_joint": (0., .2, .7, -.2),
    "left_shoulder_yaw_joint": (0., .45, -.6, .3),
    "right_shoulder_yaw_joint": (0., -.45, .6, -.3),
    "left_elbow_joint": (0., 0., .8, -.3),
    "right_elbow_joint": (0., 0., -.8, .3),
    "left_wrist_roll_joint": (0., 0., .2, 0.),
    "right_wrist_roll_joint": (0., 0., -.2, 0.),
    "left_wrist_pitch_joint": (0., .1, .25, 0.),
    "right_wrist_pitch_joint": (0., -.1, -.25, 0.),
    "left_wrist_yaw_joint": (0., .2, .15, 0.),
    "right_wrist_yaw_joint": (0., -.2, -.15, 0.),
}
GROUPS = {
    "left_leg": ("left_hip_", "left_knee_", "left_ankle_"),
    "right_leg": ("right_hip_", "right_knee_", "right_ankle_"),
    "waist_trunk": ("waist_",),
    "left_arm": ("left_shoulder_", "left_elbow_", "left_wrist_"),
    "right_arm": ("right_shoulder_", "right_elbow_", "right_wrist_"),
}


def frozen() -> dict:
    blob = subprocess.check_output([
        "git", "-C", str(ROOT), "show",
        FROZEN_SHA + ":configs/v135_frozen_chain29_offline_ablation.json",
    ])
    if blob != CFG.read_bytes():
        raise ValueError("V135_PROTOCOL_DRIFT")
    c = json.loads(blob)
    if (c["version"] != "v0.1.135"
            or c["parent_sha"] != "3268bd0515a9097c9939db21fcaa8e7d41085ffa"
            or c["dt_s"] != .005 or c["horizon_s"] != 14.0
            or c["original_dof"] != 29 or c["all_phases"] != list(PHASES)
            or c["phase_durations_s"] != [1., .8, .6, .5, .1, 1., 10.]
            or c["multipliers"] != [0., 1., 2., 4., -1.]
            or c["latent_amplitudes_rad"] != [.075, .060, .055, .070]
            or c["native_gpu_physx_episodes_while_foreign_gpu_occupied"] != 0
            or c["allow_motor_execution"] or c["allow_new_policy_physical_success_claim"]
            or not c["unsupported_root_support_torque_native_contact_must_refuse_motor"]):
        raise ValueError("V135_CONTRACT_ASSUMPTION_DRIFT")
    return c


def weight_matrix() -> np.ndarray:
    if set(SYNERGY) != set(G1_JOINT_ORDER) or len(G1_JOINT_ORDER) != 29:
        raise ValueError("NOT_ALL_ORIGINAL29_SYNERGY_JOINTS")
    w = np.asarray([SYNERGY[j] for j in G1_JOINT_ORDER], dtype=float)
    if w.shape != (29, 4) or np.linalg.matrix_rank(w) != 4:
        raise ValueError("INVALID_COUPLED_BASIS")
    return w


def source_limits(path: Path, sha: str) -> dict:
    geom.require_source(path, sha)
    root = ET.parse(path).getroot()
    allowed = set(G1_JOINT_ORDER)
    limits = {}
    for joint in root.findall("joint"):
        if joint.get("name") in allowed:
            lim = joint.find("limit")
            if lim is None:
                raise ValueError("MISSING_ORIGINAL_G1_URDF_LIMIT")
            limits[joint.get("name")] = tuple(float(lim.get(k)) for k in (
                "lower", "upper", "velocity", "effort"))
    if set(limits) != allowed:
        raise ValueError("NOT_ORIGINAL_G1_29_URDF_JOINTS")
    matrix = np.asarray([limits[j] for j in G1_JOINT_ORDER])
    if (not np.isfinite(matrix).all() or np.any(matrix[:, 0] >= matrix[:, 1])
            or np.any(matrix[:, 2:] <= 0)):
        raise ValueError("INVALID_ORIGINAL_LIMITS")
    return dict(lo=matrix[:, 0], hi=matrix[:, 1],
                vmax=matrix[:, 2], taumax=matrix[:, 3])


@dataclass(frozen=True)
class Proposal:
    t: np.ndarray
    phases: tuple[str, ...]
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    latent: np.ndarray
    multiplier: float
    source: str = "ENGINEERING_SYNTHETIC_NOT_HUMAN"
    native_pair_certified: bool = False
    human_instep_contact_truth: bool = False
    torque_or_grf_available: bool = False
    motor_execution_authorized: bool = False


def smooth(u: np.ndarray):
    if np.any((u < 0) | (u > 1)) or not np.isfinite(u).all():
        raise ValueError("INVALID_PHASE_U")
    return (6*u**5 - 15*u**4 + 10*u**3,
            30*u**4 - 60*u**3 + 30*u**2,
            120*u**3 - 180*u**2 + 60*u)


def generate(cfg: dict, q0: np.ndarray, mult: float) -> Proposal:
    if isinstance(mult, bool) or mult not in cfg["multipliers"]:
        raise ValueError("UNFROZEN_MULTIPLIER")
    q0 = np.asarray(q0, dtype=float)
    if q0.shape != (29,) or not np.isfinite(q0).all():
        raise ValueError("INVALID_SYNTHETIC_ORIGINAL_Q0")
    w = weight_matrix()
    a = np.asarray(cfg["latent_amplitudes_rad"], dtype=float) * mult
    dt = cfg["dt_s"]
    total = round(cfg["horizon_s"] / dt)
    if total != 2800:
        raise ValueError("UNFROZEN_14S_200HZ")
    t = np.arange(total, dtype=float)*dt
    q = np.empty((total, 29))
    dq = np.empty_like(q)
    ddq = np.empty_like(q)
    latent = np.empty((total, 4))
    phase_labels = []
    i = 0
    for k, (name, duration) in enumerate(zip(
            cfg["all_phases"], cfg["phase_durations_s"])):
        count = round(duration / dt)
        if count * dt != duration:
            raise ValueError("NONINTEGER_FROZEN_PHASE")
        u = np.arange(count, dtype=float) / count
        v, vd, vdd = smooth(u)
        delta = KNOTS[k+1] - KNOTS[k]
        z = KNOTS[k] + v[:, None] * delta
        dz = vd[:, None] * delta / duration
        d2z = vdd[:, None] * delta / (duration ** 2)
        sl = slice(i, i+count)
        q[sl] = q0 + (z*a) @ w.T
        dq[sl] = (dz*a) @ w.T
        ddq[sl] = (d2z*a) @ w.T
        latent[sl] = z
        phase_labels.extend([name]*count)
        i += count
    if i != 2800 or len(phase_labels) != i:
        raise ValueError("INCOMPLETE_PHASES")
    return Proposal(t, tuple(phase_labels), q, dq, ddq, latent, float(mult))


def inspect(proposal: Proposal, limits: dict, tree, cfg: dict) -> dict:
    q, dq, ddq = proposal.q, proposal.dq, proposal.ddq
    if (q.shape != (2800, 29) or dq.shape != q.shape
            or ddq.shape != q.shape or proposal.latent.shape != (2800, 4)
            or not all(np.isfinite(x).all() for x in (q, dq, ddq))
            or tuple(dict.fromkeys(proposal.phases)) != tuple(PHASES)):
        raise ValueError("INVALID_FULL29_SEVEN_PHASE_PROPOSAL")
    pos_bad = (q < limits["lo"]) | (q > limits["hi"])
    speed_ratio = abs(dq)/limits["vmax"]
    parts = {}
    for group, prefixes in GROUPS.items():
        idx = [i for i, j in enumerate(G1_JOINT_ORDER)
               if j.startswith(prefixes)]
        parts[group] = float(np.max(abs(q[:, idx]-q[0, idx])))
    if sum(len([j for j in G1_JOINT_ORDER if j.startswith(pfx)])
           for pfx in GROUPS.values()) != 29:
        raise ValueError("GROUP_PARTITION_INCOMPLETE")
    gaps = geom.clearance_for_episode(tree, q, cfg["geometry_pairs"])
    collision = {
        k: dict(min_lower_bound_m=float(g.min()),
                below_0mm_frames=int(np.count_nonzero(g <= 0)),
                below_frozen_20mm_frames=int(np.count_nonzero(g <= .020)))
        for k, g in gaps.items()
    }
    consistency = float(np.max(abs(
        np.diff(q, axis=0)/cfg["dt_s"] -
        (dq[1:]+dq[:-1])*.5)))
    reasons = ["UNKNOWN_NATIVE_PHYSX_ACTOR_COLLIDER_PAIR",
               "NO_REAL_STANCE_GRF_COP_OR_SLIP",
               "NO_MEASURED_FLOATING_BASE_COM",
               "NO_ACTUATOR_TORQUE_OR_INVERSE_DYNAMICS",
               "NO_AUTHENTIC_HIGH_SKILL_HUMAN_INSTEP_TIMING",
               "NO_TRUE_BALL_SPEED_INSTEP_IMPULSE_OR_GOAL",
               "NO_REAL_14S_POST_KICK_RECOVERY"]
    if int(np.count_nonzero(pos_bad)):
        reasons.append("ORIGINAL29_POSITION_LIMIT_EXCEEDED")
    if int(np.count_nonzero(speed_ratio > 1)):
        reasons.append("ORIGINAL29_QDOT_LIMIT_EXCEEDED")
    if any(x["below_frozen_20mm_frames"] for x in collision.values()):
        reasons.append("ORIGINAL_URDF_COLLIDER_PROXIMITY_REVIEW_REQUIRED")
    return dict(
        multiplier=proposal.multiplier,
        source=proposal.source,
        proposal_200hz_frames=2800,
        true_physx_rollouts=0,
        all_seven_phases=True,
        all_original_29_joints=True,
        max_abs_joint_rad=float(np.max(abs(q))),
        position_limit_violation_frames=int(np.count_nonzero(np.any(pos_bad, axis=1))),
        velocity_limit_violation_frames=int(np.count_nonzero(
            np.any(speed_ratio > 1, axis=1))),
        max_original_qdot_limit_ratio=float(np.max(speed_ratio)),
        peak_analytic_qddot_rad_s2=float(np.max(abs(ddq))),
        derivative_consistency_max_rad_s=consistency,
        max_joint_step_change_rad=float(np.max(abs(np.diff(q, axis=0)))),
        group_excursion_rad=parts,
        all_groups_move_above_1mrad=(all(v > .001 for v in parts.values())),
        original_urdf_four_pair_geometry=collision,
        unique_refusal_reasons=sorted(set(reasons)),
        native_contact_pair_certified=False,
        original29_motor_execution_authorized=False,
        full_kick_strict_physics_pass=False,
        is_human_or_grf_truth=False,
    )


def run_frozen_grid(urdf: Path) -> dict:
    cfg = frozen()
    limits = source_limits(urdf, cfg["original_urdf_sha256"])
    q0 = (limits["lo"]+limits["hi"])*.5
    tree = geom.parse_tree(urdf)
    rows = [inspect(generate(cfg, q0, m), limits, tree, cfg)
            for m in cfg["multipliers"]]
    if len(rows) != 5 or any(r["original29_motor_execution_authorized"]
                              or r["true_physx_rollouts"] for r in rows):
        raise RuntimeError("MOTOR_OR_FALSE_PHYSICS_GATE_BREACH")
    return dict(version="v0.1.135", parent_sha=cfg["parent_sha"],
                frozen_protocol_sha=FROZEN_SHA,
                original_urdf_sha256=cfg["original_urdf_sha256"],
                source_q0="synthetic_joint_limit_midpoint_not_balanced_stance",
                trials=rows, actual_new_physx_episodes=0,
                actual_new_policy_strict_successes=0,
                actual_new_policy_success_rate="N/A: NO_PHYSX_ROLLOUTS",
                no_human_expert_instest_training_data=True,
                no_motor_commands_issued=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    result = run_frozen_grid(args.urdf)
    with args.output.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2, allow_nan=False, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "proposals": len(result["trials"]),
        "proposal_frames": sum(x["proposal_200hz_frames"]
                               for x in result["trials"]),
        "strict_motor_authorized": sum(
            x["original29_motor_execution_authorized"]
            for x in result["trials"]),
        "new_real_physx": 0,
        "out": str(args.output),
    }))


if __name__ == "__main__":
    main()
