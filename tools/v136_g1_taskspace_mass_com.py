#!/usr/bin/env python3
"""v136: frozen 29DoF G1 task-space feasibility and inertial COM audit (CPU).

No control/simulation side effects. The right ankle LINK ORIGIN is NOT instep
contact geometry, its velocity is NOT ball speed, and an anchored ankle is
NOT a measured stance/GRF. All target poses are synthetic v135 references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import lsq_linear

import v127_collider_proximity as fk
import v135_chain29_offline as chain
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v136_frozen_taskspace_feasibility.json"
FROZEN_SHA = "bf79f52a771d1557687605630559405b7376e9b0"
URDF = Path("/home/you/projects/kinetickick/configs/paid_original/paid_main_resolved.urdf")


def frozen() -> dict:
    if CONFIG.resolve() != (ROOT / "configs/v136_frozen_taskspace_feasibility.json").resolve():
        raise ValueError("V136_PROTOCOL_DRIFT")
    data = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", FROZEN_SHA + ":" + str(CONFIG.relative_to(ROOT))]
    )
    if data != CONFIG.read_bytes():
        raise ValueError("V136_PROTOCOL_DRIFT")
    c = json.loads(data)
    if (c["version"] != "v0.1.136"
            or c["parent_sha"] != "1e6350cb3e83d2acaecce6cebf2426de23e5b691"
            or c["sample_period_s"] != .005 or c["horizon_s"] != 14.
            or c["variant_multipliers"] != [0, 1, 2, 4, -1]
            or c["swing_link_origin_velocity_targets_mps"] != [0, .5, 1, 3]
            or c["precontact_probe_time_s"] != 2.95
            or c["dof"] != 29 or not c["no_gpu_physx_while_v120_occupied"]
            or not c["no_kick_success_claim"]
            or not c["original29_urdf_qdot_hard_box"]
            or c["strict_kick_gates"]["instep_impulse_fraction_min"] != .95):
        raise ValueError("V136_UNFROZEN_ASSUMPTIONS")
    return c


def require_original(urdf: Path, cfg: dict) -> None:
    fk.require_source(urdf, cfg["urdf_sha256"])
    if set(G1_JOINT_ORDER) != set(chain.SYNERGY) or len(G1_JOINT_ORDER) != 29:
        raise ValueError("ORIGINAL_29DOF_JOINT_SET_DRIFT")


def mass_inertials(urdf: Path, tree: dict) -> list[tuple[str, float, np.ndarray]]:
    """Original inertial COM points. Massless sensor attachments are excluded."""
    root = ET.parse(urdf).getroot()
    rows = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_node = inertial.find("mass")
        if mass_node is None:
            raise ValueError("UNSPECIFIED_URDF_MASS")
        m = float(mass_node.get("value"))
        if not np.isfinite(m) or m <= 0:
            raise ValueError("INVALID_URDF_MASS")
        origin = inertial.find("origin")
        v = fk.numbers(origin.get("xyz") if origin is not None else None, "0 0 0")
        name = link.get("name")
        if name not in tree["links"]:
            raise ValueError("UNKNOWN_INERTIAL_LINK")
        rows.append((name, m, v))
    if len(rows) < 30 or len({r[0] for r in rows}) != len(rows):
        raise ValueError("INCOMPLETE_G1_INERTIAL_MODEL")
    return rows


def mass_weighted_com(poses: dict, inertials: list) -> np.ndarray:
    """N x 3 COM in pelvis/root frame; never claims measured world COM."""
    n = next(iter(poses.values()))[1].shape[0]
    mass = sum(row[1] for row in inertials)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("BAD_TOTAL_MASS")
    result = np.zeros((n, 3), dtype=float)
    for link, m, local in inertials:
        R, pos = poses[link]
        result += m * (pos + np.einsum("nij,j->ni", R, local))
    return result / mass


def ancestry(tree: dict, target: str) -> set[str]:
    by_child = {j["child"]: j for j in tree["joints"]}
    if target not in tree["links"]:
        raise ValueError("MISSING_TARGET_LINK")
    names = set()
    current = target
    while current != tree["root"]:
        if current not in by_child:
            raise ValueError("DISCONNECTED_TARGET_CHAIN")
        edge = by_child[current]
        if edge["type"] == "revolute":
            names.add(edge["name"])
        current = edge["parent"]
    return names


def point_jacobian(tree: dict, q: np.ndarray, target: str):
    """Analytic 3x29 root-fixed positional Jacobian for link origin."""
    q = np.asarray(q, dtype=float)
    if q.shape != (29,) or not np.isfinite(q).all():
        raise ValueError("INVALID_Q_ORIGINAL29")
    if target not in tree["links"]:
        raise ValueError("UNKNOWN_KINEMATIC_LINK")
    pos = fk.fk_targets(tree, q[None, :], {target})[target][1][0]
    axes = {}
    origin_by_name = {}
    root = tree["root"]
    transforms = {root: (np.eye(3), np.zeros(3))}
    index = {name: i for i, name in enumerate(G1_JOINT_ORDER)}
    for edge in tree["joints"]:
        Rp, pp = transforms[edge["parent"]]
        pivot = pp + Rp @ edge["xyz"]
        R0 = Rp @ edge["R"]
        R = R0
        if edge["type"] == "revolute":
            axis = R0 @ (edge["axis"] / np.linalg.norm(edge["axis"]))
            axes[edge["name"]] = axis
            origin_by_name[edge["name"]] = pivot
            R = R0 @ fk.axis_rotation(edge["axis"],
                                     np.array([q[index[edge["name"]]]]))[0]
        transforms[edge["child"]] = (R, pivot)
    if not np.allclose(transforms[target][1], pos, atol=1e-10):
        raise ValueError("FK_POSE_DISAGREEMENT")
    active = ancestry(tree, target)
    J = np.zeros((3, 29), dtype=float)
    for joint in active:
        J[:, index[joint]] = np.cross(axes[joint], pos - origin_by_name[joint])
    return pos, J


def fd_jacobian(tree: dict, q: np.ndarray, link: str, eps: float):
    if not np.isfinite(eps) or eps <= 0 or eps >= .001:
        raise ValueError("INVALID_FINITE_DIFFERENCE_STEP")
    q = np.asarray(q, dtype=float)
    if q.shape != (29,) or not np.isfinite(q).all():
        raise ValueError("INVALID_Q_ORIGINAL29")
    d = np.eye(29)*eps
    both = np.concatenate([q[None, :] + d, q[None, :] - d])
    points = fk.fk_targets(tree, both, {link})[link][1]
    return ((points[:29]-points[29:])/(2*eps)).T


def coupled_support_audit(proposal: chain.Proposal, tree: dict,
                          inertials: list, cfg: dict) -> dict:
    if proposal.q.shape != (2800, 29) or proposal.dq.shape != (2800, 29):
        raise ValueError("NOT_14S_ORIGINAL29")
    left, right = cfg["anchor_link"], cfg["swing_link"]
    targets = {left, right} | {x[0] for x in inertials}
    poses = fk.fk_targets(tree, proposal.q, targets)
    l = poses[left][1]
    r = poses[right][1]
    com_fixed = mass_weighted_com(poses, inertials)
    base_offset = l[0] - l     # fictitious pelvis translation, not a controller
    com_anchor = com_fixed + base_offset
    right_anchor = r + base_offset
    dt = cfg["sample_period_s"]
    i = round(cfg["precontact_probe_time_s"] / dt)
    if i <= 0 or i >= len(l)-1:
        raise ValueError("INVALID_PROBE_FRAME")
    def travel(p):
        return float(np.max(np.linalg.norm(p - p[0], axis=1)))
    def velocity(p):
        return (p[i+1] - p[i-1])/(2*dt)
    data = dict(
        multiplier=proposal.multiplier,
        reference_source=proposal.source,
        frame_count=len(proposal.q),
        fixed_root_left_ankle_origin_travel_max_m=travel(l),
        fixed_root_right_ankle_origin_travel_max_m=travel(r),
        fake_base_translation_to_lock_left_origin_max_m=travel(base_offset),
        fixed_root_inertial_com_travel_max_m=travel(com_fixed),
        left_ankle_translation_locked_inertial_com_travel_max_m=travel(com_anchor),
        right_ankle_origin_relative_to_left_travel_max_m=travel(r-l),
        left_ankle_locked_right_ankle_travel_max_m=travel(right_anchor),
        at_2p95s_root_fixed_left_origin_speed_mps=float(np.linalg.norm(velocity(l))),
        at_2p95s_right_ankle_relative_velocity_root_xyz_mps=velocity(r-l).tolist(),
        at_2p95s_right_origin_relative_speed_mps=float(np.linalg.norm(velocity(r-l))),
        at_2p95s_mass_com_fixed_root_xyz_m=com_fixed[i].tolist(),
        at_2p95s_mass_com_left_origin_locked_xyz_m=com_anchor[i].tolist(),
        original_urdf_total_inertial_mass_kg=float(sum(x[1] for x in inertials)),
        original_urdf_inertial_link_count=len(inertials),
        only_kinematic_root_hypotheses=True,
        actual_base_support_grf_cop_and_contact_unknown=True,
        original29_motor_execution_authorized=False,
        strict_physx_kick_success=False)
    return data


def _arm_indices():
    return [i for i, name in enumerate(G1_JOINT_ORDER)
            if "shoulder" in name or "elbow" in name or "wrist" in name]


def bounded_task_feasibility(proposal: chain.Proposal, limits: dict, tree: dict,
                             cfg: dict, target_x: float) -> dict:
    """CPU non-actuating instantaneous bounded least squares, NOT a controller."""
    if target_x not in cfg["swing_link_origin_velocity_targets_mps"]:
        raise ValueError("UNREGISTERED_TASK_TARGET")
    dt = cfg["sample_period_s"]
    i = round(cfg["precontact_probe_time_s"]/dt)
    q, dq = proposal.q[i], proposal.dq[i]
    _, Jl = point_jacobian(tree, q, cfg["anchor_link"])
    _, Jr = point_jacobian(tree, q, cfg["swing_link"])
    h = cfg["jacobian_fd_step_rad"]
    e_l = float(np.max(abs(Jl-fd_jacobian(tree, q, cfg["anchor_link"], h))))
    e_r = float(np.max(abs(Jr-fd_jacobian(tree, q, cfg["swing_link"], h))))
    if max(e_l, e_r) > cfg["jacobian_error_tolerance"]:
        raise ValueError("ANALYTIC_FD_JACOBIAN_MISMATCH")
    arm_mask = np.zeros(29)
    arm_mask[_arm_indices()] = 1
    A = np.vstack([
        np.sqrt(cfg["stance_task_weight"]) * Jl,
        np.sqrt(cfg["swingspeed_task_weight"]) * Jr,
        np.sqrt(cfg["posture_tracking_weight"])*np.eye(29),
        np.sqrt(cfg["arm_regularization_weight"])*np.diag(arm_mask),
    ])
    # Swing target is artificial root +X, not ball direction; other components zero.
    b = np.concatenate([
        np.zeros(3),
        np.sqrt(cfg["swingspeed_task_weight"]) * np.array([target_x, 0, 0]),
        np.sqrt(cfg["posture_tracking_weight"]) * dq,
        np.sqrt(cfg["arm_regularization_weight"]) * arm_mask*dq,
    ])
    vmax = np.asarray(limits["vmax"], dtype=float)
    if vmax.shape != (29,) or not np.isfinite(vmax).all() or np.any(vmax <= 0):
        raise ValueError("BAD_ORIGINAL29_VELOCITY_LIMITS")
    sol = lsq_linear(A, b, bounds=(-vmax, vmax), method="trf",
                     tol=1e-10, max_iter=1000)
    if not sol.success or not np.isfinite(sol.x).all():
        raise ValueError("TASKSPACE_OPTIMIZER_DID_NOT_CONVERGE")
    v = sol.x
    stance = Jl@v
    swing = Jr@v
    target = np.array([target_x, 0, 0])
    # No acceleration calibration, no motor torque, no physical stance.
    # Numerical solver feasible != execute, even if all residuals are small.
    return dict(
        multiplier=proposal.multiplier, target_ankle_origin_root_plus_x_mps=target_x,
        observed_ankle_origin_root_xyz_mps=swing.tolist(),
        observed_stance_ankle_origin_root_xyz_mps=stance.tolist(),
        swing_vector_residual_mps=float(np.linalg.norm(swing-target)),
        left_ankle_stance_residual_mps=float(np.linalg.norm(stance)),
        original29_qdot_max_ratio=float(np.max(abs(v)/vmax)),
        original29_qdot_limit_violation_count=int(np.count_nonzero(abs(v)>vmax+1e-8)),
        speed_task_warning=bool(np.linalg.norm(swing-target) >
                                cfg["swing_tracking_warning_mps"]),
        stance_task_warning=bool(np.linalg.norm(stance) >
                                 cfg["stance_residual_warning_mps"]),
        original_joint_reference_deviation_max_rad_s=float(np.max(abs(v-dq))),
        instantaneous_qddot_if_applied_in_5ms_max_rad_s2=float(np.max(abs(v-dq))/dt),
        max_analytic_vs_fd_jacobian_abs_error=max(e_l,e_r),
        solver_status=int(sol.status),
        solver_cost=float(sol.cost),
        physics_success_claim=False,
        native_collision_contact_certified=False,
        actual_ball_and_instep_witness=False,
        motor_execution_authorized=False)


def run(urdf: Path = URDF) -> dict:
    c = frozen()
    require_original(urdf, c)
    lim = chain.source_limits(urdf, c["urdf_sha256"])
    vcfg = chain.frozen()
    if vcfg["multipliers"] != c["variant_multipliers"]:
        raise ValueError("PARENT_EXPERIMENT_MATRIX_DRIFT")
    tree = fk.parse_tree(urdf)
    masses = mass_inertials(urdf, tree)
    q0 = .5*(lim["lo"] + lim["hi"])
    support_rows = []
    task_rows = []
    for scale in c["variant_multipliers"]:
        p = chain.generate(vcfg, q0, scale)
        support_rows.append(coupled_support_audit(p,tree,masses,c))
        for speed in c["swing_link_origin_velocity_targets_mps"]:
            task_rows.append(bounded_task_feasibility(p,lim,tree,c,speed))
    if len(support_rows) != 5 or len(task_rows) != 20:
        raise ValueError("INCOMPLETE_FROZEN_V136_GRID")
    if any(x["motor_execution_authorized"] or x["physics_success_claim"]
           for x in task_rows):
        raise ValueError("FALSE_REAL_PHYSICS_ATTESTATION")
    return dict(
        version="v0.1.136", frozen_sha=FROZEN_SHA,
        original_g1_urdf_sha256=c["urdf_sha256"],
        source_v135_sha=c["parent_sha"],
        original29_14s_200hz_reference_frames=14000,
        independent_synthetic_reference_variants=5,
        instantaneous_kinematic_task_tests=20,
        support_kinematics_rows=support_rows,
        frozen_taskspace_rows=task_rows,
        new_true_physx_rollouts=0,
        real_new_policy_successes=0,
        true_physical_success_rate="NOT APPLICABLE",
        foot_link_origin_is_not_true_instep=True,
        base_translation_is_not_real_stance_support=True,
        actual_motor_commands_issued=0,
        human_kicking_grf_training_data_claim=False,
        status="KINEMATIC_FEASIBILITY_ONLY_NO_MOTOR_PERMISSION")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, default=URDF)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = run(args.urdf)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "v136_fullbody_frames": result["original29_14s_200hz_reference_frames"],
        "taskspace_cpu_probes": result["instantaneous_kinematic_task_tests"],
        "new_physx_rollouts": 0, "output": str(args.output)
    }))


if __name__ == "__main__":
    main()
