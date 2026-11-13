#!/usr/bin/env python3
"""v137: pre-registered *instantaneous* 29DoF support-relative reachability.

Pure CPU linear programming; NEVER call as an actuator controller. 'Support'
means a fictitious translating root locks a LEFT ANKLE LINK ORIGIN, not a true
planted contact patch, frictional support, base orientation or measured GRF.
Right ankle link origin is not the G1 instep. No real football is modelled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linprog

import v127_collider_proximity as kin
import v135_chain29_offline as chain
import v136_g1_taskspace_mass_com as task
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v137_frozen_support_relative_velocity_lp.json"
FROZEN_SHA = "58231826544394e891f7c0ae08a04ba7f41f4ea6"
URDF = task.URDF
ARM = np.array(task._arm_indices(), dtype=int)
SOLVER_TOL = 1.0e-6


def frozen() -> dict:
    if CONFIG.resolve() != (ROOT / "configs/v137_frozen_support_relative_velocity_lp.json").resolve():
        raise ValueError("V137_PROTOCOL_PATH_DRIFT")
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", FROZEN_SHA + ":configs/v137_frozen_support_relative_velocity_lp.json"])
    if raw != CONFIG.read_bytes():
        raise ValueError("V137_PROTOCOL_DRIFT")
    c = json.loads(raw)
    expected = {
        "version": "v0.1.137",
        "parent_sha": "4cc754c1230e488b8ec81526f14b96125596fd22",
        "urdf_sha256": "34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761",
        "original_dof": 29, "dt_s": .005, "horizon_s": 14,
        "multipliers": [0, 1, 2, 4, -1],
        "probe_times_s": [2.25, 2.95, 3.6],
        "virtual_left_anchor_link": "left_ankle_roll_link",
        "relative_swing_link": "right_ankle_roll_link",
        "swing_relative_link_origin_velocity_targets_x_mps": [0, .5, 1, 3],
        "com_world_relative_to_virtual_stance_velocity_cap_mps": [None, .1],
        "arm_joint_velocity_deviation_cap_rad_s": [None, .25],
        "finite_difference_step_rad": 1.e-6,
        "jacobian_crosscheck_max_abs": 2.e-5,
        "linear_program_solver": "scipy.optimize.linprog(method='highs')",
    }
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError("V137_FROZEN_GRID_DRIFT")
    if (not c.get("original29_urdf_joint_velocity_hard_bounds")
            or not c.get("report_feasibility_not_execution")
            or not c.get("forbid_motor_execution")
            or not c.get("forbid_claim_human_learning")
            or not c.get("forbid_claim_new_physx_success")
            or c["strict_kick_gates"]["instep_impulse_fraction_min"] != .95):
        raise ValueError("V137_UNSAFE_PROTOCOL")
    if len(G1_JOINT_ORDER) != 29 or len(ARM) != 14:
        raise ValueError("NOT_ORIGINAL29_ARM_PARTITION")
    return c


def original_kinematics(urdf: Path, cfg: dict):
    kin.require_source(urdf, cfg["urdf_sha256"])
    return (kin.parse_tree(urdf),
            chain.source_limits(urdf, cfg["urdf_sha256"]))


def inertial_com_and_analytic_jacobian(tree: dict, masses: list,
                                       q: np.ndarray):
    """Mass-weighted point Jacobian including each inertial origin lever arm.

    J_point = J_link_origin + omega_j cross (R_link @ inertial_xyz),
    where omega_j is the upstream hinge axis in root coordinates.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (29,) or not np.isfinite(q).all():
        raise ValueError("BAD_ORIGINAL29_Q")
    if set(G1_JOINT_ORDER) != set(chain.SYNERGY):
        raise ValueError("JOINT_ORDER_DRIFT")
    target_links = {name for name, _, _ in masses}
    target_links.update(edge["parent"] for edge in tree["joints"])
    all_poses = kin.fk_targets(tree, q[None, :], target_links)
    axes = {}
    index = {name: i for i, name in enumerate(G1_JOINT_ORDER)}
    for edge in tree["joints"]:
        if edge["type"] == "revolute":
            axis = np.asarray(edge["axis"], dtype=float)
            if not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0:
                raise ValueError("INVALID_JOINT_AXIS")
            Rp = all_poses[edge["parent"]][0][0]
            axes[edge["name"]] = Rp @ edge["R"] @ (axis / np.linalg.norm(axis))
    weighted = np.zeros((3, 29), dtype=float)
    com = np.zeros(3, dtype=float)
    total_mass = 0.0
    for name, m, local in masses:
        if not np.isfinite(m) or m <= 0:
            raise ValueError("BAD_INERTIAL_MASS")
        R, p = all_poses[name]
        lever = R[0] @ np.asarray(local, dtype=float)
        com += m * (p[0] + lever)
        _, J = task.point_jacobian(tree, q, name)
        J = J.copy()
        for joint in task.ancestry(tree, name):
            J[:, index[joint]] += np.cross(axes[joint], lever)
        weighted += m * J
        total_mass += m
    if total_mass <= 0 or not np.isfinite(weighted).all():
        raise ValueError("INVALID_MASS_OR_JACOBIAN")
    return com / total_mass, weighted / total_mass


def independent_fd_com_jacobian(tree: dict, masses: list,
                                q: np.ndarray, eps: float):
    """Independent batched COM positions, not the analytic derivative above."""
    if not (np.isfinite(eps) and 0 < eps < .001):
        raise ValueError("BAD_FD_EPS")
    q = np.asarray(q, dtype=float)
    if q.shape != (29,) or not np.isfinite(q).all():
        raise ValueError("BAD_ORIGINAL29_Q")
    perturb = np.eye(29) * eps
    probe = np.concatenate([q[None, :] + perturb,
                            q[None, :] - perturb])
    pose = kin.fk_targets(tree, probe, {name for name, _, _ in masses})
    com = task.mass_weighted_com(pose, masses)
    return ((com[:29] - com[29:]) / (2 * eps)).T


def problem_at_pose(tree: dict, masses: list, q: np.ndarray, cfg: dict):
    left, Jl = task.point_jacobian(tree, q, cfg["virtual_left_anchor_link"])
    right, Jr = task.point_jacobian(tree, q, cfg["relative_swing_link"])
    com, Jc = inertial_com_and_analytic_jacobian(tree, masses, q)
    fd = independent_fd_com_jacobian(
        tree, masses, q, cfg["finite_difference_step_rad"])
    error = float(np.max(np.abs(fd - Jc)))
    if error > cfg["jacobian_crosscheck_max_abs"]:
        raise ValueError("MASS_COM_JACOBIAN_FD_DISAGREEMENT")
    jr = Jr - Jl
    jc = Jc - Jl
    if not np.isfinite(jr).all() or not np.isfinite(jc).all():
        raise ValueError("BAD_RELATIVE_TASKSPACE_JACOBIAN")
    return dict(Jleft=Jl, Jrelative=jr, Jcom_relative=jc,
                com_xyz=com, ankle_delta_xyz=right-left,
                com_jacobian_fd_max_error=error)


def constraints(qref: np.ndarray, limits: dict, case: dict, cfg: dict):
    qref = np.asarray(qref, dtype=float)
    vmax = np.asarray(limits["vmax"], dtype=float)
    if (qref.shape != (29,) or vmax.shape != (29,)
            or not np.isfinite(qref).all() or not np.isfinite(vmax).all()
            or np.any(vmax <= 0)):
        raise ValueError("BAD_ORIGINAL29_QDOT_OR_LIMITS")
    cap_arm = case["arm_cap"]
    cap_com = case["com_cap"]
    if (cap_arm not in cfg["arm_joint_velocity_deviation_cap_rad_s"]
            or cap_com not in cfg["com_world_relative_to_virtual_stance_velocity_cap_mps"]):
        raise ValueError("NOT_FROZEN_CONSTRAINT")
    lo, hi = -vmax.copy(), vmax.copy()
    if cap_arm is not None:
        lo[ARM] = np.maximum(lo[ARM], qref[ARM] - cap_arm)
        hi[ARM] = np.minimum(hi[ARM], qref[ARM] + cap_arm)
    if np.any(lo > hi):
        raise ValueError("INCONSISTENT_29DOF_HARD_BOX")
    a_ub = np.concatenate([case["Jcom_relative"], -case["Jcom_relative"]]) if cap_com is not None else None
    b_ub = np.full(6, cap_com, dtype=float) if cap_com is not None else None
    return list(zip(lo, hi)), a_ub, b_ub


def _lp(c: np.ndarray, Aeq: np.ndarray, beq: np.ndarray,
        bounds: list, Aub=None, bub=None):
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                bounds=bounds, method="highs")
    if r.status not in (0, 2):
        raise RuntimeError("HARD_LP_UNRESOLVED_STATUS:" + str(r.status))
    if r.status == 2:
        return None
    if r.x is None or not np.isfinite(r.x).all():
        raise RuntimeError("HARD_LP_NONFINITE_WITNESS")
    x = np.asarray(r.x, dtype=float)
    lo = np.array([v[0] for v in bounds])
    hi = np.array([v[1] for v in bounds])
    bound_err = max(float(np.max(lo-x)), float(np.max(x-hi)), 0.)
    eq_err = float(np.max(np.abs(Aeq @ x - beq)))
    ub_err = float(np.max(Aub@x-bub)) if Aub is not None else 0.
    if max(bound_err, eq_err, ub_err) > SOLVER_TOL:
        raise RuntimeError("HARD_LP_FALSE_FEASIBLE_WITNESS")
    return x


def reachability_case(problem: dict, qref: np.ndarray, limits: dict,
                      cfg: dict, com_cap, arm_cap) -> dict:
    case = dict(problem, com_cap=com_cap, arm_cap=arm_cap)
    bounds, Aub, bub = constraints(qref, limits, case, cfg)
    J = problem["Jrelative"]
    # +x extrema under strictly zero relative lateral and vertical velocity.
    yz = J[1:3]
    zeros = np.zeros(2)
    maximum = _lp(-J[0], yz, zeros, bounds, Aub, bub)
    minimum = _lp( J[0], yz, zeros, bounds, Aub, bub)
    if maximum is None or minimum is None:
        raise RuntimeError("NO_BASE_FEASIBLE_RELATIVE_ANKLE_VELOCITY")
    extrema = [float(J[0]@minimum), float(J[0]@maximum)]
    if extrema[0] > extrema[1] + 1e-7:
        raise RuntimeError("INVALID_LP_EXTREMA")
    rows = []
    for target in cfg["swing_relative_link_origin_velocity_targets_x_mps"]:
        if target not in cfg["swing_relative_link_origin_velocity_targets_x_mps"]:
            raise ValueError("NOT_FROZEN_TARGET")
        v = _lp(np.zeros(29), J, np.array([target, 0., 0.]),
                bounds, Aub, bub)
        row = dict(target_ankle_origin_relative_x_mps=float(target),
                   mathematically_feasible=v is not None,
                   is_physics_verified=False,
                   motor_execution_authorized=False)
        if v is not None:
            row.update(
                max_qdot_limit_ratio=float(np.max(abs(v)/limits["vmax"])),
                max_change_from_synthetic_dq_rad_s=float(np.max(abs(v-qref))),
                hypothetical_5ms_qddot_rad_s2=float(np.max(abs(v-qref))/.005),
                com_relative_velocity_root_xyz_mps=(
                    problem["Jcom_relative"]@v).tolist(),
                virtual_root_translation_rate_mps=(
                    -problem["Jleft"]@v).tolist(),
                right_ankle_relative_velocity_mps=(J@v).tolist(),
                witness_norm_qdot=float(np.linalg.norm(v)),
            )
        rows.append(row)
    return dict(
        com_velocity_cap_mps=com_cap,
        arm_joint_velocity_deviation_cap_rad_s=arm_cap,
        min_attainable_relative_x_mps=extrema[0],
        max_attainable_relative_x_mps=extrema[1],
        targets=rows,
        feasible_target_count=sum(x["mathematically_feasible"] for x in rows),
        original29_joint_velocities_hard_bounded=True,
        exact_relative_yz_zero=True,
        virtual_left_ankle_origin_anchored_by_translation_only=True,
        wholebody_real_grf_torque_and_contact_unauthenticated=True,
        human_instep_force_learning_not_attested=True,
        motor_execution_authorized=False,
        native_physx_rollouts=0,
        strict_kick_success=False,
    )


def run(urdf: Path = URDF):
    cfg = frozen()
    tree, limits = original_kinematics(urdf, cfg)
    mass = task.mass_inertials(urdf, tree)
    parent = chain.frozen()
    if parent["multipliers"] != cfg["multipliers"]:
        raise ValueError("V135_PARENT_MULTIPLIERS_DRIFT")
    q0 = (limits["lo"] + limits["hi"])/2.
    results = []
    for mult in cfg["multipliers"]:
        p = chain.generate(parent, q0, mult)
        for t in cfg["probe_times_s"]:
            idx = round(t/cfg["dt_s"])
            if idx <= 0 or idx >= len(p.q) or abs(p.t[idx]-t)>1e-9:
                raise ValueError("FROZEN_PROBE_TIME_MISSING")
            problem = problem_at_pose(tree,mass,p.q[idx],cfg)
            for com_cap in cfg["com_world_relative_to_virtual_stance_velocity_cap_mps"]:
                for arm_cap in cfg["arm_joint_velocity_deviation_cap_rad_s"]:
                    row = reachability_case(problem,p.dq[idx],limits,cfg,com_cap,arm_cap)
                    row.update(multiplier=float(mult), synthetic_time_s=float(t),
                               com_jacobian_fd_max_error=problem["com_jacobian_fd_max_error"])
                    results.append(row)
    if len(results)!=60 or sum(len(x["targets"]) for x in results)!=240:
        raise RuntimeError("MISSING_FROZEN_GRID")
    if any(r["strict_kick_success"] or r["motor_execution_authorized"] for r in results):
        raise RuntimeError("FALSE_PHYSICAL_CLAIM")
    return dict(version="v0.1.137",
                parent_sha=cfg["parent_sha"], protocol_commit_sha=FROZEN_SHA,
                urdf_sha256=cfg["urdf_sha256"], original_robot_dof=29,
                frozen_synthetic_ref_proposals=5,
                frozen_probe_pose_conditions=15, hard_lp_envelopes=60,
                exact_target_feasibility_decisions=240,
                max_com_jacobian_fd_error=float(max(r["com_jacobian_fd_max_error"] for r in results)),
                original_urdf_inertial_mass_kg=float(sum(x[1] for x in mass)),
                new_true_physx_rollouts=0,
                original_urdf_ankle_origin_is_not_true_instep=True,
                fake_base_anchor_is_not_measured_ground_support=True,
                learned_professional_human_motion=False,
                actual_ball_kick_speed_and_impulse_verified=False,
                motor_execution_authorized=False,
                actual_strict_physx_successes=0,
                new_strict_physx_success_rate="N/A: NO NEW PHYSX ROLLOUTS",
                cases=results)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--urdf",type=Path,default=URDF)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=run(args.urdf)
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "hard_lp_envelopes":result["hard_lp_envelopes"],
        "decisions":result["exact_target_feasibility_decisions"],
        "feasible":sum(t["mathematically_feasible"] for row in result["cases"] for t in row["targets"]),
        "max_com_jac_fd_error":result["max_com_jacobian_fd_error"],
        "new_true_physx_rollouts":0, "motor_execution_authorized":False,
        "output":str(args.output)},indent=2))


if __name__=="__main__":
    main()
