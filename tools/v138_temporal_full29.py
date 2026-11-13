#!/usr/bin/env python3
"""v138 SHA-frozen, CPU-only 29DoF *kinematic* temporal counterexamples.

No motor commands, GRF, PhysX rollouts, genuine football instep or learned
human kick. A planted ankle LINK ORIGIN under a fictitious translating root
is not a stance foot. All failures are retained.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

import v127_collider_proximity as geom
import v135_chain29_offline as chain
import v136_g1_taskspace_mass_com as task
import v137_hard_support_relative_lp as lp

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v138_frozen_temporal_coordinated_29dof.json"
FROZEN_SHA = "c032c6f2de72ef0ace3d4e00834dd265164b98f0"
N = 29


def frozen() -> dict:
    if CONFIG.resolve() != (ROOT / "configs/v138_frozen_temporal_coordinated_29dof.json").resolve():
        raise ValueError("V138_CONFIG_PATH_DRIFT")
    b = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                 FROZEN_SHA + ":configs/v138_frozen_temporal_coordinated_29dof.json"])
    if b != CONFIG.read_bytes():
        raise ValueError("V138_CONFIG_BYTE_DRIFT")
    c = json.loads(b)
    expected = {
        "version": "v0.1.138",
        "parent_sha": "aabc122da8497583a8d1322185d7c8d8c96f7d70",
        "urdf_sha256": "34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761",
        "original_dof": 29, "sample_period_s": .005, "horizon_s": 14.,
        "contact_proxy_time_s": 2.95, "multipliers": [0, 1, 2, 4, -1],
        "relative_swing_ankle_origin_x_mps": [0, 1, 3],
        "compact_bump_halfwidth_s": [.10, .20, .40],
        "com_velocity_relative_to_virtual_anchor_xyz_cap_mps": .1,
        "arm_deviation_from_baseline_qdot_cap_rad_s": .25,
        "descriptive_acceleration_screen_rad_s2": 40.,
        "geometry_warning_m": .02, "kinematic_window_before_s": .4,
        "kinematic_window_after_s": 1.25,
        "constraint_tolerance": 2e-5, "opt_ftol": 1e-10,
        "opt_maxiter": 250,
    }
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError("V138_GRID_OR_GATE_DRIFT")
    if (not c["original29_qdot_hard_box"] or not c["mandatory_fail_closed"]
            or c["motor_execution_authorized"] or c["real_grf_available"]
            or c["native_pair_truth_available"]
            or c["professional_human_instep_learned"]
            or c["actual_new_physx_rollouts"] != 0
            or c["strict_real_physics_gates"]["instep_impulse_fraction_min"] != .95):
        raise ValueError("V138_UNSAFE_PHYSICS_CLAIM")
    return c


def basis(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact compact C2 bump h, h', h''; all three vanish at +/-1."""
    s = np.asarray(s, dtype=float)
    if not np.isfinite(s).all():
        raise ValueError("NONFINITE_LOCAL_TIME")
    h = np.zeros_like(s)
    dh = np.zeros_like(s)
    d2h = np.zeros_like(s)
    mask = np.abs(s) < 1
    x = s[mask]
    h[mask] = x * (1-x*x)**4
    dh[mask] = (1-x*x)**3 * (1-9*x*x)
    d2h[mask] = -24*x*(1-x*x)**2*(1-3*x*x)
    return h, dh, d2h


def min_change_witness(problem: dict, baseline_velocity: np.ndarray,
                       limits: dict, c: dict, parent: dict, target: float) -> dict:
    """SLSQP min weighted quadratic qdot departure, with hard G1 velocity box."""
    vref = np.asarray(baseline_velocity, float)
    if vref.shape != (N,) or not np.isfinite(vref).all():
        raise ValueError("BAD_29DOF_BASELINE_VELOCITY")
    if isinstance(target, bool) or target not in c["relative_swing_ankle_origin_x_mps"]:
        raise ValueError("UNREGISTERED_VELOCITY_TARGET")
    case = dict(problem, com_cap=c["com_velocity_relative_to_virtual_anchor_xyz_cap_mps"],
                arm_cap=c["arm_deviation_from_baseline_qdot_cap_rad_s"])
    bounds, A, b = lp.constraints(vref, limits, case, parent)
    J = problem["Jrelative"]
    desired = np.array([target, 0., 0.], dtype=float)
    start = lp._lp(np.zeros(N), J, desired, bounds, A, b)
    if start is None:
        return dict(success=False, reason="EXACT_HARD_VELOCITY_INFEASIBLE")
    lo = np.array([x[0] for x in bounds])
    hi = np.array([x[1] for x in bounds])
    scale = np.maximum(limits["vmax"], 1e-6)
    def obj(v):
        d = (v-vref)/scale
        return float(.5 * d@d)
    def jac(v):
        return (v-vref)/scale**2
    eq = LinearConstraint(J, desired, desired)
    inequalities = LinearConstraint(A, np.full(6, -np.inf), b)
    result = minimize(obj, start, jac=jac, method="SLSQP",
                      bounds=Bounds(lo, hi), constraints=[eq, inequalities],
                      options={"ftol": c["opt_ftol"], "maxiter": c["opt_maxiter"]})
    x = np.asarray(result.x, dtype=float) if result.x is not None else None
    if x is None or x.shape != (N,) or not np.isfinite(x).all():
        return dict(success=False, reason="SLSQP_NO_FINITE_QDOT", solver_status=str(result.message))
    residual = float(np.max(abs(J @ x-desired)))
    bound_violation = max(0., float(np.max(lo-x)), float(np.max(x-hi)))
    com_violation = max(0., float(np.max(A @ x-b)))
    if max(residual, bound_violation, com_violation) > c["constraint_tolerance"]:
        return dict(success=False, reason="SLSQP_FALSE_FEASIBLE_REJECTED",
                    equality_error_mps=residual, box_error_rad_s=bound_violation,
                    com_inequality_error_mps=com_violation,
                    solver_status=str(result.message))
    if not result.success:
        return dict(success=False, reason="SLSQP_NOT_CONVERGED",
                    solver_status=str(result.message))
    return dict(success=True, qdot=x, weighted_departure_obj=obj(x),
                max_joint_velocity_ratio=float(np.max(abs(x)/limits["vmax"])),
                instant_velocity_change_rad_s=float(np.max(abs(x-vref))),
                maximum_capped_com_velocity_mps=float(np.max(abs(problem["Jcom_relative"]@x))),
                relative_ankle_origin_velocity_mps=(J@x).tolist(),
                equality_error_mps=residual)


def temporal_bridge(p: chain.Proposal, c: dict, width: float,
                    target_velocity: np.ndarray):
    """14s analytic q,dq,ddq at 200 Hz, with exact center velocity and C2 edges."""
    if isinstance(width, bool) or width not in c["compact_bump_halfwidth_s"]:
        raise ValueError("UNFROZEN_C2_HALFWIDTH")
    v = np.asarray(target_velocity, float)
    if v.shape != (N,) or not np.isfinite(v).all():
        raise ValueError("BAD_TARGET_ORIGINAL29_VELOCITY")
    if p.q.shape != (2800, N) or p.dq.shape != p.q.shape or p.ddq.shape != p.q.shape:
        raise ValueError("NOT_14S_200HZ_ORIGINAL29")
    ci = round(c["contact_proxy_time_s"]/c["sample_period_s"])
    if abs(p.t[ci]-c["contact_proxy_time_s"]) > 1e-10:
        raise ValueError("NO_SYNTHETIC_CONTACT_CLOCK")
    dv = v-p.dq[ci]
    s = (p.t-c["contact_proxy_time_s"])/width
    h, dh, d2h = basis(s)
    q = p.q + width*h[:, None]*dv[None, :]
    dq = p.dq + dh[:, None]*dv[None, :]
    ddq = p.ddq + (d2h/width)[:, None]*dv[None, :]
    if not np.isfinite(q).all() or not np.isfinite(dq).all() or not np.isfinite(ddq).all():
        raise RuntimeError("NONFINITE_TEMPORAL_BRIDGE")
    if np.max(abs(q[ci]-p.q[ci])) > 1e-12 or np.max(abs(dq[ci]-v)) > 1e-10:
        raise RuntimeError("FAILED_CENTER_VELOCITY_INTERPOLATION")
    outside = np.abs(s) >= 1
    if (not np.array_equal(q[outside], p.q[outside]) or
            not np.array_equal(dq[outside], p.dq[outside]) or
            not np.array_equal(ddq[outside], p.ddq[outside])):
        raise RuntimeError("COMPACT_SUPPORT_BREACH")
    return q, dq, ddq


def kinematic_window(tree, masses, fullq: np.ndarray, c: dict, pairs):
    t = np.arange(2800)*c["sample_period_s"]
    focus = (t >= c["contact_proxy_time_s"]-c["kinematic_window_before_s"]-1e-8) & (
        t <= c["contact_proxy_time_s"]+c["kinematic_window_after_s"]+1e-8)
    q = fullq[focus]
    links = {n for n, _, _ in masses} | {c["virtual_anchor_link"], c["swing_link"]}
    poses = geom.fk_targets(tree, q, links)
    left = poses[c["virtual_anchor_link"]][1]
    right = poses[c["swing_link"]][1]
    com = task.mass_weighted_com(poses, masses)
    # These are only coordinates relative to a virtual translating ankle origin.
    gaps = geom.clearance_for_episode(tree, q, pairs)
    return dict(
        samples=int(q.shape[0]),
        com_minus_ankle_m=com-left,
        right_minus_left_ankle_m=right-left,
        minimum_urdf_capsule_margin_m=float(min(np.min(x) for x in gaps.values())),
        count_urdf_proximity_warning_frames=int(np.count_nonzero(
            np.any(np.array([x < c["geometry_warning_m"] for x in gaps.values()]), axis=0))),
        per_pair_minimum_capsule_separation_m={
            name: float(np.min(values)) for name, values in gaps.items()})


def evaluate(p: chain.Proposal, vstar: np.ndarray, tree, masses, lim: dict,
             c: dict, width: float, baseline_window: dict):
    q, dq, ddq = temporal_bridge(p, c, width, vstar)
    bad_position = np.any((q < lim["lo"][None, :]) | (q > lim["hi"][None, :]), axis=1)
    bad_velocity = np.any(abs(dq) > lim["vmax"][None, :]+1e-9, axis=1)
    bad_accel = np.any(abs(ddq) > c["descriptive_acceleration_screen_rad_s2"], axis=1)
    win = kinematic_window(tree, masses, q, c, chain.frozen()["geometry_pairs"])
    com_delta = win["com_minus_ankle_m"]-baseline_window["com_minus_ankle_m"]
    ankle_delta = win["right_minus_left_ankle_m"]-baseline_window["right_minus_left_ankle_m"]
    ci = round(c["contact_proxy_time_s"]/c["sample_period_s"])
    return {
        "halfwidth_s":float(width),
        "max_absolute_qdot_limit_ratio":float(np.max(abs(dq)/lim["vmax"][None,:])),
        "position_limit_violating_200hz_frames":int(bad_position.sum()),
        "velocity_limit_violating_200hz_frames":int(bad_velocity.sum()),
        "descriptive_accel_cap_rad_s2":float(c["descriptive_acceleration_screen_rad_s2"]),
        "descriptive_accel_violating_200hz_frames":int(bad_accel.sum()),
        "max_absolute_analytic_qddot_rad_s2":float(np.max(abs(ddq))),
        "max_qdot_step_change_rad_s":float(np.max(abs(np.diff(dq,axis=0)))),
        "max_q_200hz_step_rad":float(np.max(abs(np.diff(q,axis=0)))),
        "max_center_qdot_change_from_baseline_rad_s":float(np.max(abs(dq[ci]-p.dq[ci]))),
        "max_ankle_relative_path_change_from_base_m":float(np.max(np.linalg.norm(ankle_delta,axis=1))),
        "max_com_stance_relative_path_change_from_base_m":float(np.max(np.linalg.norm(com_delta,axis=1))),
        "window_samples":win["samples"],
        "min_urdf_four_pair_capsule_margin_m":win["minimum_urdf_capsule_margin_m"],
        "urdf_four_pair_warning_frames":win["count_urdf_proximity_warning_frames"],
        "four_pair_minimum_m":win["per_pair_minimum_capsule_separation_m"],
        "descriptive_cpu_screen_pass":bool(
            bad_position.sum()==0 and bad_velocity.sum()==0 and bad_accel.sum()==0
            and win["count_urdf_proximity_warning_frames"]==0),
        "real_g1_physx_rollouts":0,
        "genuine_instep_impulse_or_ball_speed_witness":False,
        "native_contact_pair_certified":False,
        "measured_grf_or_stance_patch":False,
        "motor_execution_authorized":False,
        "strict_physx_success":False
    }


def run(urdf: Path = task.URDF) -> dict:
    c = frozen()
    parent = lp.frozen()
    tree, lim = lp.original_kinematics(urdf, parent)
    masses = task.mass_inertials(urdf, tree)
    q0 = (lim["lo"]+lim["hi"])/2
    base_cfg = chain.frozen()
    cases = []
    counts = dict(min_change_solutions=0, min_change_rejections=0,
                  temporal_attempts=0, cpu_screen_pass=0,
                  q_limit_bad_attempts=0, speed_limit_bad_attempts=0,
                  accel_screen_bad_attempts=0, urdf_proximity_warning_attempts=0)
    for mult in c["multipliers"]:
        p = chain.generate(base_cfg,q0,mult)
        idx = round(c["contact_proxy_time_s"]/c["sample_period_s"])
        problem = lp.problem_at_pose(tree, masses, p.q[idx], parent)
        base_window = kinematic_window(tree,masses,p.q,c,base_cfg["geometry_pairs"])
        for target in c["relative_swing_ankle_origin_x_mps"]:
            witness = min_change_witness(problem,p.dq[idx],lim,c,parent,target)
            rows = []
            if witness["success"]:
                counts["min_change_solutions"] += 1
            else:
                counts["min_change_rejections"] += 1
            for width in c["compact_bump_halfwidth_s"]:
                counts["temporal_attempts"] += 1
                if witness["success"]:
                    metrics = evaluate(p,witness["qdot"],tree,masses,lim,c,width,base_window)
                    counts["cpu_screen_pass"] += int(metrics["descriptive_cpu_screen_pass"])
                    counts["q_limit_bad_attempts"] += int(metrics["position_limit_violating_200hz_frames"]>0)
                    counts["speed_limit_bad_attempts"] += int(metrics["velocity_limit_violating_200hz_frames"]>0)
                    counts["accel_screen_bad_attempts"] += int(metrics["descriptive_accel_violating_200hz_frames"]>0)
                    counts["urdf_proximity_warning_attempts"] += int(metrics["urdf_four_pair_warning_frames"]>0)
                    rows.append(metrics)
                else:
                    rows.append({"halfwidth_s":float(width),"not_evaluated_reason":witness["reason"],
                                 "descriptive_cpu_screen_pass":False,
                                 "motor_execution_authorized":False,"strict_physx_success":False})
            safe_witness = {k:v for k,v in witness.items() if k!="qdot"}
            cases.append(dict(multiplier=float(mult),target_right_ankle_link_origin_x_mps=float(target),
                              witness=safe_witness,temporal=rows))
    if len(cases) != 15 or counts["temporal_attempts"] != 45:
        raise RuntimeError("FROZEN_MATRIX_TRUNCATED")
    if any(t.get("motor_execution_authorized") or t.get("strict_physx_success")
           for r in cases for t in r["temporal"]):
        raise RuntimeError("MOTOR_PERMISSION_OR_PHYSICAL_SUCCESS_FORGED")
    return {
        "version":"v0.1.138", "parent_sha":c["parent_sha"],
        "protocol_sha":FROZEN_SHA, "original_urdf_sha256":c["urdf_sha256"],
        "original29_dof":29, "full_14s_joint_frames_per_temporal_proposal":2800,
        "contact_proxy_time_s":2.95, "is_real_instep_contact_time":False,
        "is_human_motion_or_ground_force_learned":False,
        "is_contact_physics_authenticated":False,"real_physx_rollouts_new":0,
        "strict_real_physx_success_rate":"N/A - no new physics rollouts",
        "motor_execution_authorized":False,
        "summary_counts":counts,"cases":cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--urdf",type=Path,default=task.URDF)
    a = parser.parse_args()
    results = run(a.urdf)
    with a.output.open("x",encoding="utf8") as f:
        json.dump(results,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({"counts":results["summary_counts"],"out":str(a.output),
                      "real_physx_rollouts":0},ensure_ascii=False))


if __name__=="__main__":
    main()
