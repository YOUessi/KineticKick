#!/usr/bin/env python3
"""v130: SHA-locked real 200Hz G1 COM, support-force proxy, 29DoF work audit.

Measured trace fields only; no human GRF, CoP, collider paths, contact manifold,
motor command, or unsampled termination-state reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import v128_v54_event_geometry as original
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v130_frozen_com_support_coupling.json"
FROZEN_SHA = "3d315bdb818ef9459304b92c5948f8d0fc7303fe"
CASES = {"frozen_v32": 2800, "cmu_10_02": 2800, "cmu_10_05": 1029}


def frozen() -> dict:
    committed = subprocess.check_output(
        ["git", "-C", str(ROOT), "show",
         FROZEN_SHA + ":configs/v130_frozen_com_support_coupling.json"])
    if CONFIG.read_bytes() != committed:
        raise ValueError("V130_PRE_REGISTERED_PROTOCOL_DRIFT")
    cfg = json.loads(committed)
    if (cfg["version"] != "v0.1.130" or
            cfg["parent_commit"] != "cf1fa1ac824c97492e9738b4b94e12dc1d959482" or
            cfg["native_physx_contact_pair_certified"] or
            cfg["no_training_parameter_tuning"] is not True or
            cfg["no_motor_command_or_model_weight_changes"] is not True or
            cfg["root_orientation_available"] is not False or
            cfg["time_shift_negative_control_lags_frames"] != [0, 40, 80] or
            cfg["dt_s"] != .005):
        raise ValueError("V130_UNSAFE_OR_MUTATED_CONTRACT")
    return cfg


def _finite_matrix(rows: list[dict], field: str, count: int) -> np.ndarray:
    try:
        data = np.asarray([r[field] for r in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("BAD_REAL_PHYSICS_FIELD:" + field) from e
    if data.shape != (len(rows), count) or not np.isfinite(data).all():
        raise ValueError("BAD_FINITE_REAL_PHYSICS_FIELD:" + field)
    return data


def pearson(x, y):
    """No NaN/undefined correlations silently converted to success."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("INVALID_CORRELATION_SERIES")
    if x.size < 4 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_shares(forces: np.ndarray, cfg: dict):
    """Only force *proxy* based side fractions, NOT native contact/GRF."""
    f = np.asarray(forces, dtype=float)
    if f.ndim != 2 or f.shape[1] != 2 or not np.isfinite(f).all() or (f < 0).any():
        raise ValueError("INVALID_FOOT_FORCE_PROXIES")
    total = f.sum(axis=1)
    observed = total >= cfg["force_proxy_N_min_for_load_share"]
    left = np.full(len(f), np.nan)
    left[observed] = f[observed, 0] / total[observed]
    dominant = np.zeros(len(f), dtype=np.int8)
    threshold = cfg["load_share_dominance_threshold"]
    dominant[observed & (left > threshold)] = 1
    dominant[observed & (left < (1-threshold))] = -1
    return total, observed, left, dominant


def windows(t: np.ndarray, contact: float, cfg: dict):
    """Require all eighty/two-hundred-fifty real impact-time samples."""
    t = np.asarray(t, dtype=float)
    dt = cfg["dt_s"]
    if (not t.size or not np.isfinite(t).all() or
            not np.allclose(t, dt*np.arange(1,len(t)+1), rtol=0, atol=2e-5)):
        raise ValueError("UNCERTIFIED_OR_NONCONTIGUOUS_200HZ")
    pre = (t >= contact-cfg["impact_window_pre_s"]-1e-7) & (t < contact-1e-7)
    post = (t >= contact-1e-7) & (t < contact+cfg["impact_window_post_s"]-1e-7)
    late = (t >= contact+cfg["impact_window_post_s"]-1e-7)
    if pre.sum() != 80 or post.sum() != 250 or (pre & post).any():
        raise ValueError("MISSING_PRESCRIBED_IMPACT_WINDOWS")
    return {"pre_400ms": pre, "post_1250ms": post, "observed_late_recovery": late}


def interval_stats(t, qdot, tau, com, com_vel, force, nonfoot, mask, cfg):
    idx = np.flatnonzero(mask)
    if not len(idx):
        raise ValueError("EMPTY_OBSERVED_PHASE")
    dt = cfg["dt_s"]
    power = qdot[idx] * tau[idx]  # original 29 joints, signed watts
    total = power.sum(axis=1)
    ft = force[idx]
    force_total, valid, left_fraction, dominant = load_shares(ft, cfg)
    # The time-shift diagnostic is purely correlational, not inferred GRF.
    accel_z = np.gradient(com_vel[idx,2], dt) if len(idx) >= 3 else np.zeros(len(idx))
    correlations = {}
    for lag in cfg["time_shift_negative_control_lags_frames"]:
        # A shift by the full window length is mathematically identical
        # to zero lag, NOT an independent permutation null. Preserve the
        # preregistered lag and explicitly mark it noninformative.
        if lag != 0 and lag % len(idx) == 0:
            correlations[str(lag)] = None
            continue
        shifted = np.roll(force_total, lag)
        correlations[str(lag)] = pearson(shifted[valid], accel_z[valid])
    # Count direct switches only at *consecutive* witnessed samples.
    # A switch after a low-force/unknown gap is not a directly witnessed
    # support transfer and must remain separately unresolved.
    observed_switches = (dominant[1:] * dominant[:-1] == -1)
    direct_transitions = int(np.count_nonzero(observed_switches))
    compressed = dominant[dominant != 0]
    possible_across_gaps = int(np.count_nonzero(np.diff(compressed) != 0))
    unresolved_gap_transitions = max(0, possible_across_gaps-direct_transitions)
    joint_work = (power * dt).sum(axis=0)
    return {
        "physical_200hz_frames": int(len(idx)),
        "start_s": float(t[idx[0]]),
        "end_s": float(t[idx[-1]]),
        "total_signed_joint_motor_work_proxy_J": float(np.sum(total) * dt),
        "total_positive_joint_motor_work_proxy_J": float(np.maximum(power, 0).sum()*dt),
        "total_negative_joint_motor_work_proxy_J": float(np.minimum(power, 0).sum()*dt),
        "max_absolute_sum_29_joint_power_W": float(np.abs(total).max()),
        "max_absolute_single_joint_power_W": float(np.abs(power).max()),
        "max_joint_power_joint": G1_JOINT_ORDER[int(np.unravel_index(np.argmax(np.abs(power)),power.shape)[1])],
        "top_5_joints_by_abs_signed_work": [
            {"joint": G1_JOINT_ORDER[j], "signed_work_J": float(joint_work[j])}
            for j in np.argsort(-np.abs(joint_work))[:5]],
        "com_world_displacement_xyz_m": (com[idx[-1]]-com[idx[0]]).tolist(),
        "com_world_displacement_norm_m": float(np.linalg.norm(com[idx[-1]]-com[idx[0]])),
        "com_speed_peak_mps": float(np.linalg.norm(com_vel[idx], axis=1).max()),
        "com_vertical_velocity_start_end_mps": [float(com_vel[idx[0],2]),float(com_vel[idx[-1],2])],
        "left_proxy_integral_Ns": float(ft[:,0].sum()*dt),
        "right_proxy_integral_Ns": float(ft[:,1].sum()*dt),
        "force_proxy_observable_frames": int(valid.sum()),
        "force_proxy_unknown_or_low_frames": int((~valid).sum()),
        "left_proxy_dominant_frames": int((dominant==1).sum()),
        "right_proxy_dominant_frames": int((dominant==-1).sum()),
        "direct_consecutive_side_proxy_switches": direct_transitions,
        "side_proxy_switches_across_unknown_gap_not_certified":
            unresolved_gap_transitions,
        "com_vertical_acceleration_vs_total_support_proxy_shifted_correlation":
            correlations,
        "invalid_circular_null_lags_aliasing_zero": [
            lag for lag in cfg["time_shift_negative_control_lags_frames"]
            if lag and lag % len(idx) == 0],
        "nonfoot_net_force_peak_N": float(nonfoot[idx].max()),
        "nonfoot_net_force_nonzero_frames": int(np.count_nonzero(nonfoot[idx] > 0)),
        "native_physx_foot_pair_or_GRF_certified": False,
        "support_polygon_or_CoP_available": False,
    }


def analyze_case(case: str, cfg: dict, original_cfg: dict, root: Path):
    if case not in CASES or case not in cfg["source_original_v54_cases"]:
        raise ValueError("UNFROZEN_CASE")
    basename = "v054_" + case + "_contact_diagnostics.json"
    path = root/basename
    original.require_sha(path, original_cfg["input_sha256"][basename])
    raw = json.loads(path.read_text())
    series = original.validate_rows(raw, CASES[case], cfg["dt_s"])
    rows = raw["traces"][0][:len(series["t"])]
    qdot = series["dq"]
    tau = _finite_matrix(rows, "joint_torque_estimate", 29)
    com = series["com"]
    com_vel = _finite_matrix(rows,"whole_body_com_velocity",3)
    force = series["support"]
    nonfoot = series["nonfoot"]
    masks = windows(series["t"],series["impact"],cfg)
    results = {k: interval_stats(series["t"],qdot,tau,com,com_vel,force,
                                  nonfoot,mask,cfg) for k,mask in masks.items()}
    # These are measured quantities from the existing 200Hz simulation, NOT
    # a reconstructed contact manifold or a qualified human motion label.
    strict_full_recovery = bool(len(series["t"])==2800 and
                                abs(series["t"][-1]-14.0)<2e-5)
    if (case=="cmu_10_05") == strict_full_recovery:
        raise ValueError("UNEXPECTED_TERMINATION_OR_RECOVERY_DURATION")
    return {
        "case":case,"source_sha256": original_cfg["input_sha256"][basename],
        "original_physics_frames":int(len(series["t"])),
        "nonphysics_sentinel_excluded":int(series["terminal_sentinel"]),
        "original_200hz_end_s":float(series["t"][-1]),
        "contact_time_s":float(series["impact"]),
        "complete_14_second_recovery_observed":strict_full_recovery,
        "missing_late_recovery_is_censored":not strict_full_recovery,
        "phases":results,
        "native_colliders_and_human_contact_truth":"UNKNOWN",
        "human_soccer_grf_cop_witnessed":False,
        "physical_motor_command_issued":False
    }


def audit(data_root: Path | None = None):
    cfg = frozen()
    source_cfg = original.frozen_protocol()
    if source_cfg["version"]!="v0.1.128" or source_cfg["native_contact_pair_certified"]:
        raise ValueError("PARENT_PHYSICAL_EVIDENCE_DRIFT")
    root = Path(data_root) if data_root else Path(source_cfg["data_root"])
    cases = [analyze_case(c,cfg,source_cfg,root) for c in cfg["source_original_v54_cases"]]
    if sum(c["complete_14_second_recovery_observed"] for c in cases)!=cfg["expected_nominal_cases"]:
        raise ValueError("FROZEN_COVERAGE_COUNT_DRIFT")
    return {
        "version":"v0.1.130",
        "protocol_commit":FROZEN_SHA,
        "original_v54_data_root":str(root),
        "new_physx_episodes":0,
        "verified_original_physx_episodes":len(cases),
        "physical_200hz_frames":sum(c["original_physics_frames"] for c in cases),
        "complete_14s_episodes":sum(c["complete_14_second_recovery_observed"] for c in cases),
        "censored_failures":sum(c["missing_late_recovery_is_censored"] for c in cases),
        "native_collider_manifold_certification":"UNKNOWN",
        "high_performance_human_instep_impact_training_truth":False,
        "physical_actuation_or_training_performed":False,
        "evidence":cases
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path)
    p.add_argument("--output",type=Path,
                   default=ROOT/"reports/v130_real_com_support_work.audit.json")
    a=p.parse_args()
    result=audit(a.data_root)
    with a.output.open("x") as out:
        json.dump(result,out,indent=2,allow_nan=False);out.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="evidence"},indent=2))
    for e in result["evidence"]:
        print(e["case"],e["original_physics_frames"],e["original_200hz_end_s"],
              "pre_work_J",round(e["phases"]["pre_400ms"]["total_signed_joint_motor_work_proxy_J"],3),
              "post_work_J",round(e["phases"]["post_1250ms"]["total_signed_joint_motor_work_proxy_J"],3),
              "late_nonfoot_N",round(e["phases"]["observed_late_recovery"]["nonfoot_net_force_peak_N"],4))
if __name__=="__main__":
    main()
