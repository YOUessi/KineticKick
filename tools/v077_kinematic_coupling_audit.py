#!/usr/bin/env python3
"""Source-SHA-locked exploratory event-aligned 200 Hz whole-body coordination audit.

This is measured G1 motion, not professional human ground truth, learned
policy validation, force plate data, direct collision manifold or a new trial.
No optimization or parameter selection on these nine already-seen recordings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

V67 = Path("/home/you/projects/kinetickick-wt-v067/reports")
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "f299cae1a9d044d99e2c936f3c75c02d10605c264e87a134719ba1d25ef2a928"
POLICIES = ("v32", "synth", "reverse")
STANTES = ("a", "b", "c")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for b in iter(lambda: fp.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def corr_lags(left: np.ndarray, right: np.ndarray, lags: tuple[int, ...] = (-20,-10,0,10,20)) -> dict:
    """Lag k compares left[t] with right[t+k], at measured 200 Hz."""
    if len(left) != len(right) or len(left) < 42:
        return {str(k): None for k in lags}
    results = {}
    for lag in lags:
        if lag > 0: a, b = left[:-lag], right[lag:]
        elif lag < 0: a, b = left[-lag:], right[:lag]
        else: a,b = left,right
        if len(a) < 20 or float(np.std(a)) < 1e-7 or float(np.std(b)) < 1e-7:
            results[str(lag)] = None
        else:
            results[str(lag)] = round(float(np.corrcoef(a,b)[0,1]), 6)
    return results


def audit_trace(p: dict, truth: dict) -> dict:
    if len(p.get("joint_names", [])) != 29:
        raise ValueError("29DOF_REQUIRED")
    rows = p.get("traces", [])
    if not rows or not isinstance(rows[0], list) or len(rows[0]) < 2:
        raise ValueError("NO_REAL_PHYSPROFILE")
    rows = rows[0]
    times = np.asarray([r["time_s"] for r in rows], dtype=float)
    dt = np.diff(times)
    # Original early-censored runs append one termination evidence sample 1us
    # after the last 5ms physical frame. Never treat it as a 200Hz step.
    terminal_extra = bool(len(dt) and 0 < dt[-1] <= 0.0001)
    if terminal_extra:
        rows = rows[:-1]
        times = times[:-1]
        dt = np.diff(times)
    if not np.all(np.isfinite(times)) or not np.all(dt > 0) or not np.all(np.isclose(dt, 0.005, atol=2e-4)):
        raise ValueError("NOT_200HZ_UNIQUE_ORDERED_TRACE")
    qdot = np.asarray([r["joint_vel"] for r in rows], dtype=float)
    qpos = np.asarray([r["joint_pos"] for r in rows], dtype=float)
    lim = np.asarray(p["joint_velocity_limits"], dtype=float).reshape(-1)
    if qdot.shape != (len(rows),29) or qpos.shape != qdot.shape or lim.shape != (29,):
        raise ValueError("INVALID_29DOF_SHAPE")
    if not np.all(np.isfinite(qdot)) or not np.all(lim > 0):
        raise ValueError("BAD_ORIGINAL_QDOT_LIMIT")
    ratio = float(np.max(np.abs(qdot) / lim))
    names = p["joint_names"]
    sensors = {
        "waist_yaw": names.index("waist_yaw_joint"),
        "waist_roll": names.index("waist_roll_joint"),
        "right_hip_pitch": names.index("right_hip_pitch_joint"),
        "right_knee": names.index("right_knee_joint"),
        "left_shoulder_yaw": names.index("left_shoulder_yaw_joint"),
        "left_shoulder_pitch": names.index("left_shoulder_pitch_joint"),
        "right_shoulder_pitch": names.index("right_shoulder_pitch_joint"),
    }
    rec = p["records"][0]
    contact = rec.get("contact_time_s")
    if contact is not None and (not math.isfinite(float(contact)) or contact < 0):
        raise ValueError("INVALID_CONTACT_TIMESTAMP")
    end = float(times[-1])
    instep = rec.get("requested_surface_impulse_fraction")
    nonfoot = max(float(row.get("nonfoot_contact_N",0.0)) for row in rows)
    missing = contact is None or not math.isfinite(float(contact))
    sufficient = (not missing and times[0] <= contact-0.4+0.0051
                  and end >= contact+1.25-0.0051)
    windows = {}
    if sufficient:
        for label, lo, hi in (
            ("plant_preimpact_minus_0p4_to_0",contact-0.4,contact),
            ("followthrough_0_to_plus_0p25",contact,contact+0.25),
            ("recovery_plus_0p25_to_plus_1p25",contact+0.25,contact+1.25),
        ):
            sub = (times >= lo-1e-6) & (times < hi-1e-6)
            if int(sub.sum()) < 35:
                windows[label] = {"status":"CENSORED_IN_WINDOW"}
                continue
            q = qdot[sub]
            left = q[:, sensors["left_shoulder_yaw"]]
            right = q[:, sensors["right_hip_pitch"]]
            waist = q[:, sensors["waist_yaw"]]
            knee = q[:, sensors["right_knee"]]
            supports = [r["support_foot_force_N"] for r, ok in zip(rows,sub) if ok]
            left_f = np.asarray([float(s["left"]) for s in supports])
            right_f = np.asarray([float(s["right"]) for s in supports])
            both = (left_f > 10) & (right_f > 10)
            none = (left_f <= 10) & (right_f <= 10)
            windows[label] = {
                "status":"MEASURED_ONLY", "rows":int(sub.sum()),
                "torso_opposite_arm_lagged_signed_corr_200hz":corr_lags(waist,left),
                "swing_hip_knee_lagged_signed_corr_200hz":corr_lags(right,knee),
                "waist_yaw_rms_rads":round(float(np.sqrt(np.mean(waist**2))),6),
                "swing_hip_pitch_rms_rads":round(float(np.sqrt(np.mean(right**2))),6),
                "swing_knee_rms_rads":round(float(np.sqrt(np.mean(knee**2))),6),
                "left_shoulder_yaw_rms_rads":round(float(np.sqrt(np.mean(left**2))),6),
                "both_support_frames":int(both.sum()),
                "zero_support_sensor_frames":int(none.sum()),
                "left_support_mean_N":round(float(left_f.mean()),4),
                "right_support_mean_N":round(float(right_f.mean()),4),
                "minimum_com_z_m":round(min(float(r["whole_body_com_xyz"][2]) for r,ok in zip(rows,sub) if ok),6),
            }
    return {
        "raw_rows":len(rows)+(1 if terminal_extra else 0),
        "unique_200hz_integration_rows":len(rows),
        "terminal_nonphysical_evidence_sample_1us":terminal_extra,
        "actual_end_s":round(end,6),
        "original_29_qdot_limit_peak_ratio":round(ratio,8),
        "original_29_qdot_not_exceeding_limit":ratio <= 1+1e-6,
        "actual_contact_s":contact,
        "contact_plus_1p25_observed":sufficient,
        "missing_or_censored_contact_window":not sufficient,
        "nonfoot_net_force_peak_N":round(nonfoot,4),
        "original_goal":bool(rec.get("goal")),
        "original_surface_instep_fraction":instep,
        "instep_95pct_gate":bool(instep is not None and instep >= .95),
        "original_record_recovered":bool(rec.get("recovered")),
        "authoritative_independent_all_hard_gate_pass":bool(truth["hard_gate_pass"]),
        "authoritative_failed_gates":truth.get("failed_gates",[]),
        "windows":windows,
        "support_force_is_not_ground_reaction_impulse_or_CoP":True,
        "correlation_is_not_causation_or_professional_naturalness":True,
    }


def run() -> dict:
    if sha(ROOT/"configs/v077_frozen_boot_diagnostics.json") != PROTOCOL_SHA:
        raise RuntimeError("V077_PROTOCOL_DRIFT")
    historical = V67/"v067_independent_hard_gate_audit_postregistered.json"
    hard = json.loads(historical.read_text())
    if hard.get("physical_original_GPU_G1_attempts") != 9:
        raise ValueError("HISTORICAL_DENOMINATOR_DRIFT")
    source = {e["id"]:e for e in hard["episodes"]}
    results = []
    for stance in STANTES:
        for control in POLICIES:
            case=f"new_stance_{stance}_{control}"
            truth=source[case]
            raw=V67/f"v067_{case}.json"
            actual_sha=sha(raw)
            if actual_sha != truth["raw_sha256"]:
                raise ValueError("PHYSX_SOURCE_SHA_DRIFT:"+case)
            p=json.loads(raw.read_text())
            row=audit_trace(p,truth["independent_postregistered_hard_gates"])
            row.update(case=case,policy=control,stance=stance,original_sha256=actual_sha)
            results.append(row)
    passed=sum(bool(r["authoritative_independent_all_hard_gate_pass"]) for r in results)
    full=sum(r["actual_end_s"]>=14-1e-5 for r in results)
    observed=sum(r["contact_plus_1p25_observed"] for r in results)
    return {
        "version":"v0.1.77",
        "frozen_protocol_sha256":PROTOCOL_SHA,
        "source_hard_gate_audit_sha256":sha(historical),
        "original_gpu_g1_attempts":9,"this_batch_new_gpu_g1_attempts":0,
        "historical_strict_pass":passed,
        "historical_strict_pass_rate":passed/9,
        "historical_complete_14s":full,
        "contact_window_0p4_pre_1p25_post_valid":observed,
        "protocol_policy":"POSTREGISTERED_ALREADY_SEEN_HYPOTHESIS_ANALYSIS",
        "windows_use_actual_200hz_joint_velocity_and_foot_force":True,
        "no_contact_pair_manifold_certified":True,
        "no_professional_human_impulse_or_GRF_truth":True,
        "no_optimization_or_holdout_claims":True,
        "cases":results,
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    a=parser.parse_args()
    result=run()
    out=Path(a.output)
    with out.open("x") as fp:
        json.dump(result,fp,indent=2,ensure_ascii=False)
        fp.write("\n")
    print("V077_200HZ_AUDIT",json.dumps({k:v for k,v in result.items() if k!="cases"}),flush=True)


if __name__=="__main__":
    main()
