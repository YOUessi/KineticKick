#!/usr/bin/env python3
"""v147: real SHA-locked original G1 200Hz wholebody mechanical *proxies*.

Input is the actual v54 recorded PhysX state, NOT a fresh PhysX episode.
Joint torque is simulator estimated; foot forces include non-ground contacts.
Do not infer human motor-learning, causal transfer or actual GRF from outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from v146_real_200hz_coupled_audit import (
    verified_sources, split_real_5ms_rows, verify_g1_metadata,
)
from v147_exact_off_on_parity import frozen, FROZEN_SHA
from v142_native_pair_ledger import safe_write_once

DT = .005


def grouping(joint_names):
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("V147_WRONG_G1_JOINT_NAMES")
    names = list(joint_names)
    labels = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")
    groups = {key: [] for key in labels}
    for i, name in enumerate(names):
        if name.startswith("waist_"):
            group = "waist"
        elif name.startswith("left_") and any(
                part in name for part in ("hip_", "knee_", "ankle_")):
            group = "left_leg"
        elif name.startswith("right_") and any(
                part in name for part in ("hip_", "knee_", "ankle_")):
            group = "right_leg"
        elif name.startswith("left_") and any(
                part in name for part in ("shoulder_", "elbow_", "wrist_")):
            group = "left_arm"
        elif name.startswith("right_") and any(
                part in name for part in ("shoulder_", "elbow_", "wrist_")):
            group = "right_arm"
        else:
            raise ValueError("V147_UNCLASSIFIED_ORIGINAL_G1_JOINT:" + name)
        groups[group].append(i)
    if [len(groups[k]) for k in labels] != [6,6,3,7,7]:
        raise ValueError("V147_BODY_GROUP_MAP_DRIFT")
    return groups


def _float_matrix(rows, key, columns):
    a = np.asarray([row[key] for row in rows], dtype=np.float64)
    if a.shape != (len(rows), columns) or not np.isfinite(a).all():
        raise ValueError("V147_BAD_NUMERICAL_MATRIX_" + key)
    return a


def summarize_segment(indices, dq, torque, comv, net_l, net_r, nonfoot,
                      velocity_limits, group_map):
    if len(indices) == 0:
        raise ValueError("V147_EMPTY_FROZEN_WINDOW")
    # Valid force ratio is ONLY a ratio of net sensor magnitudes, NOT a CoP/GRF estimate.
    bilateral_sum = net_l[indices] + net_r[indices]
    finite_share = bilateral_sum > 1e-10
    left_share = net_l[indices][finite_share] / bilateral_sum[finite_share]
    central = indices[(indices > 0) & (indices < len(comv)-1)]
    central_acc = (comv[central+1] - comv[central-1])/(2*DT)
    abs_power = np.abs(torque[indices] * dq[indices])
    local = {
        "sample_count": int(len(indices)),
        "first_time_s": round(float((indices[0]+1)*DT), 9),
        "last_time_s": round(float((indices[-1]+1)*DT), 9),
        "max_original29_qdot_over_limit": float(np.max(
            np.abs(dq[indices]) / velocity_limits[None,:])),
        "sim_torque_abs_joint_power_proxy_integral_J": float(
            np.sum(abs_power, dtype=np.float64) * DT),
        "maximum_absolute_com_velocity_mps": float(np.max(np.linalg.norm(
            comv[indices], axis=1))),
        "max_central_finite_diff_com_accel_proxy_mps2": float(np.max(
            np.linalg.norm(central_acc, axis=1))) if len(central) else None,
        "max_nonfoot_net_contact_N": float(np.max(nonfoot[indices])),
        "net_force_left_share_median_not_grf": (
            float(np.median(left_share)) if len(left_share) else None),
        "valid_net_force_share_samples": int(finite_share.sum()),
        "group_absolute_sim_torque_speed_proxy_integrals_J": {},
        "group_max_original_qdot_ratio": {},
    }
    for key, joints in group_map.items():
        local["group_absolute_sim_torque_speed_proxy_integrals_J"][key] = (
            float(np.sum(abs_power[:,joints], dtype=np.float64)*DT))
        local["group_max_original_qdot_ratio"][key] = float(np.max(
            np.abs(dq[np.ix_(indices,joints)])/velocity_limits[joints][None,:]))
    return local


def audit_case(source, name, expected_rows):
    frozen()
    raw = Path(source).read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    limits, torque_limits = verify_g1_metadata(data)
    rows, notes = split_real_5ms_rows(data["traces"][0], expected_rows)
    group_map = grouping(data["joint_names"])
    q = _float_matrix(rows, "joint_pos", 29)
    dq = _float_matrix(rows, "joint_vel", 29)
    torque = _float_matrix(rows, "joint_torque_estimate", 29)
    comv = _float_matrix(rows, "whole_body_com_velocity", 3)
    com = _float_matrix(rows, "whole_body_com_xyz", 3)
    net_l = np.asarray([r["support_foot_force_N"]["left"] for r in rows], dtype=float)
    net_r = np.asarray([r["support_foot_force_N"]["right"] for r in rows], dtype=float)
    nonfoot = np.asarray([r["nonfoot_contact_N"] for r in rows], dtype=float)
    if not all(np.isfinite(x).all() for x in (net_l,net_r,nonfoot)) or min(
            float(net_l.min()),float(net_r.min()),float(nonfoot.min())) < 0:
        raise ValueError("V147_INVALID_NET_CONTACT_MAGNITUDE")
    t = np.arange(1,len(rows)+1,dtype=float)*DT
    cfg = frozen()
    cuts = {
        "pre": tuple(cfg["precontact_window"]),
        "post": tuple(cfg["postcontact_window"]),
        "late_failed_time_support_net": tuple(cfg["failed_event_window"]),
    }
    result = {}
    for label, (start,end) in cuts.items():
        mask = np.flatnonzero((t >= start-1e-9) & (t <= end+1e-9))
        result[label] = summarize_segment(mask,dq,torque,comv,net_l,net_r,
                                          nonfoot,np.asarray(limits),group_map)
    result["full_real_trace"] = summarize_segment(
        np.arange(len(rows)),dq,torque,comv,net_l,net_r,nonfoot,
        np.asarray(limits),group_map)
    event_index = int(np.argmax(nonfoot))
    high = np.argwhere(np.abs(dq)/np.asarray(limits)[None,:] > 1.0)
    return {
        "policy": name,
        "source_file_sha256": source_sha,
        "actual_original29_200hz_rows": len(rows),
        "terminal_nonphysical_annotations": len(notes),
        "last_real_5ms_state_s": round(float(t[-1]),9),
        "complete_14s_coverage": len(rows)==2800,
        "recorded_legacy_foot_ball_event_s": float(data["records"][0]["contact_time_s"]),
        "max_nonfoot_net_N": float(nonfoot[event_index]),
        "max_nonfoot_net_time_s": round(float(t[event_index]),9),
        "original29_joint_qdot_limit_violations": int(len(high)),
        "source_29q_is_complete": q.shape==(len(rows),29),
        "estimated_torque_over_effort_limit_max": float(np.max(
            np.abs(torque)/np.asarray(torque_limits)[None,:])),
        "group_joint_count": {k:len(v) for k,v in group_map.items()},
        "all_window_metrics": result,
        "native_wrist_hip_collider_manifold": "UNKNOWN",
        "ground_only_grf": "NOT_MEASURED",
        "actuator_output_power_certified": False,
        "human_professional_instep_style_learned": False,
        "new_physx_episodes": 0,
        "new_strict_g1_shot_success": None,
    }


def audit_all():
    cfg = frozen()
    inputs = verified_sources()
    if list(inputs) != list(cfg["original_v54_expected_rows"]):
        raise ValueError("V147_CASES_CHANGED")
    cases = {name:audit_case(path,name,cfg["original_v54_expected_rows"][name])
             for name,path in inputs.items()}
    return {
        "frozen_protocol_sha": FROZEN_SHA,
        "version": cfg["version"],
        "real_historical_200hz_rows_checked": sum(
            c["actual_original29_200hz_rows"] for c in cases.values()),
        "historical_real_physx_episodes_reanalysed": len(cases),
        "new_real_physx_episodes": 0,
        "human_contact_and_grf_truth_available": False,
        "all_native_pair_truth": "UNKNOWN",
        "cases": cases,
        "full_all_gates_outcome_new_policy": None,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    result=audit_all()
    digest=safe_write_once(args.output,result)
    print(json.dumps({"path":str(args.output),"sha256":digest,
                      "history_rows":result["real_historical_200hz_rows_checked"],
                      "case_summaries":{k:{"pre":v["all_window_metrics"]["pre"],
                                         "post":v["all_window_metrics"]["post"],
                                         "nonfoot_peak_N":v["max_nonfoot_net_N"]}
                      for k,v in result["cases"].items()}},ensure_ascii=False))


if __name__=="__main__":
    main()
