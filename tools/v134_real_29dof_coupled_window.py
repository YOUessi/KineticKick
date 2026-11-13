#!/usr/bin/env python3
"""v134 SHA-qualified true original-G1 200 Hz coupled event-window evidence.

Frozen at named sensor time 5.145s for the v54 failed CMU10_05 case.
The other controller windows share a clock only, not an independently
certified same-contact event. No missing after-event states are imputed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

import v128_v54_event_geometry as v128
from wholebody_safety_interface_v055 import G1_JOINT_ORDER
import v134_native_contact_vector_gate as contract

CASES = ("frozen_v32", "cmu_10_02", "cmu_10_05")
CONTACT_ANCHOR = 5.145
PRE_WINDOW = .400
POST_WINDOW = 1.250
DT = .005


def group_indices():
    names = tuple(G1_JOINT_ORDER)
    labels = {
        "left_hip_knee_ankle": ("left_hip_", "left_knee_", "left_ankle_"),
        "right_hip_knee_ankle": ("right_hip_", "right_knee_", "right_ankle_"),
        "waist_trunk": ("waist_",),
        "left_shoulder_elbow_wrist": ("left_shoulder_", "left_elbow_", "left_wrist_"),
        "right_shoulder_elbow_wrist": ("right_shoulder_", "right_elbow_", "right_wrist_"),
    }
    groups = {
        k: np.array([i for i, n in enumerate(names) if n.startswith(prefix)], dtype=int)
        for k, prefix in labels.items()
    }
    if sum(len(i) for i in groups.values()) != 29:
        raise ValueError("NOT_EXACT_ORIGINAL_29DOF_JOINT_GROUPS")
    unique = np.concatenate(list(groups.values()))
    if set(unique) != set(range(29)):
        raise ValueError("JOINT_GROUP_OVERLAP_OR_GAP")
    return groups


def frame_window_audit(case, x):
    if case not in CASES:
        raise ValueError("NOT_FROZEN_ORIGINAL_V54_CASE")
    t, q, dq, com, support, nonfoot, vlim = (
        np.asarray(x[k], dtype=float) for k in
        ("t", "q", "dq", "com", "support", "nonfoot", "limits"))
    if (q.shape != (len(t), 29) or dq.shape != q.shape
            or com.shape != (len(t), 3) or support.shape != (len(t), 2)
            or nonfoot.shape != (len(t),) or vlim.shape != (29,)
            or not all(np.isfinite(a).all() for a in (
                t, q, dq, com, support, nonfoot, vlim))
            or not np.all(np.diff(t) > 0)
            or np.any(vlim <= 0)):
        raise ValueError("BAD_ORIGINAL_G1_COUPLED_PHYSICS_SHAPE")
    if np.any(np.abs(np.diff(t) - DT) > 1e-7):
        raise ValueError("TIME_GAP_OR_WRONG_200HZ_IN_RAW_STATES")
    tol = 1e-7
    before = np.flatnonzero(
        (t >= CONTACT_ANCHOR - PRE_WINDOW - tol) &
        (t < CONTACT_ANCHOR - tol))
    after = np.flatnonzero(
        (t >= CONTACT_ANCHOR - tol) &
        (t <= CONTACT_ANCHOR + POST_WINDOW + tol))
    if len(before) != 80:
        raise ValueError("MISSING_FROZEN_400MS_PRE_EVENT_WINDOW")
    if case == "cmu_10_05" and len(after):
        raise ValueError("FAILED_EPISODE_HAS_FABRICATED_POST_STATE")
    if case != "cmu_10_05" and len(after) != 251:
        raise ValueError("SUCCESSFUL_CONTROL_HAS_INCOMPLETE_1250MS_WINDOW")
    if case == "cmu_10_05" and not np.isclose(t[-1], 5.140):
        raise ValueError("FAILED_EPISODE_LAST_PHYSICAL_STEP_DRIFT")
    ratio = np.abs(dq) / vlim[None, :]
    groups = group_indices()
    result = {
        "case": case,
        "anchor_s": CONTACT_ANCHOR,
        "anchor_kind": ("NAMED_SENSOR_NO_DIRECT_STATE" if case == "cmu_10_05"
                        else "CLOCK_ONLY_NO_MATCHED_CONTACT_CERTIFICATION"),
        "recorded_real_200hz_physics_frames": len(t),
        "real_pre_0p4s_frames": len(before),
        "required_post_1p25s_frames": 251,
        "real_post_1p25s_frames": len(after),
        "missing_post_frames": 251 - len(after),
        "event_5p145_has_real_29_joint_state": bool(len(after)),
        "full_original29_max_speed_ratio": float(ratio.max()),
        "pre_0p4s_max_original29_speed_ratio": float(ratio[before].max()),
        "pre_0p4s_joint_speed_exceedance_frames": int(np.count_nonzero(
            np.max(ratio[before], axis=1) > 1.0)),
        "pre_0p4s_left_right_foot_support_force_proxy_integrals_Ns": (
            DT * np.sum(support[before], axis=0)).tolist(),
        "pre_0p4s_COM_horizontal_path_length_m": float(np.linalg.norm(
            np.diff(com[before, :2], axis=0), axis=1).sum()),
        "pre_0p4s_max_nonfoot_net_force_N": float(nonfoot[before].max()),
        "pre_0p4s_nonfoot_over_20N_frames": int(np.count_nonzero(
            nonfoot[before] > 20.0)),
        "pre_0p4s_named_sensor_force_peak_s": (
            float(t[before[np.argmax(nonfoot[before])]])
            if np.max(nonfoot[before]) > 0 else None),
        "joints_group_peak_excursion_pre_0p4s_rad": {
            label: float(np.max(np.abs(q[before][:, indices] -
                                       q[before[0], indices])))
            for label, indices in groups.items()},
        "joints_group_peak_qdot_ratio_pre_0p4s": {
            label: float(np.max(ratio[before][:, indices]))
            for label, indices in groups.items()},
        "post_1p25s_joint_speed_exceedance_frames": (
            int(np.count_nonzero(np.max(ratio[after], axis=1) > 1.0))
            if len(after) else None),
        "post_1p25s_left_right_support_proxy_integrals_Ns": (
            (DT * np.sum(support[after], axis=0)).tolist()
            if len(after) else None),
        "post_1p25s_nonfoot_peak_N": (
            float(nonfoot[after].max()) if len(after) else None),
        "raw_trace_is_real_200hz_physx_states": True,
        "contact_pair_manifold_truth_not_obtained": True,
        "missing_post_failure_trajectory_imputed": False,
        "support_proxy_not_true_GRF_or_COP": True,
        "not_human_kinematic_or_force_truth": True,
        "not_new_physics_rollout": True,
        "physical_motor_command_issued": False,
    }
    if result["real_pre_0p4s_frames"] != 80:
        raise ValueError("PRE_EVENT_ALIGNMENT_UNEXPECTED")
    return result


def exact_legacy_off_on_common_field_parity(off, on, *, field_names):
    """Strict exact Python object equality over frozen common physics schemas."""
    if set(field_names) != set(v128.ORIGINAL_SHARED_PHYSICAL_FIELDS):
        raise ValueError("LEGACY_COMMON_PHYSICS_FIELD_SET_CHANGED")
    if any(field not in off or field not in on for field in field_names):
        raise ValueError("ORIGINAL_SOURCE_COMMON_FIELD_MISSING")
    differences = [field for field in field_names if off[field] != on[field]]
    return dict(common_fields_compared=len(field_names),
                all_common_physical_fields_exact=not differences,
                nonidentical_fields=differences,
                this_is_replayed_historical_physx_not_new_rollout=True)


def reverify_original_v52_v54_off_on():
    """Recheck SHA-proven v52 vs original read-only v54 traces directly."""
    frozen = v128.frozen_protocol()
    root = Path(frozen["data_root"])
    historical_witness = json.loads(
        (root / "v054_original_named_nonfoot_audit.json").read_text())
    if historical_witness.get("all_physics_same_except_added_diagnostics") is not True:
        raise ValueError("ORIGINAL_V54_OFF_ON_WITNESS_NOT_POSITIVE")
    result = {}
    for case in CASES:
        source = root / v128.ORIGINAL_V52_SOURCE_BY_CASE[case]
        target_name = f"v054_{case}_contact_diagnostics.json"
        target = root / target_name
        witness = historical_witness["results"][case]
        v128.require_sha(source, witness["source_sha256"])
        v128.require_sha(target, frozen["input_sha256"][target_name])
        if witness["instrumented_sha256"] != frozen["input_sha256"][target_name]:
            raise ValueError("V54_REPLAY_SOURCE_SHA_DISAGREES")
        off = json.loads(source.read_text())
        on = json.loads(target.read_text())
        parity = exact_legacy_off_on_common_field_parity(
            off, on, field_names=v128.ORIGINAL_SHARED_PHYSICAL_FIELDS)
        if (not parity["all_common_physical_fields_exact"]
                or not witness["parity_full_on_all_common_physics_records"]):
            raise ValueError("ORIGINAL_V52_V54_TRUE_PHYSICS_DIFFERED")
        parity["off_original_v52_sha256"] = witness["source_sha256"]
        parity["on_original_v54_sha256"] = witness["instrumented_sha256"]
        parity["this_is_not_current_v134_runtime_observer_invariance"] = True
        result[case] = parity
    return result


def analyze_three_original_v54():
    frozen_cfg = contract.frozen_protocol()
    old = v128.frozen_protocol()
    root = Path(old["data_root"])
    out = {}
    total = 0
    for case in CASES:
        name = f"v054_{case}_contact_diagnostics.json"
        path = root / name
        v128.require_sha(path, old["input_sha256"][name])
        raw = json.loads(path.read_text())
        expected = frozen_cfg["historical_original_g1_steps"][case]
        x = v128.validate_rows(raw,
                              expected + (1 if case == "cmu_10_05" else 0), DT)
        if len(x["t"]) != expected or int(x["terminal_sentinel"]) != int(
                case == "cmu_10_05"):
            raise ValueError("ORIGINAL_V54_SHA_AND_SAMPLING_DRIFT")
        if tuple(raw["joint_names"]) != tuple(G1_JOINT_ORDER):
            raise ValueError("ORIGINAL_G1_29JOINT_ORDER_CHANGED")
        audited = frame_window_audit(case, x)
        audited["sha256_of_exact_source_json"] = old["input_sha256"][name]
        audited["physx_runtime_contact_pair_certified"] = False
        out[case] = audited
        total += expected
    if total != 6628:
        raise ValueError("HISTORICAL_AUTHENTIC_PHYSX_TOTAL_DRIFT")
    failure = out["cmu_10_05"]
    if (failure["pre_0p4s_named_sensor_force_peak_s"] != 5.135
            or failure["real_post_1p25s_frames"] != 0
            or failure["pre_0p4s_max_nonfoot_net_force_N"] < 25.89):
        raise ValueError("NAMED_NONFOOT_PHYSICS_NEGATIVE_CHANGED")
    parity = reverify_original_v52_v54_off_on()
    return {
        "version": "v0.1.134",
        "original_v52_v54_historical_read_only_observer_off_on_exact_parity": parity,
        "source_sha_frozen": True,
        "g1_true_physx_episodes_reexamined": 3,
        "g1_true_physx_states_reexamined": total,
        "new_real_physx_episodes": 0,
        "new_policy_trials": 0,
        "real_native_callback_headers": 0,
        "old_successful_controller_clock_windows_not_actual_contacts": True,
        "native_actor_collider_pair_verified": False,
        "not_usable_as_human_instep_force_truth": True,
        "all_failed_recovery_and_nonfoot_conditions_preserved": True,
        "cases": out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = analyze_three_original_v54()
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "source_episodes": results["g1_true_physx_episodes_reexamined"],
        "source_real_physics_frames": results["g1_true_physx_states_reexamined"],
        "new_physx_episodes": 0,
        "pre_window_samples_each_case": [v["real_pre_0p4s_frames"]
                                        for v in results["cases"].values()],
        "post_window_samples_each_case": [v["real_post_1p25s_frames"]
                                         for v in results["cases"].values()],
        "failed_nonfoot_peak_N": results["cases"]["cmu_10_05"][
            "pre_0p4s_max_nonfoot_net_force_N"],
    }, indent=2))


if __name__ == "__main__":
    main()
