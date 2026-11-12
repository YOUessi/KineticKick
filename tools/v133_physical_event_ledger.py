#!/usr/bin/env python3
"""SHA-qualified 200Hz original-G1 per-step event alignment audit (no physics run).

Loads original v54 OFF/ON matched traces after v128's frozen qualifications.
Keeps 5.145s sensor report separate from 5.140s last real joint state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import v128_v54_event_geometry as v128
import v133_native_contact_frame as native

CASES = ("frozen_v32", "cmu_10_02", "cmu_10_05")


def real_event_ledger():
    cfg = native.frozen()
    frozen = v128.frozen_protocol()
    root = Path(frozen["data_root"])
    results = {}
    for case in CASES:
        name = "v054_" + case + "_contact_diagnostics.json"
        path = root / name
        v128.require_sha(path, frozen["input_sha256"][name])
        raw = json.loads(path.read_text())
        nominal = cfg["original_v54_cases"][case]
        source_rows = nominal + (1 if case == "cmu_10_05" else 0)
        x = v128.validate_rows(raw, source_rows, cfg["physical_dt_s"])
        if x["terminal_sentinel"] != int(case == "cmu_10_05"):
            raise ValueError("SOURCE_SENTINEL_COUNT_DRIFT")
        t, dq, com, support = x["t"], x["dq"], x["com"], x["support"]
        if len(t) != nominal or len(x["q"]) != nominal:
            raise ValueError("ORIGINAL_PHYSICAL_STEP_COUNT_CHANGED")
        peak_ratio = np.max(np.abs(dq) / x["limits"][None, :], axis=1)
        window = (t >= cfg["original_named_event_s"] - .065 - 1e-7) & (
            t <= cfg["original_last_real_state_s"] + 1e-7)
        if int(window.sum()) != 13:
            raise ValueError("FROZEN_PRE_EVENT_WINDOW_NOT_13_FRAMES")
        last = int(np.flatnonzero(window)[-1])
        nonfoot = x["nonfoot"]
        peak = int(np.argmax(nonfoot[window]))
        window_idx = np.flatnonzero(window)
        max_i = int(window_idx[peak])
        owner = ["left_wrist_yaw_link", "left_hip_roll_link"]
        last_observation = {
            "physical_step": last + 1,
            "time_s": float(t[last]),
            "q_dim": len(x["q"][last]),
            "dq_dim": len(dq[last]),
            "max_qdot_urdf_ratio": float(peak_ratio[last]),
            "COM_z_m": float(com[last, 2]),
            "feet_net_force_proxy_N": [float(v) for v in support[last]],
        }
        expected = v128.event_observability(t, cfg["original_named_event_s"], .005)
        if case == "cmu_10_05":
            if expected["event_has_directly_sampled_joint_state"]:
                raise ValueError("FAIL_CASE_UNEXPECTEDLY_OBSERVED_AT_5P145")
            if last_observation["time_s"] != cfg["original_last_real_state_s"]:
                raise ValueError("FAIL_CASE_LAST_REAL_TIME_CHANGED")
        outcome = raw["records"][0]
        results[case] = {
            "source_raw_sha256": frozen["input_sha256"][name],
            "source_29dof_original": tuple(raw["joint_names"]) == tuple(
                __import__("wholebody_safety_interface_v055").G1_JOINT_ORDER),
            "real_200hz_steps": len(t),
            "non_physics_terminal_rows_removed": x["terminal_sentinel"],
            "observed_last_time_s": float(t[-1]),
            "pre_event_13_step_max_nonfoot_N": float(nonfoot[window].max()),
            "pre_event_nonfoot_peak_step": max_i + 1,
            "pre_event_nonfoot_peak_t_s": float(t[max_i]),
            "pre_event_13_step_max_qdot_ratio": float(peak_ratio[window].max()),
            "full_episode_max_original29_qdot_ratio": float(peak_ratio.max()),
            "pre_event_left_right_support_force_integral_proxy_Ns": [
                float(np.sum(support[window, j]) * cfg["physical_dt_s"])
                for j in range(2)],
            "pre_event_COM_horizontal_displacement_m": float(
                np.linalg.norm(com[window_idx[-1], :2] - com[window_idx[0], :2])),
            "last_real_state_before_named_contact": last_observation,
            "event_5p145_has_real_29dof_step": expected["event_has_directly_sampled_joint_state"],
            "event_5p145_no_frame_assumption": case != "cmu_10_05" or not expected[
                "event_has_directly_sampled_joint_state"],
            "old_record_reported_success": bool(outcome["success"]),
            "old_record_reported_recovered": bool(outcome["recovered"]),
            "not_a_new_PhysX_rollout": True,
            "not_verified_physx_actor_pair": True,
            "source_expected_candidate_owner_names_not_contact_truth": owner,
        }
    fail = results["cmu_10_05"]
    if fail["pre_event_nonfoot_peak_t_s"] != 5.135:
        raise ValueError("KNOWN_PHYSICAL_FORCE_PEAK_TIME_CHANGED")
    return {
        "version": "v0.1.133",
        "frozen_protocol_sha": native.FROZEN_SHA,
        "original_physx_episodes_reaudited": 3,
        "real_physics_states_reaudited": sum(r["real_200hz_steps"] for r in results.values()),
        "new_physx_episodes": 0,
        "native_headers_captured_in_this_batch": 0,
        "direct_original_wrist_hip_manifold_certified": False,
        "sensor_named_event_s": cfg["original_named_event_s"],
        "original_failed_last_physics_s": cfg["original_last_real_state_s"],
        "joint_state_at_named_event_has_not_been_fabricated": True,
        "strict_physical_success_rate_recomputed_for_new_policy": False,
        "no_motor_actuation": True,
        "cases": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=native.ROOT / "reports/v133_original_G1_event_ledger.audit.json")
    args = ap.parse_args()
    r = real_event_ledger()
    with args.output.open("x") as f:
        json.dump(r, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "source_episodes": r["original_physx_episodes_reaudited"],
        "original_states": r["real_physics_states_reaudited"],
        "new_physx_episodes": r["new_physx_episodes"],
        "native_pair_certified": r["direct_original_wrist_hip_manifold_certified"],
        "original_failed_peak_s": r["cases"]["cmu_10_05"]["pre_event_nonfoot_peak_t_s"]
    }, indent=2))


if __name__ == "__main__":
    main()
