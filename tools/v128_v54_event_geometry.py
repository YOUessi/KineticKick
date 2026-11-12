#!/usr/bin/env python3
"""v128: SHA-frozen, read-only reconstruction of the *original v54* G1 wrist/hip incident.

Outputs cannot certify a native PhysX contact pair or authorize motor commands.
No interpolation or invented q is allowed at unobserved 5.145s termination step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import v127_collider_proximity as geometry
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/v128_frozen_v54_collision_event.json"
FROZEN_SHA = "8ce2aa190847ed60e480811e59a05fd05ab7430f"
EVIDENCE_TYPES = ("URDF_COLLIDER_GEOMETRIC_EXCLUSION", "NATIVE_PHYSX_MANIFOLD_UNKNOWN")
ORIGINAL_V52_SOURCE_BY_CASE = {
    "frozen_v32": "v052_cmu_10_02_center_zero_physx.json",
    "cmu_10_02": "v052_cmu_10_02_center_physx.json",
    "cmu_10_05": "v052_cmu_10_05_center_physx.json",
}
ORIGINAL_SHARED_PHYSICAL_FIELDS = (
    "traces", "contacts", "kinematics", "records", "sequence_records",
    "running_evidence", "stride_shot_evidence", "stride_shot_success",
    "termination_reasons",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frozen_protocol() -> dict:
    original = subprocess.check_output([
        "git", "-C", str(ROOT), "show",
        FROZEN_SHA + ":configs/v128_frozen_v54_collision_event.json"
    ])
    if PROTOCOL.read_bytes() != original:
        raise ValueError("FROZEN_PROTOCOL_DRIFT")
    cfg = json.loads(original)
    if (cfg["version"] != "v0.1.128" or cfg["native_contact_pair_certified"]
            or not cfg["no_training_tuning_or_motor_actuation"]
            or not cfg["no_unobserved_event_extrapolation"]):
        raise ValueError("UNSAFE_OR_WRONG_PROTOCOL")
    return cfg


def require_sha(path: Path, expected: str, *, allow_symlink: bool = False) -> None:
    if not path.is_file() or (path.is_symlink() and not allow_symlink):
        raise ValueError("MISSING_OR_SYMLINK_SOURCE:" + str(path))
    if len(expected) != 64 or sha256(path) != expected:
        raise ValueError("SOURCE_SHA256_DRIFT:" + str(path))


def validate_rows(raw: dict, count: int, dt_s: float):
    if tuple(raw.get("joint_names", ())) != tuple(G1_JOINT_ORDER):
        raise ValueError("NOT_ORIGINAL_G1_29DOF")
    if raw.get("root_fixed") is not False or raw.get("seed") != 1201:
        raise ValueError("ORIGINAL_SEED_OR_ROOT_DRIFT")
    if len(raw.get("traces", ())) != 1 or len(raw["traces"][0]) != count:
        raise ValueError("MISSING_OR_TRUNCATED_REAL_PHYSICS")
    rows = raw["traces"][0]
    # v54's premature-termination writer appends a non-physics 5.140001s
    # terminal marker, copying exactly the physical state at 5.140000s.
    # It is NOT the missing 5.145s contact step; never count twice.
    terminal = 0
    if count == 1029:
        old, last = rows[-2:]
        if (abs(float(old["time_s"])-5.14) > 2e-6
                or abs(float(last["time_s"])-5.140001) > 2e-6
                or old["terminated"] is not False or last["terminated"] is not True
                or old.keys() != last.keys()
                or any(old[k] != last[k] for k in old if k not in ("time_s", "terminated"))):
            raise ValueError("UNQUALIFIED_NONPHYSICAL_TERMINATION_SENTINEL")
        rows = rows[:-1]
        terminal = 1
    try:
        t = np.asarray([r["time_s"] for r in rows], dtype=float)
        q = np.asarray([r["joint_pos"] for r in rows], dtype=float)
        dq = np.asarray([r["joint_vel"] for r in rows], dtype=float)
        com = np.asarray([r["whole_body_com_xyz"] for r in rows], dtype=float)
        support = np.asarray([[r["support_foot_force_N"][s] for s in ("left", "right")]
                              for r in rows], dtype=float)
        nonfoot = np.asarray([r["nonfoot_contact_N"] for r in rows], dtype=float)
        limits = np.asarray(raw["joint_velocity_limits"], dtype=float).reshape(-1)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("INVALID_REAL_29DOF_AND_FORCE_PROXIES") from exc
    physical_count = count - terminal
    if (q.shape != (physical_count, 29) or dq.shape != (physical_count, 29)
            or com.shape != (physical_count, 3) or support.shape != (physical_count, 2)
            or nonfoot.shape != (physical_count,) or limits.shape != (29,)
            or np.any(limits <= 0) or
            not all(np.isfinite(x).all() for x in (t, q, dq, com, support, nonfoot, limits))):
        raise ValueError("INVALID_REAL_29DOF_AND_FORCE_PROXIES")
    expected_t = dt_s * np.arange(1, physical_count + 1)
    if not np.allclose(t, expected_t, rtol=0, atol=2e-5):
        raise ValueError("NOT_ORIGINAL_200HZ_STEPS")
    impact = raw["records"][0].get("contact_time_s")
    if not isinstance(impact, (int, float)) or abs(impact - 1.57) > 0.01:
        raise ValueError("ORIGINAL_CONTACT_TIME_DRIFT")
    if raw.get("post_initialization_state_writes") is not False:
        raise ValueError("POST_INITIALIZATION_STATE_TAMPERING")
    return dict(t=t, q=q, dq=dq, com=com, support=support, nonfoot=nonfoot,
                limits=limits, impact=float(impact), terminal_sentinel=terminal,
                original_logged_rows=count)


def event_observability(t: np.ndarray, event_s: float, dt_s: float) -> dict:
    """No numerical extrapolation at a sensor time unobserved by joint traces."""
    t = np.asarray(t, dtype=float)
    if not t.size or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("INVALID_MONOTONIC_TIME")
    matching = np.flatnonzero(np.abs(t - event_s) < 1e-4)
    before = np.flatnonzero(t < event_s - 1e-4)
    if not len(before):
        raise ValueError("NO_PRIOR_200HZ_STATE")
    last = int(before[-1])
    gap = float(event_s - t[last])
    return {
        "event_has_directly_sampled_joint_state": bool(len(matching)),
        "event_frame_index": int(matching[0]) if len(matching) else None,
        "last_pre_event_frame_index": last,
        "last_pre_event_time_s": float(t[last]),
        "unobserved_elapsed_time_s": gap if not len(matching) else 0.0,
        "about_one_200hz_step_coverage_gap": bool(not len(matching) and abs(gap - dt_s) < 2e-5),
        "collision_at_unsampled_event_certified": False,
    }


def slice_stat(t, mask, gap, qdot, limits, nonfoot, com, support):
    indexes = np.flatnonzero(mask)
    if not len(indexes):
        raise ValueError("EMPTY_REAL_PHASE_WINDOW")
    i = int(indexes[np.argmin(gap[mask])])
    ratio = np.abs(qdot[mask]) / limits[None, :]
    return {
        "physical_frames": int(len(indexes)),
        "first_time_s": float(t[indexes[0]]),
        "last_time_s": float(t[indexes[-1]]),
        "minimum_capsule_gap_m": float(gap[i]),
        "minimum_capsule_gap_at_s": float(t[i]),
        "nonpositive_gap_frames": int(np.count_nonzero(gap[mask] <= 0)),
        "inside_20mm_alert_frames": int(np.count_nonzero(gap[mask] <= .02)),
        "original29_qdot_peak_limit_ratio": float(ratio.max()),
        "original29_qdot_exceedance_frames": int(np.count_nonzero(np.any(ratio > 1+1e-6,axis=1))),
        "nonfoot_net_force_peak_N": float(nonfoot[mask].max()),
        "left_foot_ground_force_proxy_integral_Ns": float(support[mask, 0].sum()*.005),
        "right_foot_ground_force_proxy_integral_Ns": float(support[mask, 1].sum()*.005),
        "COM_xy_displacement_m": float(np.linalg.norm(com[indexes[-1],:2] - com[indexes[0],:2])),
        "native_actor_collider_manifold_certified": False,
    }


def certify_force_pair(fail: dict, opposite: dict, cfg: dict) -> dict:
    contacts = fail.get("nonfoot_contact_diagnostics")
    if not isinstance(contacts, list) or len(contacts) != 2:
        raise ValueError("MISSING_OR_AMBIGUOUS_NAMED_SENSOR_NET_FORCES")
    names = {x["body"] for x in contacts}
    if names != set(cfg["failure_pair_names"]):
        raise ValueError("WRONG_SENSOR_LINK_NAMES")
    forces = np.asarray([x["force_w_N"] for x in contacts], dtype=float)
    times = np.asarray([x["time_s"] for x in contacts], dtype=float)
    positions = np.asarray([x["position_w"] for x in contacts], dtype=float)
    if (forces.shape != (2, 3) or positions.shape != (2, 3)
            or not np.isfinite(forces).all() or not np.isfinite(positions).all()
            or not np.allclose(times, cfg["true_nonfoot_event_t_s"], rtol=0, atol=2e-6)
            or not np.allclose(forces.sum(axis=0), 0, rtol=0, atol=1e-6)
            or not np.allclose(np.linalg.norm(forces, axis=1),
                               cfg["true_nonfoot_force_N"], rtol=0, atol=2e-5)):
        raise ValueError("NO_EQUAL_OPPOSITE_OR_TIME_MISMATCH")
    row = opposite["left_wrist_hip_symmetric_force_pair"]
    if row["body_pair_directly_measured_in_contact_manifold"] is not False:
        raise ValueError("FALSELY_CERTIFIED_NATIVE_CONTACT")
    if abs(row["force_magnitude_each_N"]-cfg["true_nonfoot_force_N"]) > 2e-5:
        raise ValueError("SENSOR_FORCE_PROVENANCE_DRIFT")
    return {
        "sensor_link_names": sorted(names),
        "net_force_pair_equal_opposite": True,
        "force_magnitude_each_N": float(np.linalg.norm(forces[0])),
        "sensor_event_s": float(times[0]),
        "measured_body_origin_separation_m": float(np.linalg.norm(positions[0]-positions[1])),
        "force_is_link_net_not_direct_contact_manifold": True,
        "native_pair_owner_unknown": True,
    }


def analyze_case(case: str, row_spec: dict, raw: dict, tree: dict, cfg: dict) -> tuple[dict, dict]:
    data = validate_rows(raw, row_spec["expected_trace_rows"], cfg["dt_s"])
    t = data["t"]
    gaps = geometry.clearance_for_episode(tree, data["q"], cfg["all_geometry_pairs"])
    primary = cfg["failure_pair_names"][0]+"__"+cfg["failure_pair_names"][1]
    if primary not in gaps or not all(np.isfinite(x).all() for x in gaps.values()):
        raise ValueError("INVALID_FROZEN_PRIMARY_GEOMETRY")
    masks = {
        "pre_contact_400ms": (t >= data["impact"]-cfg["impact_pre_s"]-1e-7)&(t < data["impact"]-1e-7),
        "post_contact_1250ms": (t >= data["impact"]-1e-7)&(t < data["impact"]+cfg["impact_post_s"]-1e-7),
        "last_100ms_before_original_v54_nonfoot_event":
            (t >= cfg["true_nonfoot_event_t_s"]-cfg["failure_event_pre_s"]-1e-7)&
            (t < cfg["true_nonfoot_event_t_s"]-1e-4),
    }
    if (masks["pre_contact_400ms"].sum()!=80 or masks["post_contact_1250ms"].sum()!=250
            or masks["last_100ms_before_original_v54_nonfoot_event"].sum()!=20):
        raise ValueError("INCOMPLETE_FROZEN_WINDOWS")
    by_phase={k:slice_stat(t,m,gaps[primary],data["dq"],data["limits"],
                           data["nonfoot"],data["com"],data["support"])
              for k,m in masks.items()}
    first_alert={}
    pair_results={}
    for key,values in gaps.items():
        alerts=np.flatnonzero(values <= .02)
        collision=np.flatnonzero(values <= 0)
        first_alert[key]=float(t[alerts[0]]) if len(alerts) else None
        pair_results[key]={
            "observed_min_lower_bound_m":float(values.min()),
            "observed_min_time_s":float(t[int(np.argmin(values))]),
            "observed_no_URDF_cylinder_contact_if_every_sample_positive":bool(np.all(values>0)),
            "observed_under_20mm_frame_count":int(len(alerts)),
            "observed_nonpositive_frame_count":int(len(collision)),
            "first_20mm_alert_s":first_alert[key],
            "native_physx_collider_pair_truthed":False,
        }
    last_index=np.flatnonzero(t < cfg["true_nonfoot_event_t_s"]-1e-4)[-1]
    force_peak_i=int(np.argmax(data["nonfoot"]))
    geom_min_i=int(np.argmin(gaps[primary]))
    result = {
        "case":case,"source_original_29DOF":True,"observed_physical_frames":int(len(t)),
        "original_logged_rows_including_terminal_sentinel":data["original_logged_rows"],
        "nonphysics_termination_marker_rows":data["terminal_sentinel"],
        "observed_end_s":float(t[-1]),"observed_impact_s":data["impact"],
        "original_termination_reasons":raw["termination_reasons"][0],
        "original_recovered":bool(raw["records"][0]["recovered"]),
        "original_record_reported_success":bool(raw["records"][0]["success"]),
        "original_goal_entered":bool(raw["records"][0]["goal"]),
        "original_reported_instep_impulse_share":
            float(raw["records"][0]["requested_surface_impulse_fraction"]),
        "source_original_joint_speed_ratio_max":
            float((np.abs(data["dq"])/data["limits"][None,:]).max()),
        "source_nonfoot_net_force_peak_N":float(data["nonfoot"].max()),
        "source_nonfoot_net_force_peak_time_s":float(t[force_peak_i]),
        "URDF_gap_at_logged_trace_nonfoot_peak_m":float(gaps[primary][force_peak_i]),
        "minimum_URDF_gap_time_s":float(t[geom_min_i]),
        "geometry_minimum_and_net_force_trace_peak_coincide_at_same_step":
            bool(geom_min_i == force_peak_i),
        "named_sensor_event_minus_trace_force_peak_s":
            float(cfg["true_nonfoot_event_t_s"]-t[force_peak_i]),
        "primary_pair":primary, "all_four_frozen_URDF_pairs":pair_results,
        "primary_pair_at_last_pre_event_observed_step_m":float(gaps[primary][last_index]),
        "real_29DOF_full_observed_time_window":by_phase,
        "directly_sampled_at_nonfoot_sensor_event":
            bool(np.any(abs(t-cfg["true_nonfoot_event_t_s"]) < 1e-4)),
        "source_original_trace_was_early_terminated":len(t)<2800,
        "native_PhysX_contact_actor_collider_manifold":None,
        "wholebody_professional_human_impact_supervision":False,
        "offline_actuation_authorized":False,
    }
    # Downsample only within 0.5s of known failure to draw diagnostic trajectories.
    m=(t>=cfg["true_nonfoot_event_t_s"]-.5-1e-7)&(t<=cfg["true_nonfoot_event_t_s"]+1e-7)
    curves=dict(time_s=t[m].tolist(),primary_pair_URDF_capsule_gap_m=gaps[primary][m].tolist(),
                original29_peak_qdot_ratio_per_frame=(np.abs(data["dq"][m])/data["limits"][None,:]).max(axis=1).tolist(),
                measured_nonfoot_body_net_force_peak_N=data["nonfoot"][m].tolist(),
                COM_z_m=data["com"][m,2].tolist(),
                ground_support_left_N=data["support"][m,0].tolist(),
                ground_support_right_N=data["support"][m,1].tolist())
    return result, curves


def run(cfg: dict) -> dict:
    originals=Path(cfg["data_root"])
    for name, expected in cfg["input_sha256"].items():
        require_sha(originals/name, expected)
    require_sha(Path(cfg["original_G1_urdf"]), cfg["original_G1_urdf_sha256"])
    tree = geometry.parse_tree(Path(cfg["original_G1_urdf"]))
    witness = json.loads((originals/"v054_original_named_nonfoot_audit.json").read_text())
    if not witness["all_physics_same_except_added_diagnostics"] or             not all(witness["results"][r["case"]]["parity_full_on_all_common_physics_records"]
                    for r in cfg["original_v54_cases"]):
        raise ValueError("ORIGINAL_V54_OFF_ON_PARITY_UNVERIFIED")
    opposite = json.loads((originals/"v054_opposite_nonfoot_link_forces.json").read_text())
    if opposite["direct_original_contact_pair_label_verified"] is not False:
        raise ValueError("FALSE_NATIVE_CONTACT_CLAIM")
    cases={};curves={}
    for spec in cfg["original_v54_cases"]:
        name=spec["case"]
        raw=json.loads((originals/("v054_"+name+"_contact_diagnostics.json")).read_text())
        original_path=originals/ORIGINAL_V52_SOURCE_BY_CASE[name]
        arow=witness["results"][name]
        require_sha(original_path,arow["source_sha256"])
        if arow["instrumented_sha256"] != cfg["input_sha256"]["v054_"+name+"_contact_diagnostics.json"]:
            raise ValueError("OLD_OR_NEW_REPLAY_HASH_DISAGREES_WITH_V54")
        old=json.loads(original_path.read_text())
        if any(old.get(k)!=raw.get(k) for k in ORIGINAL_SHARED_PHYSICAL_FIELDS):
            raise ValueError("ORIGINAL_OFF_ON_REPLAY_CHANGED_REAL_PHYSICS")
        if (old["policy_sha256"]!=raw["policy_sha256"]
                or old["joint_strike"]["checkpoint_sha256"]!=raw["joint_strike"]["checkpoint_sha256"]):
            raise ValueError("CONTROL_OR_ORIGINAL_WEIGHT_CHANGED")
        if raw.get("nonfoot_contact_diagnostics") is None:
            raise ValueError("MISSING_READONLY_SENSOR_RECORDS")
        if bool(raw["records"][0]["recovered"]) == spec["failure"]:
            raise ValueError("RECOVERY_OR_FAILURE_INVERTED")
        if (name=="cmu_10_05") != ("physical_nonfoot" in raw["termination_reasons"][0]):
            raise ValueError("UNEXPECTED_NONFOOT_TERMINATION")
        result, trend=analyze_case(name,spec,raw,tree,cfg)
        cases[name]=result;curves[name]=trend
        if name=="cmu_10_05":
            pair=certify_force_pair(raw,opposite,cfg)
    fail=cases["cmu_10_05"]
    obs=event_observability(np.asarray(curves["cmu_10_05"]["time_s"]),
                            cfg["true_nonfoot_event_t_s"],cfg["dt_s"])
    # Note: this function is given 0.5s curve, so index is diagnostic-curve-local.
    if obs["event_has_directly_sampled_joint_state"]:
        raise ValueError("UNEXPECTED_DIRECT_PHYSICS_STATE_AT_NONFOOT_EVENT")
    comparisons={}
    for case in ("frozen_v32","cmu_10_02"):
        comparisons[case]={
            "same_5p14s_pre_event_step_gap_difference_to_failed_m":
            (cases[case]["primary_pair_at_last_pre_event_observed_step_m"] -
             fail["primary_pair_at_last_pre_event_observed_step_m"]),
            "baseline_continues_to_14s":cases[case]["observed_physical_frames"]==2800,
            "baseline_has_no_nonfoot_termination":
                cases[case]["original_termination_reasons"]==[],
        }
    return {
        "version":cfg["version"],"frozen_protocol_commit":FROZEN_SHA,
        "inputs_all_SHA256_rechecked":True,
        "original_read_only_instrumentation_all_common_physics_fields_identical":True,
        "v128_independently_rechecked_v52_vs_v54_all_physical_fields":True,
        "v128_exact_common_physical_field_names":list(ORIGINAL_SHARED_PHYSICAL_FIELDS),
        "real_G1_PhysX_episodes_reused":3,
        "new_G1_PhysX_episodes":0,
        "original_total_observed_physics_steps":sum(x["observed_physical_frames"] for x in cases.values()),
        "original_total_logged_rows_including_terminal_marker":
            sum(x["original_logged_rows_including_terminal_sentinel"] for x in cases.values()),
        "original_reported_success_in_these_cases":
            sum(x["original_record_reported_success"] for x in cases.values()),
        "independent_all_original_hard_gates_reaudited":False,
        "v54_equal_opposite_net_forces":pair,
        "failure_at_event_time_joint_state_observability":obs,
        "trace_sensor_time_alignment_note":
            "Sensor event stamp 5.145s, nonfoot trace peak and minimum URDF "
            "capsule gap at 5.135s, last genuine 200Hz q at 5.140s; "
            "distinct clocks/logging paths, no inferred direct contact pair.",
        "original_failed_case_5p14s_vs_matched_control":comparisons,
        "per_case":cases, "measured_pre_event_curves":curves,
        "no_contact_pair_manifold_collider_paths_truth":True,
        "not_a_new_PhysX_replay":True,
        "not_a_collision_proof_from_force_pair":True,
        "physical_motor_command_issued":False,
        "human_high_performance_shot_learned":False,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    cfg=frozen_protocol()
    output=run(cfg)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as f:
        json.dump(output,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "created":str(a.output),"steps":output["original_total_observed_physics_steps"],
        "success":output["original_reported_success_in_these_cases"],
        "event_observability":output["failure_at_event_time_joint_state_observability"],
        "case_primary_min":{n:x["all_four_frozen_URDF_pairs"][x["primary_pair"]]["observed_min_lower_bound_m"]
                            for n,x in output["per_case"].items()},
        "last_pre_event_URDF_gap":{n:x["primary_pair_at_last_pre_event_observed_step_m"]
                                  for n,x in output["per_case"].items()},
        "force_pair":output["v54_equal_opposite_net_forces"]},indent=2))


if __name__ == "__main__":
    main()
