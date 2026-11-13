#!/usr/bin/env python3
"""Original v54 *real* 200Hz G1 dynamics reconstruction, without rerunning GPU.

Read SHA-verified raw historical PhysX output from the original machine only.
Never manufacture a missing next-step state, true wrist-hip collider path, GRF,
or an actual high-power human instep label. The aborted trial has a duplicate
terminal annotation at 5.140001s: NOT a 1,029th 5ms physics sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import v142_native_pair_ledger as evidence
from v146_native_original_bridge import frozen, PHYSICS_FIELDS, json_sha

SOURCE_ROOT = Path("/home/you/projects/kinetickick")
PROGRESS = SOURCE_ROOT / "reports/v054_contact_diag_progress.json"
ORIGINAL_V54_CFG = SOURCE_ROOT / "configs/cmu_named_nonfoot_diagnostics_v054.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_sources(root=SOURCE_ROOT):
    cfg = frozen()
    root = Path(root)
    protocol = root / "configs/cmu_named_nonfoot_diagnostics_v054.json"
    pr = root / "reports/v054_contact_diag_progress.json"
    if sha(protocol) != "add8693911cdd2489f3fb944adb77698238ddca50f14867a2a388b31eaeeeefa":
        raise ValueError("HISTORICAL_V54_PROTOCOL_SHA_DRIFT")
    sources = json.loads(pr.read_text())
    if (sources["version"] != "0.1.54" or sources["protocol_sha256"] != sha(protocol)):
        raise ValueError("HISTORICAL_V54_PROGRESS_DRIFT")
    cases = cfg["original_v54_existing_episodes"]
    if [r["name"] for r in sources["runs"]] != list(cases):
        raise ValueError("HISTORICAL_CASE_ORDER_CHANGED")
    for row in sources["runs"]:
        p = root / row["result"]
        if not p.is_file() or sha(p) != row["sha256"] or p.stat().st_size != row["bytes"]:
            raise ValueError("HISTORICAL_PHYSX_SOURCE_HASH_MISMATCH:" + row["name"])
    return {r["name"]: root / r["result"] for r in sources["runs"]}


def num(x, tag):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError("NONFINITE_HISTORICAL_" + tag)
    return float(x)


def vec(x, n, tag):
    if not isinstance(x, list) or len(x) != n:
        raise ValueError("INCOMPLETE_HISTORICAL_" + tag)
    return [num(v, tag) for v in x]


def split_real_5ms_rows(trace, expected_count):
    """Terminal side-note at duplicate timestamp is not a physics step."""
    if not isinstance(trace, list) or len(trace) < expected_count:
        raise ValueError("MISSING_200HZ_HISTORICAL_TRACE")
    dt = frozen()["physics_dt_s"]
    rows = trace[:expected_count]
    for i, r in enumerate(rows, start=1):
        if abs(num(r["time_s"], "TIME") - dt*i) > 1e-6:
            raise ValueError("DROPPED_OR_DUPLICATE_200HZ_PHYSICS_STEP")
        if abs(num(r["physics_dt_s"], "DT") - dt) > 1e-8:
            raise ValueError("NOT_ORIGINAL_200HZ_PHYSICS")
        for key in ("joint_pos", "joint_vel", "joint_torque_estimate"):
            vec(r[key], 29, key)
        for key in ("whole_body_com_xyz", "whole_body_com_velocity",
                    "ball_xyz", "ball_velocity", "right_foot_point_velocity"):
            vec(r[key], 3, key)
        for key in ("feet_support_N", "nonfoot_contact_N"):
            num(r[key], key)
        if (not isinstance(r["support_foot_force_N"], dict)
                or set(r["support_foot_force_N"]) != {"left", "right"}):
            raise ValueError("INCOMPLETE_TWO_FOOT_NET_SUPPORT")
        for value in r["support_foot_force_N"].values():
            num(value, "FOOT_NET")
    terminal = trace[expected_count:]
    if terminal:
        if expected_count != 1028 or len(terminal) != 1:
            raise ValueError("UNREGISTERED_EXTRA_PHYSICS_ROWS")
        if (not terminal[0].get("terminated")
                or abs(float(terminal[0]["time_s"]) - rows[-1]["time_s"]) > 1e-4):
            raise ValueError("NOT_KNOWN_ORIGINAL_TERMINAL_ANNOTATION")
    return rows, terminal


def verify_g1_metadata(data):
    from wholebody_safety_interface_v055 import G1_JOINT_ORDER
    names = data["joint_names"]
    if tuple(names) != G1_JOINT_ORDER:
        raise ValueError("ORIGINAL_29_JOINT_ORDER_CHANGED")
    limits = data["joint_velocity_limits"]
    if not isinstance(limits, list) or len(limits) != 1:
        raise ValueError("INVALID_ORIGINAL_29_QDOT_LIMITS")
    velocity_limits = vec(limits[0], 29, "ORIGINAL_QDOT_LIMITS")
    if min(velocity_limits) <= 0:
        raise ValueError("NONPOSITIVE_ORIGINAL_QDOT_LIMIT")
    torque_limits = vec(data["joint_effort_limits"][0], 29, "ORIGINAL_EFFORT_LIMITS")
    if min(torque_limits) <= 0:
        raise ValueError("NONPOSITIVE_ORIGINAL_EFFORT_LIMIT")
    return velocity_limits, torque_limits


def coupled_rows(rows, velocity_limits, torque_limits):
    out = []
    for i, r in enumerate(rows, start=1):
        q = vec(r["joint_pos"],29,"Q")
        dq = vec(r["joint_vel"],29,"DQ")
        tau = vec(r["joint_torque_estimate"],29,"TAU")
        com = vec(r["whole_body_com_xyz"],3,"COM")
        com_v = vec(r["whole_body_com_velocity"],3,"COM_V")
        feet = r["support_foot_force_N"]
        foot = {name:num(feet[name],"FOOT_NET") for name in ("left","right")}
        ratio = [abs(v)/lim for v,lim in zip(dq,velocity_limits)]
        effort_ratio = [abs(v)/lim for v,lim in zip(tau,torque_limits)]
        out.append({
            "physics_step":i, "time_s":r["time_s"],
            "q29":q, "qdot29":dq, "sim_torque29":tau,
            "qdot_ratio29":ratio,
            "estimated_effort_ratio29":effort_ratio,
            "com_world":com, "com_velocity_world":com_v,
            "left_foot_net_contact_N":foot["left"],
            "right_foot_net_contact_N":foot["right"],
            "both_foot_net_contact_N":num(r["feet_support_N"],"NET_FEET"),
            "nonfoot_net_contact_N":num(r["nonfoot_contact_N"],"NET_NONFOOT"),
            "root_xyz":vec(r["root_xyz"],3,"ROOT"),
            "root_tilt_deg":num(r["root_tilt_deg"],"ROOT_TILT"),
            "ball_xyz":vec(r["ball_xyz"],3,"BALL"),
            "ball_velocity":vec(r["ball_velocity"],3,"BALL_V"),
            "right_foot_point_velocity":vec(r["right_foot_point_velocity"],3,"RIGHT_FOOT_V"),
            "phase":str(r["coordination_phase"]),
            "estimated_torque_not_measured_hardware":True,
            "net_foot_contact_not_isolated_grf":True,
            "arm_hip_true_collider_pair":None,
        })
    return out


def windows(full_rows, contact_t, *, before_s=.4, after_s=1.25):
    """Contact is original recorded foot-ball timestamp, NOT human contact."""
    dt=.005
    if contact_t is None:raise ValueError("MISSING_TRUE_ROBOT_FOOT_CONTACT")
    center=int(round(contact_t/dt))
    if (abs(center*dt-contact_t)>1e-6
            or center-80<1 or center+250>len(full_rows)):
        raise ValueError("INSUFFICIENT_REAL_PRE_POST_CONTACT_WINDOW")
    pre=full_rows[center-81:center-1]
    after=full_rows[center:center+250]
    if len(pre)!=80 or len(after)!=250:
        raise ValueError("WINDOW_SAMPLE_COUNT_DRIFT")
    return {"before_steps":80,"after_steps":250,
            "first_before_time_s":pre[0]["time_s"],
            "last_before_time_s":pre[-1]["time_s"],
            "first_after_time_s":after[0]["time_s"],
            "last_after_time_s":after[-1]["time_s"],
            "max_pre_qdot_ratio":max(max(x["qdot_ratio29"]) for x in pre),
            "max_post_qdot_ratio":max(max(x["qdot_ratio29"]) for x in after),
            "max_post_nonfoot_net_N":max(x["nonfoot_net_contact_N"] for x in after),
            "min_post_left_net_contact_N":min(x["left_foot_net_contact_N"] for x in after),
            "min_post_right_net_contact_N":min(x["right_foot_net_contact_N"] for x in after),
            "pre_or_post_foot_force_is_net_not_grf":True}


def case_audit(data, *, name, expected):
    qdot_limits, effort_limits = verify_g1_metadata(data)
    original, terminal = split_real_5ms_rows(data["traces"][0], expected)
    contact_time = num(data["records"][0]["contact_time_s"], "CONTACT_TIME")
    rows = coupled_rows(original, qdot_limits, effort_limits)
    return {
        "policy":name,
        "original_real_5ms_steps":len(rows),
        "terminal_non_5ms_annotations_not_recounted":len(terminal),
        "last_actual_5ms_state_time_s":rows[-1]["time_s"],
        "real_14s_coverage":len(rows)==2800,
        "recorded_original_foot_ball_contact_s":contact_time,
        "window":windows(rows,contact_time),
        "qdot_limit_violating_steps":sum(max(x["qdot_ratio29"])>1 for x in rows),
        "peak_original29_qdot_limit_ratio":max(max(x["qdot_ratio29"]) for x in rows),
        "peak_estimated_torque_limit_ratio":max(max(x["estimated_effort_ratio29"]) for x in rows),
        "peak_nonfoot_net_N":max(x["nonfoot_net_contact_N"] for x in rows),
        "max_com_speed_mps":max(math.sqrt(sum(v*v for v in x["com_velocity_world"])) for x in rows),
        # These two v54 legacy success flags are NOT the frozen v53 4-ball
        # mandatory all-gates result, nor a new model evaluation.
        "v54_legacy_center_record_success_flag":bool(data["records"][0]["success"]),
        "v54_legacy_center_stride_shot_success_flag":bool(data["stride_shot_success"]),
        "new_full_original29_instep_physics_gate_certified":False,
        "left_wrist_hip_native_pair":"UNKNOWN",
        "recovery_if_missing_14s":"UNKNOWN",
        "full_real_200hz_coupled_rows":rows,
    }


def interpolicy_divergence(results):
    """Compare actual same-center original G1 state at every shared 200Hz tick."""
    keys=list(results)
    out={}
    for aa,bb in ((keys[0],keys[1]),(keys[0],keys[2]),(keys[1],keys[2])):
        a=results[aa]["full_real_200hz_coupled_rows"]
        b=results[bb]["full_real_200hz_coupled_rows"]
        shared=min(len(a),len(b))
        def first_difference(field):
            for i in range(shared):
                if a[i][field]!=b[i][field]:
                    return round(a[i]["time_s"],6)
            return None
        # Source-locked qualitative tests of whether the 29DoF original
        # physical controllers actually diverge BEFORE first foot-ball contact.
        pre=[(a[i],b[i]) for i in range(shared)
             if 1.405-1e-8 <= a[i]["time_s"] < 1.57-1e-8]
        group_idx={
            "legs":[0,1,3,4,6,7,9,10,13,14,17,18],
            "waist":[2,5,8],
            "arms":[11,12,15,16,19,20,21,22,23,24,25,26,27,28],
        }
        groups={}
        for g,indices in group_idx.items():
            groups[g]={
                "max_abs_29q_difference_rad":max(
                    abs(x["q29"][k]-y["q29"][k]) for x,y in pre for k in indices),
                "max_abs_29qdot_difference_rad_s":max(
                    abs(x["qdot29"][k]-y["qdot29"][k]) for x,y in pre for k in indices),
                "max_abs_sim_torque_difference_Nm":max(
                    abs(x["sim_torque29"][k]-y["sim_torque29"][k])
                    for x,y in pre for k in indices),
            }
        out[aa+"__"+bb]={
            "shared_original_physics_steps":shared,
            "pre_contact_after_first_divergence_shared_steps":len(pre),
            "pre_contact_original29_body_group_differences":groups,
            "first_different_original29_q_time_s":first_difference("q29"),
            "first_different_original29_qdot_time_s":first_difference("qdot29"),
            "first_different_com_time_s":first_difference("com_world"),
            "first_different_support_pair_time_s":first_difference("left_foot_net_contact_N"),
            "pre_contact_1p57_q_identical_steps":sum(
                a[i]["q29"]==b[i]["q29"] for i in range(min(314,shared))),
            "same_ball_and_no_policy_ablations_certified":False,
        }
    return out


def audit_all(source_root=SOURCE_ROOT):
    cfg = frozen()
    files=verified_sources(source_root)
    prior = evidence.historical_recheck()
    if prior["historical_physics_5ms_frames"] != 6628:
        raise ValueError("ORIGINAL_PHYSICS_6628_SOURCE_CHANGED")
    results={}
    for name,expected in cfg["original_v54_existing_episodes"].items():
        d=json.loads(files[name].read_text())
        x=case_audit(d,name=name,expected=expected)
        results[name]=x
    return {
        "schema":"v146_original_physics_cpu_reread_not_new_rollout",
        "original_historical_episodes":3,
        "historical_5ms_frames":sum(x["original_real_5ms_steps"] for x in results.values()),
        "excluded_terminal_duplicate_rows":sum(x["terminal_non_5ms_annotations_not_recounted"] for x in results.values()),
        "new_physx_episodes":0,
        "native_wrist_hip_contact_manifold_truth":"UNKNOWN",
        "left_right_net_foot_forces_not_individual_ground_reaction_forces":True,
        "wholebody_arm_leg_contact_distance_missing_in_original_5ms_trace":True,
        "professional_human_impact_or_grf_supervision_certified":False,
        "new_policy_success_rate":None,
        "interpolicy_original_center_ball_comparison":interpolicy_divergence(results),
        "cases":results,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    result=audit_all()
    digest=evidence.safe_write_once(a.output,result)
    compact={name:{k:v for k,v in case.items() if k!="full_real_200hz_coupled_rows"}
             for name,case in result["cases"].items()}
    print(json.dumps({"output":str(a.output),"sha256":digest,
                      "original_physics_states":result["historical_5ms_frames"],
                      "terminal_annotation_excluded":result["excluded_terminal_duplicate_rows"],
                      "interpolicy":result["interpolicy_original_center_ball_comparison"],
                      "cases":compact},ensure_ascii=False))


if __name__=="__main__":
    main()
