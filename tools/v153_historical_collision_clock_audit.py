#!/usr/bin/env python3
"""Read SHA-frozen historical original G1 PhysX; audit body net force and clocks.

This does not turn a ContactSensor net-force vector into native pair/manifold
truth. The named-body diagnostic is after the last original physical sample.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from v142_native_pair_ledger import safe_write_once
from v146_real_200hz_coupled_audit import (
    verified_sources, split_real_5ms_rows, verify_g1_metadata,
)
from v153_contact_graph_identifiability import frozen, vector, magnitude

SOURCE_ROOT=Path("/home/you/projects/kinetickick")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def timeline_relation(spike_s, terminal_s, named_s, dt):
    vals=(spike_s,terminal_s,named_s,dt)
    if any(type(x) not in (float,int) or not math.isfinite(x) for x in vals):
        raise ValueError("V153_NONFINITE_EVENT_CLOCK")
    if dt<=0 or spike_s<0 or terminal_s<spike_s:
        raise ValueError("V153_INVALID_PHYSICS_CLOCK_ORDER")
    if named_s <= terminal_s+1e-6:
        raise ValueError("V153_NAMED_EVENT_NOT_CENSORED")
    if named_s-spike_s < dt+1e-6:
        raise ValueError("V153_EXPECTED_CROSS_STEP_AMBIGUITY")
    return {
        "nonfoot_net_spike_at_real_trace_s":spike_s,
        "last_original_recorded_5ms_state_s":terminal_s,
        "named_body_sensor_diag_s":named_s,
        "gap_named_to_nonfoot_spike_ms":(named_s-spike_s)*1000,
        "gap_named_beyond_last_real_row_ms":(named_s-terminal_s)*1000,
        "named_diag_has_same_step_full_original29_state":False,
        "named_sensor_pair_proves_contact_collider_paths":False,
        "native_physx_contact_pair_truth":"UNKNOWN",
    }

def assert_recorded_pair(named):
    cfg=frozen()
    expected=cfg["nonfoot_body_pair"]
    if not isinstance(named,list) or len(named)!=2 or {x.get("body") for x in named}!=set(expected):
        raise ValueError("V153_REQUIRED_NAMED_BODY_PAIR_NOT_PRESENT")
    index={r["body"]:r for r in named}
    wrist=index[expected[0]]
    hip=index[expected[1]]
    wa=vector(wrist["force_w_N"])
    hb=vector(hip["force_w_N"])
    tol=1e-6
    if not magnitude(wa)>0 or any(abs(a+b)>tol for a,b in zip(wa,hb)):
        raise ValueError("V153_NAMED_FORCES_NOT_ANTI_SYMMETRIC")
    wt=float(wrist["time_s"]); ht=float(hip["time_s"])
    if abs(wt-ht)>1e-6 or abs(wt-cfg["named_body_event_time_s"])>1e-5:
        raise ValueError("V153_DIAG_EVENT_CLOCK_DRIFT")
    wp=vector(wrist["position_w"])
    hp=vector(hip["position_w"])
    return {
        "named_pair_verification":"NET_FORCE_ONLY",
        "observed_wrist_net_force_N":wa,
        "observed_hip_net_force_N":hb,
        "observed_wrist_force_norm_N":magnitude(wa),
        "wrist_hip_link_origin_distance_m":magnitude([a-b for a,b in zip(wp,hp)]),
        "named_event_s":wt,
        "not_narrow_phase_contact_position":True,
        "no_verified_original29_state_at_named_event":True,
        "possible_net_contact_paths_are_not_identified":True,
    }

def section_summary(name,source,expected):
    data=json.loads(source.read_text())
    qlim,tlim=verify_g1_metadata(data)
    original,notes=split_real_5ms_rows(data["traces"][0],expected)
    if len(notes)!=(1 if name=="cmu_10_05" else 0):
        raise ValueError("V153_UNEXPECTED_TERMINAL_ANNOTATION")
    peak=max(enumerate(original),key=lambda x:x[1]["nonfoot_contact_N"])
    peak_i,peak_row=peak
    violations=0
    max_speed_ratio=0.
    max_effort_ratio=0.
    one_foot_steps=0
    max_com_speed=0.
    for r in original:
        v=vector(r["whole_body_com_velocity"])
        max_com_speed=max(max_com_speed,magnitude(v))
        ratios=[abs(float(q))/lim for q,lim in zip(r["joint_vel"],qlim)]
        tratio=[abs(float(t))/lim for t,lim in zip(r["joint_torque_estimate"],tlim)]
        if max(ratios)>1:violations+=1
        max_speed_ratio=max(max_speed_ratio,max(ratios))
        max_effort_ratio=max(max_effort_ratio,max(tratio))
        support=r["support_foot_force_N"]
        if (support["left"]<=1e-9) != (support["right"]<=1e-9):
            one_foot_steps+=1
    return data,original,{
        "name":name,"source_sha256":sha(source),
        "original_real_physx_200hz_rows":len(original),
        "excluded_nonstep_terminal_annotations":len(notes),
        "full_14s_physics_completed":len(original)==2800,
        "last_recorded_real_physics_time_s":original[-1]["time_s"],
        "peak_nonfoot_net_N":peak_row["nonfoot_contact_N"],
        "peak_nonfoot_net_real_physics_s":peak_row["time_s"],
        "qdot_over_original_limit_steps":violations,
        "max_original29_velocity_ratio":max_speed_ratio,
        "max_sim_estimated_torque_ratio":max_effort_ratio,
        "only_one_net_support_foot_positive_steps":one_foot_steps,
        "max_com_velocity_norm_mps":max_com_speed,
        "support_force_is_aggregate_contact_not_isolated_grf":True,
        "no_true_native_collider_pair_observed":True,
    }

def audit_all():
    cfg=frozen()
    sources=verified_sources()
    data_by_name={}
    audits=[]
    total=0
    for name,expected in cfg["expected_historical_rows"].items():
        data,rows,details=section_summary(name,sources[name],expected)
        audits.append(details)
        total+=len(rows)
        data_by_name[name]=(data,rows)
    if total!=6628:raise ValueError("V153_ORIGINAL_PHYSX_STATE_COUNT_DRIFT")
    failed,rows=data_by_name["cmu_10_05"]
    pair=assert_recorded_pair(failed.get("nonfoot_contact_diagnostics"))
    if abs(pair["observed_wrist_force_norm_N"]-max(r["nonfoot_contact_N"] for r in rows))>2e-5:
        raise ValueError("V153_RECORDED_FORCE_MAGNITUDE_NOT_SUPPORTED")
    spike=next(r for r in rows if abs(r["time_s"]-cfg["peak_nonfoot_trace_time_s"])<1e-7)
    if abs(spike["nonfoot_contact_N"]-pair["observed_wrist_force_norm_N"])>2e-5:
        raise ValueError("V153_NAMED_AND_NET_PEAK_MAGNITUDE_DRIFT")
    if abs(rows[-1]["time_s"]-cfg["last_real_trace_time_s"])>1e-7:
        raise ValueError("V153_LAST_REAL_TRACE_TIME_DRIFT")
    timeline=timeline_relation(spike["time_s"],rows[-1]["time_s"],pair["named_event_s"],cfg["physics_dt_s"])
    # Real last 0.4 seconds (up to and including the terminal 5ms row).
    sub=[r for r in rows if r["time_s"] >= rows[-1]["time_s"]-.4-1e-7]
    if len(sub)!=81:raise ValueError("V153_LAST_400MS_WINDOW_INCOMPLETE")
    timeline["last_400ms_actual_step_count"]=len(sub)
    timeline["min_foot_support_N_over_400ms"]=min(r["feet_support_N"] for r in sub)
    timeline["max_nonfoot_net_contact_N_over_400ms"]=max(r["nonfoot_contact_N"] for r in sub)
    timeline["last_400ms_peak_29dof_joint_speed"]=max(
        abs(v) for r in sub for v in r["joint_vel"])
    timeline["last_400ms_com_speed_max_mps"]=max(
        magnitude(r["whole_body_com_velocity"]) for r in sub)
    timeline["root_pose_does_not_prove_fall"]=True
    return {"version":"v0.1.153","historical_real_physics_reanalyzed":True,
            "new_physx_runs":0,
            "original_real_physics_200hz_rows":total,
            "historical_failed_trial_sha256":sha(sources["cmu_10_05"]),
            "source_protocol_sha256":sha(SOURCE_ROOT/"configs/cmu_named_nonfoot_diagnostics_v054.json"),
            "historical_cases":audits,"last_400ms_censored_trace_timeline":timeline,
            **pair,"actual_left_wrist_hip_physx_contact_pair":"UNKNOWN",
            "human_qualified_contact_and_grf":False,
            "strict_new_policy_shot_success":None}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=audit_all()
    checksum=safe_write_once(a.output,report)
    print(json.dumps({"original_200hz_states":report["original_real_physics_200hz_rows"],
                      "force_norm_N":report["observed_wrist_force_norm_N"],
                      "named_event_s":report["named_event_s"],
                      "last_real_state_s":report["last_400ms_censored_trace_timeline"]["last_original_recorded_5ms_state_s"],
                      "output_sha256":checksum,"new_physx":0}))

if __name__=="__main__":
    main()
