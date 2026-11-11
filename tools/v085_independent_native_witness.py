#!/usr/bin/env python3
"""Independent postregistered native witness and positive-control calibration, no USD calls."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"reports/v085_g1_physics_final.audit.json"
DEST=ROOT/"reports/v085_native_witness_independent.audit.json"
WATCH=("left_wrist_yaw_link","left_hip_roll_link")
FOOTBALL=("right_ankle_roll_link","SoccerBall")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def exact_segment_match(path, label):
    return label in str(path).split("/")
def matching_pair(event,pair):
    for key in ("actor_paths","collider_paths"):
        p=event.get(key,[])
        if len(p)!=2:continue
        if (exact_segment_match(p[0],pair[0]) and exact_segment_match(p[1],pair[1]) or
            exact_segment_match(p[0],pair[1]) and exact_segment_match(p[1],pair[0])):
            return True
    return False
def positive_geometry(event):
    for c in event.get("contacts",()):
        xyz=c.get("point_w");n=c.get("normal_w")
        if not (isinstance(xyz,list) and isinstance(n,list) and len(xyz)==3 and len(n)==3):
            continue
        try:
            nums=[float(v) for v in xyz+n]
            if not all(math.isfinite(v) for v in nums):continue
            if sum(v*v for v in nums[-3:])<0.01:continue
            values=c.get("impulse_w")
            scalar=c.get("impulse_scalar")
            valid_vector=bool(isinstance(values,list) and len(values)==3 and
                all(math.isfinite(float(v)) for v in values) and
                sum(float(v)**2 for v in values)>0)
            valid_scalar=bool(isinstance(scalar,(float,int)) and
                math.isfinite(float(scalar)) and scalar>0)
            if valid_vector or valid_scalar:return True
        except (ValueError,TypeError,OverflowError):continue
    return False
def assess(raw):
    original=raw.get("records",[{}])[0]
    contact_t=original.get("contact_time_s")
    sampled=raw.get("traces",[[]])[0]
    actual_recorded_contact=bool(
        isinstance(contact_t,(int,float)) and math.isfinite(contact_t) and
        any(abs(float(r["time_s"])-contact_t)<=.03 and
            float(r.get("contact_force_N",0))>0 for r in sampled))
    obs=raw.get("physx_native_report_audit_v085")
    if obs is None:return dict(observer_present=False,conclusion="NOT_OBSERVED")
    if obs.get("errors"):return dict(observer_present=True,conclusion="OBSERVER_ERROR",
                                     errors=obs["errors"])
    entries=obs.get("saved_interesting_events",[])
    foot=[v for v in entries if matching_pair(v,FOOTBALL) and positive_geometry(v)]
    wh=[v for v in entries if matching_pair(v,WATCH) and positive_geometry(v)]
    inventory=obs.get("report_api_inventory",{})
    return dict(observer_present=True, original_ball_contact_event=actual_recorded_contact,
        physical_contact_time_s=contact_t,
        native_callback_calls=obs.get("callback_calls"),native_callback_headers=obs.get("callback_headers"),
        native_poll_calls=obs.get("poll_calls"),native_poll_headers=obs.get("poll_headers"),
        native_positive_control_count=len(foot),native_wrist_hip_witness_count=len(wh),
        stage_report_api_and_threshold_inventory=inventory,
        native_positive_control_calibrated=bool(foot and not obs.get("dropped_events")),
        direct_native_wrist_hip_pair_certified=bool(wh and not obs.get("dropped_events")),
        conclusion=("NATIVE_WRIST_HIP_COLLIDER_WITNESS" if wh else
            "UNQUALIFIED_REPORTER: NO_NATIVE_FOOT_BALL_POSITIVE_CONTROL" if not foot else
            "CALIBRATED_FOOT_BALL_ONLY__WRIST_HIP_STILL_UNKNOWN"),
        never_interpret_zero_native_as_no_collision=True)
def report():
    batch=json.loads(SOURCE.read_text());entries=[]
    for r in batch["runs"]:
        if not r["true_original_200Hz_G1_CUDA_PhysX"]:continue
        path=ROOT/"reports"/("v085_"+r["name"]+".json")
        if sha(path)!=r["raw_sha256"]:raise ValueError("SOURCE_SHA_CHANGED:"+r["name"])
        d=json.loads(path.read_text())
        entries.append({"name":r["name"],"raw_sha256":r["raw_sha256"], **assess(d)})
    result={"version":"v0.1.85","postregistered_independent":True,
        "source_batch_sha256":sha(SOURCE),
        "read_only_original_episodes_checked":len(entries),
        "observed_original_contact_but_uncalibrated":sum(
            bool(e.get("original_ball_contact_event")) and e.get("conclusion","").startswith("UNQUALIFIED")
            for e in entries),
        "native_direct_wrist_hip_certified":any(e.get("direct_native_wrist_hip_pair_certified",False) for e in entries),
        "records":entries}
    if DEST.exists():raise FileExistsError(DEST)
    DEST.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("V085_NATIVE_WITNESS",json.dumps({k:v for k,v in result.items() if k!="records"}))
if __name__=="__main__":report()
