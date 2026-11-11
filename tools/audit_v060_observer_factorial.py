#!/usr/bin/env python3
"""Independent v060 original-G1 frozen four-factor observer qualification.

No new physics; each certified pair requires real matching actor/collider paths
and a valid native point. All invalid/missing cases are excluded from success
denominator, not counted as failures. No GPU/CPU parity assumption.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from audit_v059_observer_parity import compare_raw, DYNAMIC_FIELDS
from summarize_v058_censored_g1 import pair_points

ROOT=Path(__file__).resolve().parents[1]
DEVICE=re.compile(r"Environment device\s*:\s*(cpu|cuda:\d+)")

def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def assert_complete_physics_fields(frames:list[dict])->None:
    """Parity is invalid if both rollouts silently omit the same measured field."""
    required=set(DYNAMIC_FIELDS)
    for i,frame in enumerate(frames):
        missing=required.difference(frame)
        if missing:
            raise ValueError(f"Missing recorded 200Hz physical fields at row {i}: {sorted(missing)}")
        if len(frame["joint_pos"])!=29 or len(frame["joint_vel"])!=29:
            raise ValueError(f"Incomplete original 29DoF state at row {i}")

def audit(root:Path)->dict:
    cfgpath=root/"configs/g1_observer_factorial_v060.json"
    cfg=json.loads(cfgpath.read_text())
    lock=json.loads((root/"reports/v060_protocol_lock.json").read_text())
    progress=json.loads((root/"reports/v060_batch_progress.json").read_text())
    digest=sha(cfgpath)
    if len(cfg["cases"])!=10 or len(progress["runs"])!=10:
        raise ValueError("Case matrix incomplete")
    if lock["protocol_sha256"]!=digest or progress["protocol_sha256"]!=digest:
        raise ValueError("Frozen protocol drift")
    if cfg["parent_sha"]!="4107bebb4ee4604d2316acf3a2bfad86a49ac339":
        raise ValueError("Parent SHA drift")
    for name,expected in cfg["source_hashes"].items():
        if sha(root/name)!=expected:
            raise ValueError("Source SHA drift "+name)
    raw={}
    rows=[]
    for case,result in zip(cfg["cases"],progress["runs"]):
        if case["id"]!=result["id"]:
            raise ValueError("Case order drift")
        if result["observer"]!=case["observer"] or result["backend_requested"]!=case["backend"]:
            raise ValueError("Frozen assignment drift")
        log=root/case["output"].replace(".json",".log")
        if not log.exists() or sha(log)!=result["log_sha256"]:
            raise ValueError("Log missing or changed")
        runtime=DEVICE.findall(log.read_text(errors="replace"))
        device=runtime[0] if runtime else None
        if device!=case["backend"] or result["backend_observed"]!=device:
            raise ValueError("Runtime backend differs")
        path=root/case["output"]
        if path.exists()!=bool(result.get("raw_sha256")):
            raise ValueError("Raw data presence drift")
        if path.exists() and sha(path)!=result["raw_sha256"]:
            raise ValueError("Original G1 physics JSON changed")
        valid=result["status"]=="VALID_G1" and result["exit_code"]==0 and path.exists()
        entry={"id":case["id"],"observer":case["observer"],
               "backend":device,"policy":case["policy"],"seed":case["seed"],
               "status":result["status"],"exit_code":result["exit_code"],"valid":valid,
               "strict_success":result.get("strict_success") if valid else None,
               "raw_sha":result.get("raw_sha256")}
        if valid:
            r=json.loads(path.read_text())
            if r["root_fixed"] is not False or len(r["joint_names"])!=29:
                raise ValueError("Not original free-root 29DoF G1")
            frames=r["traces"][0]
            if len(frames)<30 or any(len(f["joint_vel"])!=29 for f in frames):
                raise ValueError("Missing original 200Hz G1 joint state")
            assert_complete_physics_fields(frames)
            end=frames[-1]
            marker=bool(end.get("terminated") is True
                        and 0<end["time_s"]-frames[-2]["time_s"]<.003)
            physics=frames[:-1] if marker else frames
            if any(b["time_s"]-a["time_s"]<.003 or b["time_s"]-a["time_s"]>.007
                   for a,b in zip(physics,physics[1:])):
                raise ValueError("Not 200Hz original frames")
            limits=r["joint_velocity_limits"][0]
            if len(limits)!=29 or min(limits)<=0:
                raise ValueError("Invalid original joint limits")
            ratio=max(abs(x)/lim for frame in physics for x,lim in zip(frame["joint_vel"],limits))
            contact=r["records"][0].get("contact_time_s")
            r0=r["records"][0]
            entry.update(physical_frames=len(physics),
                         trace_terminal_status_marker=marker,
                         contact_s=contact, end_s=physics[-1]["time_s"],
                         post_1p25_complete=contact is not None and physics[-1]["time_s"]>=contact+1.25,
                         goal=r0["goal"],instep_fraction=r0["requested_surface_impulse_fraction"],
                         qdot_peak_ratio=ratio,
                         termination_reasons=r["termination_reasons"])
            if case["observer"] in ("full","native_only"):
                native=r["physx_native_report_v056"]
                if native.get("errors") or native.get("dropped_events") or not native.get("report_capture_complete"):
                    raise ValueError("Native reporter lost events")
                pair_pelvis=pair_points(native)
                pair_wrist=pair_points(native,{"left_wrist_yaw_link","left_hip_roll_link"})
                entry.update(native_headers=native.get("native_headers"),
                             native_callback_calls=native.get("callback_calls"),
                             native_pelvis_right_hip_points=len(pair_pelvis),
                             native_wrist_left_hip_points=len(pair_wrist),
                             native_pelvis_right_hip_examples=pair_pelvis[:5],
                             native_wrist_left_hip_examples=pair_wrist[:5])
            raw[case["id"]]=r
        rows.append(entry)
    comparisons={}
    for label,base,variant in [
        ("cuda_v32_native_only_vs_none","cuda_v32_none","cuda_v32_native_only"),
        ("cuda_v32_dynamics_only_vs_none","cuda_v32_none","cuda_v32_dynamics_only"),
        ("cuda_v32_full_vs_none","cuda_v32_none","cuda_v32_full"),
        ("cpu_v32_native_only_vs_none","cpu_v32_none","cpu_v32_native_only"),
        ("cpu_v32_dynamics_only_vs_none","cpu_v32_none","cpu_v32_dynamics_only")]:
        comparisons[label]=compare_raw(raw[base],raw[variant]) if base in raw and variant in raw else {"available":False,"reason":"one or both original G1 outputs absent"}
    old=root/"reports/v059_cuda_v32_off_s1201.json"
    if old.exists() and "cuda_v32_none" in raw:
        comparisons["cuda_none_v059_vs_v060_reproduction"]=compare_raw(json.loads(old.read_text()),raw["cuda_v32_none"])
    def n(where,flag):
        return sum(bool(x[flag]) for x in rows if x["valid"] and x["backend"]==where)
    return {"version":"v0.1.60","parent_sha":cfg["parent_sha"],
            "frozen_protocol_sha":digest,
            "physics_attempts":len(rows),"validated_original_G1":sum(x["valid"] for x in rows),
            "validated_cuda_original_G1":sum(x["valid"] and x["backend"]=="cuda:0" for x in rows),
            "validated_cpu_original_G1":sum(x["valid"] and x["backend"]=="cpu" for x in rows),
            "strict_successes":sum(x["strict_success"] is True for x in rows if x["valid"]),
            "not_a_new_policy_training_run":True,
            "four_factor_observer_comparisons":comparisons,
            "runs":rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result=audit(args.root.resolve())
    with args.output.open("x") as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
    print("AUDIT_SUMMARY",{k:v for k,v in result.items() if k not in ("runs","four_factor_observer_comparisons")})
    for r in result["runs"]:
        print(r["id"],r["status"],"strict",r["strict_success"],
              "real G1 collider points",r.get("native_pelvis_right_hip_points"),r.get("native_wrist_left_hip_points"))
    print("COMPARISONS",json.dumps(result["four_factor_observer_comparisons"],ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
