#!/usr/bin/env python3
"""Independent read-only original-G1 v058 CPU native-contact and 200Hz audit.

Never compare CPU body trajectories to GPU as physically interchangeable.
Only events with actual two actor/collider identities and native manifold point
count toward a directly observed pair. No extrapolation beyond early stop.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GPU_DEVICE=re.compile(r"Environment device\s*:\s*(cpu|cuda:\d+)")

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def native_pair_truth(raw:dict,pair:set[str])->list[dict]:
    native=raw["physx_native_report_v056"]
    if native["errors"] or native["dropped_events"] or not native["report_capture_complete"]:
        return []
    result=[]
    for x in native["saved_interesting_events"]:
        actors=x["actor_paths"];colliders=x["collider_paths"]
        if len(actors)!=2 or len(colliders)!=2 or not all(
            a.startswith("/World/envs/env_0/Robot/") for a in actors
        ):continue
        names={a.rsplit("/",1)[-1] for a in actors}
        if names!=pair:continue
        for pt in x["point_sample"]:
            if pt.get("point_w") is None or pt.get("normal_w") is None: continue
            if not all(math.isfinite(v) for v in (*pt["point_w"],*pt["normal_w"])):continue
            result.append(dict(t_s=x["time_s"],actors=actors,colliders=colliders,
                               point_w=pt["point_w"],normal_w=pt["normal_w"],
                               impulse_xyz=pt.get("impulse_xyz"),
                               separation_m=pt.get("separation_m")))
    return result

def audit(root:Path)->dict:
    cfgpath=root/"configs/g1_backend_recovery_v058.json"
    cfg=json.loads(cfgpath.read_text())
    lock=json.loads((root/"reports/v058_protocol_lock.json").read_text())
    progress=json.loads((root/"reports/v058_batch_progress.json").read_text())
    if sha(cfgpath)!=lock["protocol_sha256"] or progress["protocol_sha256"]!=lock["protocol_sha256"]:
        raise ValueError("Frozen protocol SHA drift")
    if len(cfg["cases"])!=7 or len(progress["runs"])!=7:
        raise ValueError("Missing case")
    rows=[]
    for case,row in zip(cfg["cases"],progress["runs"]):
        if case["id"]!=row["id"]:raise ValueError("Run order drift")
        log=root/case["output"].replace(".json",".log")
        if not log.exists() or sha(log)!=row["log_sha256"]:
            raise ValueError("Raw log drift")
        device=GPU_DEVICE.findall(log.read_text(errors="replace"))
        device=device[0] if device else None
        if device!=case["backend"] or row["backend_log_observed"]!=device:
            raise ValueError("Runtime device mismatch")
        p=root/case["output"]
        if p.exists()!=bool(row.get("report_sha256")):raise ValueError("Incomplete report tracking")
        if p.exists() and sha(p)!=row["report_sha256"]:
            raise ValueError("Raw physics data mutated")
        valid=row["status"]=="PHYSICAL_VALIDATED" and row["exit_code"]==0 and p.exists()
        entry=dict(id=case["id"],requested=case["backend"],observed=device,
                   status=row["status"],validated=valid,strict_success=row.get("strict_success") if valid else None)
        if valid:
            raw=json.loads(p.read_text())
            original=raw["traces"][0]
            if len(raw["joint_names"])!=29 or any(len(f["joint_vel"])!=29 for f in original):
                raise ValueError("29DoF drift")
            velocity_limits=raw["joint_velocity_limits"]
            if len(velocity_limits)!=1 or len(velocity_limits[0])!=29:
                raise ValueError("Original 1x29 qdot limits missing")
            limit=velocity_limits[0]
            ratio=max(abs(v)/q for f in original for v,q in zip(f["joint_vel"],limit))
            contact_time=raw["records"][0].get("contact_time_s")
            end=original[-1]["time_s"]
            pre=bool(contact_time is not None and original[0]["time_s"]<=contact_time-.4+.005)
            post=bool(contact_time is not None and end>=contact_time+1.25)
            native=raw["physx_native_report_v056"]
            certified_right=native_pair_truth(raw,{"pelvis","right_hip_roll_link"})
            certified_left=native_pair_truth(raw,{"left_wrist_yaw_link","left_hip_roll_link"})
            entry.update(dict(frames=len(original),end_s=end,
                              contact_s=contact_time,pre_0p4_complete=pre,post_1p25_complete=post,
                              qdot_peak_original_limit_ratio=ratio,
                              root_max_tilt_deg=max(f["root_tilt_deg"] for f in original),
                              native_headers=native["native_headers"],
                              callback_calls=native["callback_calls"],
                              native_capture_complete=native["report_capture_complete"],
                              direct_pelvis_right_hip_manifold_points=len(certified_right),
                              pelvis_right_hip_samples=certified_right,
                              direct_wrist_hip_manifold_points=len(certified_left),
                              cpu_only_not_gpu_truth=case["backend"]=="cpu",
                              final_termination=raw["termination_reasons"],
                              instep_fraction=raw["records"][0]["requested_surface_impulse_fraction"],
                              goal=raw["records"][0]["goal"],
                              sequence_success=raw["sequence_records"][0]["sequence_success"]))
        rows.append(entry)
    return dict(version="v0.1.58",parent_sha=cfg["parent_sha"],protocol_sha=lock["protocol_sha256"],
                case_count=7,valid_original_G1=sum(x["validated"] for x in rows),
                cuda_attempts_without_raw=sum(x["observed"]=="cuda:0" and not x["validated"] for x in rows),
                cpu_valid_G1=sum(x["observed"]=="cpu" and x["validated"] for x in rows),
                strict_success_by_backend={
                    dev:sum(x["strict_success"] is True for x in rows if x["validated"] and x["observed"]==dev)
                    for dev in ("cpu","cuda:0")},
                wrist_hip_GPU_directly_certified=False,
                cpu_gpu_parity_established=False,
                rows=rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit(a.root.resolve())
    with a.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    summary={k:v for k,v in result.items() if k!="rows"}
    print(json.dumps(summary,indent=2))
    for x in result["rows"]:
        print(x["id"],x["status"],x.get("strict_success"),
              "native_pelvis_hip_points",x.get("direct_pelvis_right_hip_manifold_points"))

if __name__=="__main__":main()
