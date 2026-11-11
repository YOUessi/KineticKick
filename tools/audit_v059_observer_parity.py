#!/usr/bin/env python3
"""Independent v059 G1 case, PhysX-native G1 pair, observer-parity audit.

Only true original CPU G1 contact-point pairs count as CPU evidence.
Never infer GPU parity or post-termination recovery, never score missing JSON.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from summarize_v058_censored_g1 import pair_points

ROOT=Path(__file__).resolve().parents[1]
BACKEND=re.compile(r"Environment device\s*:\s*(cpu|cuda:\d+)")
DYNAMIC_FIELDS=(
    "time_s","root_xyz","root_speed_xy","root_tilt_deg","ball_xyz","ball_velocity",
    "contact_force_N","contact_point_foot","contact_normal_foot","feet_support_N",
    "nonfoot_contact_N","terminated","contacts","joint_pos","joint_vel",
    "joint_torque_estimate","joint_target","joint_velocity_target",
    "whole_body_com_xyz","whole_body_com_velocity","support_foot_force_N",
    "robot_ball_contact_force_N","right_foot_point_velocity","ball_angular_velocity",
    "coordination_phase"
)

def sha(p:Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def compare_payload(a,b,prefix="",tol=1e-6,limit=12):
    """Return absolute difference maximum and truncated mismatch examples; no blind rounding."""
    bad=[]
    max_abs=0.
    counts=0
    def walk(x,y,path):
        nonlocal max_abs,counts
        if isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)) and not isinstance(y,bool):
            if math.isfinite(x) and math.isfinite(y):
                v=abs(x-y)
                max_abs=max(max_abs,v)
                if v<=tol:return
            elif x==y:return
        elif isinstance(x,dict) and isinstance(y,dict):
            if x.keys()!=y.keys():
                counts+=1
                if len(bad)<limit:bad.append(path+".keys")
                return
            for k in x:walk(x[k],y[k],path+"."+str(k))
            return
        elif isinstance(x,list) and isinstance(y,list):
            if len(x)!=len(y):
                counts+=1
                if len(bad)<limit:bad.append(path+".length")
                return
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+"["+str(i)+"]")
            return
        elif x==y:return
        counts+=1
        if len(bad)<limit:bad.append(path)
    walk(a,b,prefix)
    return {"matches_at_abs_tol_1e_6":counts==0,"mismatch_count":counts,
            "sample_mismatch_paths":bad,"max_abs_numeric_difference":max_abs}

def compare_raw(a:dict,b:dict):
    ta,tb=a["traces"][0],b["traces"][0]
    if len(ta)!=len(tb):
        return {"matches_at_abs_tol_1e_6":False,"rows_left":len(ta),
                "rows_right":len(tb),"mismatch_count":1,"sample_mismatch_paths":["frames.length"]}
    xa=[{k:v for k,v in f.items() if k in DYNAMIC_FIELDS} for f in ta]
    xb=[{k:v for k,v in f.items() if k in DYNAMIC_FIELDS} for f in tb]
    compare=compare_payload(xa,xb,"raw_200hz")
    compare.update(rows_left=len(ta),rows_right=len(tb),
                   common_field_count=len(DYNAMIC_FIELDS))
    return compare

def audit(root:Path):
    cfgpath=root/"configs/g1_observer_parity_v059.json"
    cfg=json.loads(cfgpath.read_text())
    lock=json.loads((root/"reports/v059_protocol_lock.json").read_text())
    progress=json.loads((root/"reports/v059_batch_progress.json").read_text())
    if cfg["parent_sha"]!="3eaa84e45b7eb7bc64a51f7e8a289e6602833f12":
        raise ValueError("Unexpected parent")
    if lock["protocol_sha256"]!=sha(cfgpath) or progress["protocol_sha256"]!=sha(cfgpath):
        raise ValueError("Frozen protocol drift")
    if len(cfg["cases"])!=10 or len(progress["runs"])!=10:
        raise ValueError("Incomplete case matrix")
    cases=[]
    raws={}
    for c,p in zip(cfg["cases"],progress["runs"]):
        if c["id"]!=p["id"] or p["backend_requested"]!=c["backend"] or p["observer"]!=c["observer"]:
            raise ValueError("Case/progress drift")
        log=root/c["output"].replace(".json",".log")
        if sha(log)!=p["log_sha256"]:raise ValueError("Historical log drift")
        runtime=BACKEND.findall(log.read_text(errors="replace"))
        detected=runtime[0] if runtime else None
        if detected!=c["backend"] or p["backend_observed"]!=detected:
            raise ValueError("Runtime backend not as frozen")
        file=root/c["output"]
        if file.exists()!=bool(p.get("raw_sha256")):raise ValueError("Raw presence drift")
        if file.exists() and sha(file)!=p["raw_sha256"]:raise ValueError("Raw SHA drift")
        valid=p["status"]=="VALID_G1" and p["exit_code"]==0 and file.exists()
        x={"id":c["id"],"backend":detected,"policy":c["policy"],"observer":c["observer"],
           "seed":c["seed"],"perturb_reset":c["perturb_reset"],
           "status":p["status"],"valid":valid,
           "strict_success":p.get("strict_success") if valid else None}
        if valid:
            raw=json.loads(file.read_text())
            if raw["seed"]!=c["seed"] or raw["root_fixed"] is not False or len(raw["joint_names"])!=29:
                raise ValueError("Original G1 identity invalid")
            limits=raw["joint_velocity_limits"][0]
            if len(limits)!=29 or min(limits)<=0:raise ValueError("Original qdot limits bad")
            frames=raw["traces"][0]
            if any(len(z["joint_vel"])!=29 for z in frames):raise ValueError("Original 29DoF missing")
            cutoff=bool(frames[-1].get("terminated") and 0<frames[-1]["time_s"]-frames[-2]["time_s"]<.003)
            phys=frames[:-1] if cutoff else frames
            qratio=max(abs(v)/lim for f in phys for v,lim in zip(f["joint_vel"],limits))
            ct=raw["records"][0].get("contact_time_s")
            x.update(raw_sha256=p["raw_sha256"],frames_200hz=len(phys),
                     terminal_status_marker=cutoff,peak_qdot_original_limit_ratio=qratio,
                     contact_time_s=ct,physical_end_s=phys[-1]["time_s"],
                     pre_0p4_complete=ct is not None and phys[0]["time_s"]<=ct-.4+.0051,
                     post_1p25_complete=ct is not None and phys[-1]["time_s"]>=ct+1.25,
                     goal=raw["records"][0]["goal"],
                     instep_fraction=raw["records"][0]["requested_surface_impulse_fraction"],
                     termination=raw["termination_reasons"])
            if c["observer"]=="full":
                n=raw["physx_native_report_v056"]
                x.update(native_complete=bool(n["report_capture_complete"] and not n["errors"] and not n["dropped_events"]),
                         direct_cpu_pelvis_right_hip_manifold_points=len(pair_points(n)),
                         direct_cpu_wrist_left_hip_manifold_points=len(pair_points(n,{"left_wrist_yaw_link","left_hip_roll_link"})))
            raws[c["id"]]=raw
        cases.append(x)
    comparison={}
    for name,off,full in [
        ("cpu_frozen_v32_same_seed_off_vs_full","cpu_v32_off_s1201","cpu_v32_full_s1201"),
        ("cpu_cmu_10_05_same_seed_off_vs_full","cpu_cmu05_off_s1201","cpu_cmu05_full_s1201")]:
        comparison[name]=compare_raw(raws[off],raws[full]) if off in raws and full in raws else {"available":False}
    reference=root/"reports/v058_v058_cpu_frozen_v32.json"
    if reference.exists() and sha(reference)=="4c834f62c23f1fb94ae0bd1d043081f877d0570dad7c4afecd5e54446aa30644" and "cpu_v32_full_s1201" in raws:
        comparison["v058_cpu_full_vs_v059_cpu_full_reproducibility"]=compare_raw(json.loads(reference.read_text()),raws["cpu_v32_full_s1201"])
    for name,a,b in [
        ("cpu_v32_unperturbed_vs_perturbed","cpu_v32_full_s1201","cpu_v32_full_s2333_perturb"),
        ("cpu_cmu05_unperturbed_vs_perturbed","cpu_cmu05_full_s1201","cpu_cmu05_full_s2333_perturb")]:
        if a in raws and b in raws:
            x,y=raws[a]["traces"][0][0],raws[b]["traces"][0][0]
            comparison[name]={"first_root_position_equal":x["root_xyz"]==y["root_xyz"],
                              "first_joint_position_equal":x["joint_pos"]==y["joint_pos"],
                              "distinct_actual_initial_observations":x["root_xyz"]!=y["root_xyz"] or x["joint_pos"]!=y["joint_pos"]}
        else:comparison[name]={"available":False}
    return {"version":"v0.1.59","parent_sha":cfg["parent_sha"],"frozen_protocol_sha":sha(cfgpath),
            "case_attempts":len(cases),"valid_original_G1":sum(c["valid"] for c in cases),
            "valid_cpu_original_G1":sum(c["valid"] and c["backend"]=="cpu" for c in cases),
            "valid_cuda_original_G1":sum(c["valid"] and c["backend"]=="cuda:0" for c in cases),
            "strict_successes":sum(c["strict_success"] is True for c in cases if c["valid"]),
            "cpu_direct_pair_observability_only_not_gpu_truth":True,
            "read_only_instrumentation_parity_evidence":comparison,
            "cases":cases}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit(a.root.resolve())
    with a.output.open("x") as f:
        json.dump(result,f,indent=2,ensure_ascii=False);f.write("\n")
    for case in result["cases"]:
        print(case["id"],case["status"],case.get("strict_success"),
              "pelvis_points",case.get("direct_cpu_pelvis_right_hip_manifold_points"))
    print("SUMMARY",{k:v for k,v in result.items() if k not in ("cases","read_only_instrumentation_parity_evidence")})
    print("PARITY",json.dumps(result["read_only_instrumentation_parity_evidence"],indent=2))

if __name__=="__main__":main()
