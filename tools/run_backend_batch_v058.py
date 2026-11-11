#!/usr/bin/env python3
"""v058 reproducible, SHA-locked seven-case CPU/GPU G1 diagnostic matrix.

GPU-exclusive; write-once; invalid/missing physical evidence never scored.
v056 original G1 control remains immutable; v058 fixes only v057 inner argv.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/g1_backend_recovery_v058.json"
LOCK=ROOT/"reports/v058_protocol_lock.json"
PROGRESS=ROOT/"reports/v058_batch_progress.json"
MUTEX=ROOT/"experiments/kinetickick_physics_gpu.lock"
PYTHON=Path("/home/you/conda_env/env_isaaclab/bin/python")
BASE="f7c3c2f696aea7e1a6cf955b6dc3632982958f50"
DEVICE_RE=re.compile(r"Environment device\s*:\s*(cpu|cuda:\d+)")

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def preflight(c:dict)->str:
    if c.get("version")!="v0.1.58" or c.get("parent_sha")!=BASE:
        raise RuntimeError("Unexpected protocol version or parent")
    if len(c["cases"])!=7 or len({x["output"] for x in c["cases"]})!=7:
        raise RuntimeError("Seven unique frozen cases mandatory")
    for src,hash_before in c["source_hashes"].items():
        if sha(ROOT/src)!=hash_before:
            raise RuntimeError("Frozen source changed: "+src)
    return sha(CFG)

def health():
    stat=os.statvfs(ROOT)
    free_gib=stat.f_bavail*stat.f_frsize/(1024**3)
    if free_gib<20:raise RuntimeError(f"Only {free_gib:.2f} GiB free")
    smi=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],
                       capture_output=True,text=True,timeout=15)
    if smi.returncode!=0 or smi.stdout.strip():
        raise RuntimeError("Another CUDA process active; do not share GPU")

def argv(c:dict,case:dict)->list[str]:
    name="tools/run_placement_strike_v056.py" if case["runner"]=="v056" else "tools/run_placement_strike_v058.py"
    a=[str(PYTHON),str(ROOT/name),"--config",str(ROOT/c["policies"][case["policy"]]),
       "--output",str(ROOT/case["output"]),"--ball-offset",*map(str,case["ball_offset_xy_m"]),
       "--velocity-delta",str(case["velocity_y_delta_mps"]),"--eval-seed",str(case["seed"]),
       "--contact-diagnostics","--pair-dynamics-v055","--native-readonly-pair-v056"]
    if case["runner"]=="v058":a+=["--sim-device",case["backend"]]
    return a

def validate_output(case:dict,raw:dict,device:str|None,log:str)->dict:
    if device!=case["backend"]:
        raise ValueError(f"Observed backend {device} != expected {case['backend']}")
    if raw.get("seed")!=case["seed"] or raw.get("root_fixed") is not False:
        raise ValueError("Not original free-root seeded G1")
    if len(raw.get("joint_names",[]))!=29:
        raise ValueError("Missing original 29DoF order")
    if raw.get("policy_sha256")!=sha(ROOT/config["policies"][case["policy"]]):
        raise ValueError("Policy hash mismatch")
    for name in ["traces","records","termination_reasons","stride_shot_success",
                 "physx_native_report_v056","physx_pair_dynamics_v055"]:
        if name not in raw: raise ValueError("Missing "+name)
    frames=raw["traces"][0]
    if len(frames)<35 or any(len(x.get("joint_vel",[]))!=29 for x in frames):
        raise ValueError("Absent 200Hz full 29DoF G1 trajectory")
    if case["runner"]=="v058":
        if raw.get("v057_requested_physics_backend")!=case["backend"]:
            raise ValueError("Requested backend is inconsistent")
        if "V058_BACKEND_ARG_FORWARDED" not in log or "V058_NATIVE_RETURN" not in log:
            raise ValueError("No runtime forwarding and return evidence")
    success=raw["stride_shot_success"]
    if isinstance(success,dict): success=success.get("success")
    if not isinstance(success,bool): raise ValueError("Ambiguous strict success flag")
    native=raw["physx_native_report_v056"]
    dynamics=raw["physx_pair_dynamics_v055"]
    if len(dynamics.get("frames",[]))<30:raise ValueError("No 200Hz whole-body observer")
    return dict(strict_success=success,frames=len(frames),
                dynamics_frames=len(dynamics["frames"]),
                native_headers=native.get("native_headers"),
                direct_wrist_hip_events=native.get("wrist_hip_direct_path_events"),
                termination_reasons=raw["termination_reasons"])

if __name__=="__main__":
    if Path(sys.executable).resolve()!=PYTHON.resolve():
        raise RuntimeError("Use pinned IsaacLab Python")
    config=json.loads(CFG.read_text())
    digest=preflight(config)
    if PROGRESS.exists():
        state=json.loads(PROGRESS.read_text())
        if state["protocol_sha256"]!=digest:raise RuntimeError("Progress drift")
    else: state=dict(protocol_sha256=digest,runs=[])
    MUTEX.parent.mkdir(exist_ok=True,parents=True)
    with MUTEX.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        health()
        if LOCK.exists():
            lock=json.loads(LOCK.read_text())
            if lock["protocol_sha256"]!=digest:raise RuntimeError("Frozen lock drift")
        else:
            with LOCK.open("x") as f:
                json.dump(dict(parent_sha=BASE,protocol_sha256=digest,
                               created_before_physics=True,case_count=7),f,indent=2)
        if not PROGRESS.exists():
            with PROGRESS.open("x") as f:json.dump(state,f,indent=2)
        for i,case in enumerate(config["cases"]):
            preflight(config)
            health()
            if i<len(state["runs"]):
                row=state["runs"][i]
                if row["id"]!=case["id"]:raise RuntimeError("Progress out of order")
                if row.get("report_sha256") and sha(ROOT/case["output"])!=row["report_sha256"]:
                    raise RuntimeError("Raw historical evidence mutated")
                print("V058_PRESERVED",row["id"],row["status"],flush=True)
                continue
            dest=ROOT/case["output"]; log=dest.with_suffix(".log")
            if dest.exists() or log.exists():raise FileExistsError("Write-once collision "+str(dest))
            env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
                     MKL_NUM_THREADS="1",PYTHONPATH=":".join(["tools",
                     str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]))
            start=time.monotonic()
            print("V058_START",i+1,len(config["cases"]),case["id"],flush=True)
            with log.open("x") as f:
                try:
                    r=subprocess.run(argv(config,case),cwd=ROOT,env=env,
                                     stdout=f,stderr=subprocess.STDOUT,timeout=360)
                    code=r.returncode
                except subprocess.TimeoutExpired:
                    code=124
                    f.write("\nCHILD_TIMED_OUT_360S; raw reports retained.\n")
            logfile=log.read_text(errors="replace")
            devices=DEVICE_RE.findall(logfile)
            observed=devices[0] if devices else None
            row=dict(id=case["id"],runner=case["runner"],
                     backend_requested=case["backend"],backend_log_observed=observed,
                     wall_s=round(time.monotonic()-start,2),exit_code=code,
                     log_sha256=sha(log))
            if dest.exists():
                row["report_sha256"]=sha(dest)
                try:
                    raw=json.loads(dest.read_text())
                    row.update(validate_output(case,raw,observed,logfile))
                    row["status"]="PHYSICAL_VALIDATED" if code==0 else "RAW_BUT_ABNORMAL_EXIT"
                except (ValueError,KeyError,TypeError,IndexError,AttributeError) as error:
                    row.update(status="RAW_UNSCORABLE",reason=str(error))
            else:
                row.update(status="NO_COMPLETE_PHYSICS",reason="No complete original G1 JSON")
            state["runs"].append(row)
            tmp=PROGRESS.with_suffix(".tmp")
            tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n")
            os.replace(tmp,PROGRESS)
            print("V058_END",case["id"],row["status"],row.get("strict_success"),flush=True)
        print("V058_BATCH_COMPLETE",len(state["runs"]),
              sum(r["status"]=="PHYSICAL_VALIDATED" for r in state["runs"]),flush=True)
