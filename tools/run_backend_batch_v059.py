#!/usr/bin/env python3
"""v059 SHA-frozen ORIGINAL G1 backend/observer parity diagnostic (no tuning).

Ten case matrix: CPU same-seed subscriber off/on, CPU CMU variants,
GPU off/on, and CPU independent perturbed initial conditions. No new policy.
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
CFG=ROOT/"configs/g1_observer_parity_v059.json"
LOCK=ROOT/"reports/v059_protocol_lock.json"
PROGRESS=ROOT/"reports/v059_batch_progress.json"
MUTEX=ROOT/"experiments/kinetickick_physics_gpu.lock"
PYTHON=Path("/home/you/conda_env/env_isaaclab/bin/python")
BASE="3eaa84e45b7eb7bc64a51f7e8a289e6602833f12"
DEVICE=re.compile(r"Environment device\s*:\s*(cpu|cuda:\d+)")

def sha(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def freeze(c:dict):
    if c["version"]!="v0.1.59" or c["parent_sha"]!=BASE or len(c["cases"])!=10:
        raise RuntimeError("Invalid preregistered protocol")
    ids=[x["id"] for x in c["cases"]]
    outs=[x["output"] for x in c["cases"]]
    if len(set(ids))!=10 or len(set(outs))!=10:
        raise RuntimeError("Duplicated case or report path")
    if any(x["backend"] not in ("cpu","cuda:0") or x["observer"] not in ("none","full") for x in c["cases"]):
        raise RuntimeError("Unknown backend/observer")
    for file,expected in c["source_hashes"].items():
        if sha(ROOT/file)!=expected:raise RuntimeError("Source drift "+file)
    for file,expected in c["policy_hashes"].items():
        if sha(ROOT/file)!=expected:raise RuntimeError("Policy drift "+file)
    return sha(CFG)

def health():
    s=os.statvfs(ROOT)
    gb=s.f_bavail*s.f_frsize/1024**3
    if gb<20:raise RuntimeError(f"Only {gb:.2f} GiB free")
    smi=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],
                       capture_output=True,text=True,timeout=15)
    if smi.returncode or smi.stdout.strip():
        raise RuntimeError("GPU compute work already present; do not proceed")

def command(c:dict,k:dict):
    cmd=[str(PYTHON),str(ROOT/"tools/run_placement_strike_v058.py"),
         "--config",str(ROOT/c["policies"][k["policy"]]),
         "--output",str(ROOT/k["output"]),
         "--eval-seed",str(k["seed"]),
         "--ball-offset","0","0","--velocity-delta","0",
         "--sim-device",k["backend"]]
    if k["observer"]=="full":
        cmd+=["--pair-dynamics-v055","--native-readonly-pair-v056","--contact-diagnostics"]
    if k["perturb_reset"]:
        cmd+=["--perturb-reset-v056"]
    return cmd

def validate(c:dict,k:dict,raw:dict,log:str,observed:str|None)->dict:
    if observed!=k["backend"] or raw.get("v057_requested_physics_backend")!=k["backend"]:
        raise ValueError("Backend mismatch")
    if "V058_NATIVE_RETURN" not in log or "V058_BACKEND_ARG_FORWARDED" not in log:
        raise ValueError("Missing complete runtime")
    if raw.get("root_fixed") is not False or raw.get("seed")!=k["seed"]:
        raise ValueError("Seed/free-root mismatch")
    if raw.get("policy_sha256")!=c["policy_hashes"][c["policies"][k["policy"]]]:
        raise ValueError("Policy drift in actual JSON")
    names=raw.get("joint_names",[])
    frames=raw.get("traces",[[]])[0]
    if len(names)!=29 or len(set(names))!=29 or len(frames)<35:
        raise ValueError("Not original G1 29DoF")
    if any(len(f.get("joint_vel",[]))!=29 for f in frames):
        raise ValueError("Incomplete qdot")
    if any(frames[i+1]["time_s"]<=frames[i]["time_s"] for i in range(len(frames)-1)):
        raise ValueError("Non-increasing timestamps")
    if k["observer"]=="full":
        if "physx_native_report_v056" not in raw or "physx_pair_dynamics_v055" not in raw:
            raise ValueError("Missing requested original pair observer")
        native=raw["physx_native_report_v056"]
        if native.get("errors") or native.get("dropped_events") or not native.get("report_capture_complete"):
            raise ValueError("Incomplete native pair capture")
    else:
        if "physx_native_report_v056" in raw or "physx_pair_dynamics_v055" in raw:
            raise ValueError("Observer OFF not actually off")
    success=raw.get("stride_shot_success")
    if not isinstance(success,bool):raise ValueError("Not strict full-task decision")
    return {"strict_success":success,"trace_rows_including_terminal":len(frames),
            "contact_s":raw["records"][0].get("contact_time_s"),
            "end_s":frames[-1]["time_s"],"termination":raw.get("termination_reasons"),
            "native_headers":raw.get("physx_native_report_v056",{}).get("native_headers"),
            "native_right_hip_headers":raw.get("physx_native_report_v056",{}).get("pelvis_right_hip_direct_path_events"),
            "native_wrist_hip_headers":raw.get("physx_native_report_v056",{}).get("wrist_hip_direct_path_events")}

def main():
    if Path(sys.executable).resolve()!=PYTHON.resolve():
        raise RuntimeError("Use pinned IsaacLab Python")
    if subprocess.run(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,
                      capture_output=True,text=True).stdout.strip():
        raise RuntimeError("Tracked source modified during experiment")
    c=json.loads(CFG.read_text())
    digest=freeze(c)
    if PROGRESS.exists():
        state=json.loads(PROGRESS.read_text())
        if state["protocol_sha256"]!=digest:raise RuntimeError("Recorded protocol drift")
    else:
        state={"protocol_sha256":digest,"runs":[]}
    MUTEX.parent.mkdir(parents=True,exist_ok=True)
    with MUTEX.open("a") as lock_handle:
        fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        health()
        if LOCK.exists():
            lock=json.loads(LOCK.read_text())
            if lock["protocol_sha256"]!=digest:raise RuntimeError("Lock drift")
        else:
            with LOCK.open("x") as f:
                json.dump({"protocol_sha256":digest,"parent_sha":BASE,
                           "preregistered_before_physics":True,"count":10},f,indent=2)
        if not PROGRESS.exists():
            with PROGRESS.open("x") as f:json.dump(state,f,indent=2)
        for i,k in enumerate(c["cases"]):
            freeze(c);health()
            if i<len(state["runs"]):
                row=state["runs"][i]
                if row["id"]!=k["id"]:raise RuntimeError("Out-of-order progress")
                p=ROOT/k["output"]
                if p.exists()!=bool(row.get("raw_sha256")):
                    raise RuntimeError("Historical output existence drift")
                if p.exists() and sha(p)!=row["raw_sha256"]:
                    raise RuntimeError("Historical physics mutated")
                log=p.with_suffix(".log")
                if sha(log)!=row["log_sha256"]:raise RuntimeError("Historical log mutated")
                print("V059_PRESERVED",k["id"],row["status"],flush=True)
                continue
            p=ROOT/k["output"]
            log=p.with_suffix(".log")
            if p.exists() or log.exists():raise FileExistsError(p)
            env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
                     MKL_NUM_THREADS="1",PYTHONPATH=":".join(["tools",
                     str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]))
            print("V059_START",i+1,len(c["cases"]),k["id"],flush=True)
            t0=time.monotonic()
            with log.open("x") as f:
                try:
                    process=subprocess.run(command(c,k),env=env,cwd=ROOT,
                                           stdout=f,stderr=subprocess.STDOUT,timeout=300)
                    code=process.returncode
                except subprocess.TimeoutExpired:
                    code=124
                    f.write("\nCHILD_TIMEOUT_300; original evidence never overwritten\n")
            content=log.read_text(errors="replace")
            devices=DEVICE.findall(content)
            observed=devices[0] if devices else None
            row={"id":k["id"],"backend_requested":k["backend"],"backend_observed":observed,
                 "policy":k["policy"],"observer":k["observer"],"seed":k["seed"],
                 "perturb_reset":k["perturb_reset"],"exit_code":code,
                 "duration_wall_s":round(time.monotonic()-t0,2),
                 "log_sha256":sha(log)}
            if p.exists():
                row["raw_sha256"]=sha(p)
                try:
                    row.update(validate(c,k,json.loads(p.read_text()),content,observed))
                    row["status"]="VALID_G1" if code==0 else "RAW_ABNORMAL_CHILD_EXIT"
                except (ValueError,KeyError,IndexError,TypeError,AttributeError) as exc:
                    row.update(status="RAW_UNSCORABLE",reason=str(exc))
            else:
                row.update(status="NO_G1_OUTPUT",reason="Exit code does not substitute for 200Hz raw G1 JSON")
            state["runs"].append(row)
            temp=PROGRESS.with_suffix(".tmp")
            temp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+"\n")
            os.replace(temp,PROGRESS)
            print("V059_END",k["id"],row["status"],row.get("strict_success"),flush=True)
        print("V059_COMPLETE",len(state["runs"]),
              sum(x["status"]=="VALID_G1" for x in state["runs"]),flush=True)

if __name__=="__main__":
    main()
