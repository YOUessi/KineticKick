#!/usr/bin/env python3
"""SHA-frozen CPU/PhysX infrastructure checks, then unchanged original G1 OFF/ON.

No GPU work on foreign compute usage, insufficient disk or missing lock. Every child
gets new write-once outputs. Failed pre-step launches have no shot denominator.
"""
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v102_frozen_diagnostics.json"
OUTPUT=ROOT/"reports/v102_batch.audit.json"
FROZEN=[
 "configs/v102_frozen_diagnostics.json","docs/V102_FROZEN_PROTOCOL.md",
 "tools/v101_trace_launcher.py","tools/v101_cpu_onnx_control.py",
 "tools/v102_batch.py","tests/test_v102_diagnostics.py",
 "tools/v080_run_poll_strike.py","tools/v098_run_pair_strike.py",
 "tools/v094_box_pair_probe.py"
]
PY="/home/you/conda_env/env_isaaclab/bin/python"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()

def preflight():
    c=json.loads(CFG.read_text())
    if git("branch","--show-current")!="version/v0.1.102":raise RuntimeError("WRONG_BRANCH")
    if git("status","--porcelain","--untracked-files=no"):raise RuntimeError("TRACKED_DIRTY")
    if c["parent_sha"] not in git("log","--format=%H"):raise RuntimeError("PARENT_SHA_MISSING")
    if OUTPUT.exists():raise FileExistsError("WRITE_ONCE_MASTER")
    if len(c["g1_cases"])!=6 or len({t["name"] for t in c["g1_cases"]})!=6:
        raise ValueError("CASE_COUNT")
    if [(x["control"],x["observer"]) for x in c["g1_cases"]] != [
        ("v32",False),("v32",True),("cmu02",False),("cmu02",True),
        ("cmu05",False),("cmu05",True)]:
        raise ValueError("CASE_ORDER")
    if c["frozen_legacy_matrix_sha256"]!=sha(ROOT/"configs/v100_frozen_matrix.json"):
        raise ValueError("HISTORICAL_PROTOCOL_DRIFT")
    for path in FROZEN:
        expected=subprocess.check_output(["git","show","HEAD:"+path],cwd=ROOT)
        if (ROOT/path).read_bytes()!=expected:raise ValueError("SOURCE_SHA_DRIFT:"+path)
    for path in c["controls"].values():
        cfg=json.loads((ROOT/path).read_text())
        for name,h in cfg["frozen_files_sha256"].items():
            if sha(ROOT/name)!=h:raise ValueError("UPSTREAM_DRIFT:"+name)
        for name in ("template","reference","expert_plan"):
            if sha(ROOT/cfg[name])!=cfg[name+"_sha256"]:raise ValueError("ASSET_DRIFT:"+name)
    for required in (ROOT/".pretrained_checkpoints/paid_official/policy_30000.onnx",
                     ROOT/"experiments/runtime/onnxruntime/onnxruntime",
                     ROOT/"configs/paid_original/paid_main_resolved.urdf"):
        if not required.exists():raise FileNotFoundError(required)
    if c["max_child_s"]<20 or c["max_child_s"]>180:raise ValueError("WATCHDOG_LIMIT")
    guard(c)
    return c

def guard(c):
    free=shutil.disk_usage(ROOT).free/1024**3
    if free<20 or free<c["min_free_gib"]:raise RuntimeError("DISK_BELOW_20GIB")
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
                                  "--format=csv,noheader"],text=True).strip()
    if apps:raise RuntimeError("GPU_BUSY:"+apps)
    return round(free,3)

def run_child(c,name,args,timeout,kind,original=False):
    out=ROOT/"reports"/("v102_"+name+".json")
    log=ROOT/"reports"/("v102_"+name+".log")
    audit=ROOT/"reports"/("v102_"+name+".audit.json")
    if any(x.exists() or x.is_symlink() for x in (out,log,audit)):
        raise FileExistsError("WRITE_ONCE:"+name)
    free=guard(c)
    env=dict(os.environ,PYTHONUNBUFFERED="1",CUDA_VISIBLE_DEVICES="0",
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
             PYTHONPATH=":".join([str(ROOT/"tools"),
                 str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]))
    argv=[PY,*args]
    started=time.monotonic()
    with log.open("x") as f:
        f.write("FROZEN_HEAD="+git("rev-parse","HEAD")+"\nPROTOCOL_SHA="+sha(CFG)
                +"\nCOMMAND="+repr(argv)+"\n");f.flush()
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,
                               start_new_session=True)
        timeout_hit=False
        try:rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timeout_hit=True
            os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                rc=child.wait(timeout=6)
            f.write("\nV101_OWN_CHILD_ONLY_TIMEOUT\n");f.flush()
    txt=log.read_text(errors="replace")
    raw={}
    if out.exists():
        try:raw=json.loads(out.read_text())
        except Exception as exc:raw={"read_error":str(exc)}
    certified=False;reason="NOT_PHYSICS"
    if kind=="cpu":
        certified=rc==0 and "V101_CPU_ONNX_READY" in txt and not timeout_hit
        reason="CPU_ONNX_ONLY"
    elif kind=="box":
        certified=(rc==0 and not timeout_hit and raw.get("actual_runtime_frames")==180
                  and raw.get("observed_collision_dynamics") and "V094_RESULT" in txt
                  and raw.get("tensor_pair_witness"))
        reason="ISOLATED_PHYSX_ONLY"
    elif kind=="g1":
        from v083_run_physics_batch import certified_physics
        trial=next(t for t in c["g1_cases"] if t["name"]==name)
        certified,reason=certified_physics(raw,txt,trial,rc,timeout_hit)
    gates={"hard_gate_pass":False,"failed_gates":["UNSCORED"]}
    if kind=="g1" and certified:
        from audit_v067_hard_gates import physically_scored
        try:gates=physically_scored(raw,wrapper_certified=True)
        except Exception as exc:gates={"hard_gate_pass":False,"failed_gates":["AUDIT_ERROR"],"error":str(exc)}
    r=dict(name=name,kind=kind,original_G1=original,elapsed_s=round(time.monotonic()-started,3),
           rc=rc,timeout=timeout_hit,certified_physical_evidence=bool(certified),
           reason=reason,raw_sha256=sha(out) if out.exists() else None,
           log_sha256=sha(log),raw_200hz_steps=len(raw.get("traces",[[]])[0]) if kind=="g1" else 0,
           isolated_physx_steps=raw.get("actual_runtime_frames",0) if kind=="box" else 0,
           stack_dumps=txt.count("Timeout ("),last_log_lines=txt.splitlines()[-6:],
           disk_free_pre_gib=free,hard_gate=gates)
    with audit.open("x") as f:json.dump(r,f,indent=2);f.write("\n")
    print("V102_ATTEMPT",name,"certified",r["certified_physical_evidence"],
          "frames",r["raw_200hz_steps"]+r["isolated_physx_steps"],
          "traces",r["stack_dumps"],"rc",rc,flush=True)
    return r,raw

def parity(a,b):
    from v100_gpu_batch import _max_abs
    x,y=a.get("traces"),b.get("traces")
    if not x or not y:return {"certified":False,"reason":"MISSING_G1_TRACES"}
    d0=_max_abs(x,y)
    d1=_max_abs(a.get("records"),b.get("records"))
    return {"certified":bool(d0==0 and d1==0),"max_trace_abs_delta":d0,
            "max_record_abs_delta":d1,"true_physics_rows":len(x[0])}

def main():
    c=preflight()
    rows=[];parities=[];prior={}
    lock=Path(c["gpu_lock"])
    if not lock.exists():raise RuntimeError("GPU_GLOBAL_LOCK_MISSING")
    with lock.open("r+") as lf:
        try:fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GPU_GLOBAL_LOCK_BUSY")
        cpu,raw=run_child(c,"cpu_onnx",
          ["tools/v101_cpu_onnx_control.py"],30,"cpu")
        rows.append(cpu)
        if cpu["certified_physical_evidence"]:
            box,raw=run_child(c,"box_tensor_positive",
              ["tools/v101_trace_launcher.py","--interval-s","15",
               "tools/v094_box_pair_probe.py","--",
               "--device","cuda:0","--report-mode","on","--layout","colliding",
               "--capture","tensor","--steps","180",
               "--output",str(ROOT/"reports/v102_box_tensor_positive.json")],
              c["max_child_s"],"box")
            rows.append(box)
        if len(rows)>1 and rows[-1]["certified_physical_evidence"]:
            for t in c["g1_cases"]:
                name=t["name"]
                args=["tools/v101_trace_launcher.py","--interval-s","15",
                      "tools/v098_run_pair_strike.py" if t["observer"] else "tools/v080_run_poll_strike.py",
                      "--","--config",str(ROOT/c["controls"][t["control"]]),
                      "--output",str(ROOT/"reports"/("v102_"+name+".json")),
                      "--ball-offset",*[str(v) for v in t["ball_offset_xy"]],
                      "--velocity-delta",str(t["velocity_delta"]),
                      "--eval-seed",str(t["seed"]),"--contact-diagnostics"]
                if t["observer"]:args.append("--tensor-pair-v098")
                r,raw=run_child(c,name,args,c["max_child_s"],"g1",original=True)
                rows.append(r)
                if not r["certified_physical_evidence"]:break
                if t["observer"]:
                    p=parity(prior[t["control"]],raw)
                    parities.append({"control":t["control"],**p})
                    if not p["certified"]:break
                else:prior[t["control"]]=raw
    result=dict(version="v0.1.102",frozen_commit=git("rev-parse","HEAD"),
                frozen_config_sha256=sha(CFG),attempts=len(rows),cases=rows,
                isolated_physx_certified=sum(x["kind"]=="box" and x["certified_physical_evidence"] for x in rows),
                certified_new_G1_runs=sum(x["kind"]=="g1" and x["certified_physical_evidence"] for x in rows),
                strict_G1_pass=sum(x["kind"]=="g1" and x["certified_physical_evidence"]
                    and x["hard_gate"]["hard_gate_pass"] for x in rows),
                matching_G1_off_on_pairs=parities,
                not_run=[x["name"] for x in c["g1_cases"] if not any(
                    z["name"]==x["name"] for z in rows)],
                native_actor_collider_manifold_certified=False)
    with OUTPUT.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V102_BATCH_FINAL",json.dumps({k:v for k,v in result.items() if k!="cases"}),flush=True)

if __name__=="__main__":main()
