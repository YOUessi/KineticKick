#!/usr/bin/env python3
"""Unmodified historical v094 proven tensor-positive probe, v114 frozen A/B validation.

Write-once, isolated physical calibration, not original G1. No policy modified.
"""
from __future__ import annotations
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from v083_run_physics_batch import PYTHON
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v114_tensor_positive_frozen.json"
MASTER=ROOT/"reports/v114_tensor_positive_batch.audit.json"
PROBE=ROOT/"tools/v094_box_pair_probe.py"
FROZEN=("configs/v114_tensor_positive_frozen.json","docs/V114_TENSOR_POSITIVE_FROZEN_PROTOCOL.md",
        "tools/v114_tensor_positive_batch.py","tests/test_v114_tensor_positive_contract.py",
        "tools/v094_box_pair_probe.py")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()
def git(*xs):return subprocess.check_output(["git",*xs],cwd=ROOT,text=True).strip()
def preflight():
    c=json.loads(CFG.read_text())
    if git("branch","--show-current")!="version/v0.1.114":raise ValueError("WRONG_BRANCH")
    if git("status","--porcelain","--untracked-files=no"):raise RuntimeError("DIRTY_FROZEN_FILES")
    if MASTER.exists() or MASTER.is_symlink():raise FileExistsError("WRITE_ONCE_BATCH")
    if c["version"]!="v0.1.114" or c["source_parent"]!="07a256cb7b61fcc64bcab1b9bd23f4731b84b451":
        raise ValueError("FROZEN_SOURCE_PARENT_CHANGED")
    if c["calibration_cases"]!=[
         {"name":"old_off","report_mode":"off","layout":"colliding","capture":"pose"},
         {"name":"old_tensor_positive","report_mode":"on","layout":"colliding","capture":"tensor"},
         {"name":"old_tensor_separated","report_mode":"on","layout":"separated","capture":"tensor"}]:
        raise ValueError("CHANGED_MATRIX")
    if c["steps"]!=180 or c["watchdog"]!=120 or c["min_free_gib"]!=20:
        raise ValueError("UNFROZEN_SAFETY_PARAMETERS")
    for f in FROZEN:
        if (ROOT/f).read_bytes()!=subprocess.check_output(["git","show","HEAD:"+f],cwd=ROOT):
            raise RuntimeError("FROZEN_PROTOCOL_DRIFT:"+f)
    if shutil.disk_usage(ROOT).free/1024**3<20:raise RuntimeError("LOW_DISK")
    if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True).strip():
        raise RuntimeError("GPU_BUSY")
    return c
def run(c,row):
    name="v114_tensor_"+row["name"]
    out=ROOT/"reports"/(name+".json")
    log=ROOT/"reports"/(name+".log")
    aud=ROOT/"reports"/(name+".audit.json")
    if any(p.exists() or p.is_symlink() for p in (out,log,aud)):
        raise FileExistsError("RAW_WRITE_ONCE:"+name)
    args=[PYTHON,str(PROBE),"--device","cuda:0",
      "--report-mode",row["report_mode"],"--layout",row["layout"],
      "--capture",row["capture"],"--steps","180","--output",str(out)]
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
             MKL_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1")
    start=time.monotonic();timeout=False
    with log.open("x") as f:
        f.write("PROBE_SHA="+sha(PROBE)+"\nSOURCE_SHA="+git("rev-parse","HEAD")+"\n");f.flush()
        child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timeout=True
            os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=6)
            f.write("\nCHILD_ONLY_WATCHDOG\n")
    try:payload=json.loads(out.read_text())
    except Exception:payload={}
    good=rc==0 and not timeout and payload.get("actual_runtime_frames")==180 and "exception" not in payload
    result=dict(**row,certified_isolated_physx=good,actual_200hz_steps=180 if good else 0,
        observed_collision=payload.get("observed_collision_dynamics"),
        observed_separation=payload.get("observed_noncollision_dynamics"),
        positive_tensor_force=payload.get("tensor_peak_force_N"),
        tensor_contacts=payload.get("tensor_contact_counts"),
        native_pair_certified=False,
        physical_trace=payload.get("runtime_z_vz",[]) if good else [],
        output_sha=sha(out) if out.exists() else None,log_sha=sha(log),timeout=timeout,rc=rc,
        elapsed_s=round(time.monotonic()-start,2))
    with aud.open("x") as f:json.dump({k:v for k,v in result.items() if k!="physical_trace"},f,indent=2);f.write("\n")
    print("V114_TENSOR_POSITIVE_CASE",json.dumps({k:v for k,v in result.items() if k!="physical_trace"}),flush=True)
    return result
def main():
    c=preflight();lock=Path(c["gpu_lock"]); rows=[]
    if not lock.exists():raise FileNotFoundError("NO_GPU_LOCK")
    with lock.open("r+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GPU_LOCK_BUSY")
        for x in c["calibration_cases"]:
            if shutil.disk_usage(ROOT).free/1024**3<20:break
            if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
                    "--format=csv,noheader"],text=True).strip():break
            row=run(c,x);rows.append(row)
            if not row["certified_isolated_physx"]:break
    a,b=rows[0:1],rows[1:2]
    same=bool(a and b and a[0]["certified_isolated_physx"] and b[0]["certified_isolated_physx"] and a[0]["physical_trace"]==b[0]["physical_trace"])
    out=dict(version="v0.1.114",source_commit=git("rev-parse","HEAD"),
        unmodified_proven_v094_probe_sha=sha(PROBE),
        actual_certified_physx_runs=sum(x["certified_isolated_physx"] for x in rows),
        actual_200hz_steps=sum(x["actual_200hz_steps"] for x in rows),
        positive_tensor_report_count=sum(x["tensor_contacts"] or 0 for x in rows),
        off_vs_tensor_on_exact_kinematic_parity=same,
        original_g1_shots=0,native_collider_manifold_certified=False,
        cases=[{k:v for k,v in x.items() if k!="physical_trace"} for x in rows],
        not_executed=[x["name"] for x in c["calibration_cases"][len(rows):]])
    with MASTER.open("x") as f:json.dump(out,f,indent=2);f.write("\n")
    print("V114_TENSOR_POSITIVE_SUMMARY",json.dumps({k:v for k,v in out.items() if k!="cases"}),flush=True)
if __name__=="__main__":main()
