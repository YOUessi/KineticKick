#!/usr/bin/env python3
"""Write-once Isaac Sim 4.5 raw body-pair calibration; no G1 motor changes."""
from __future__ import annotations
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from v083_run_physics_batch import PYTHON
ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/"reports/v118_native_full_batch.audit.json"
PROBE=ROOT/"tools/v118_native_full_probe.py"
FROZEN=("tools/v118_native_full_probe.py",
        "tools/v118_native_full_batch.py","tests/test_v118_native_full.py",
        "docs/V118_NATIVE_FULL_FROZEN_PROTOCOL.md")
CASES=(("positive_off","off","pose","colliding"),
       ("positive_on","on","both","colliding"),
       ("separated_on","on","both","separated"))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def git(*x):return subprocess.check_output(["git",*x],cwd=ROOT,text=True).strip()
def preflight():
    if git("branch","--show-current")!="version/v0.1.118":
        raise RuntimeError("WRONG_BRANCH")
    if git("status","--porcelain","--untracked-files=no"):
        raise RuntimeError("TRACKED_SOURCE_DIRTY")
    if MASTER.exists() or MASTER.is_symlink():
        raise FileExistsError("WRITE_ONCE_RAW_BATCH")
    for file in FROZEN:
        assert (ROOT/file).read_bytes()==subprocess.check_output(
            ["git","show","HEAD:"+file],cwd=ROOT), "FROZEN_SOURCE_DRIFT:"+file
    free=shutil.disk_usage(ROOT).free/1024**3
    if free<20:raise RuntimeError("DISK_BELOW_20GIB")
    out=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
                                "--format=csv,noheader"],text=True).strip()
    if out:raise RuntimeError("GPU_FOREIGN_PROCESS:"+out)
    return round(free,2)
def run(name,mode,capture,layout):
    raw=ROOT/"reports"/("v118_native_full_"+name+".json")
    log=ROOT/"reports"/("v118_native_full_"+name+".log")
    audit=ROOT/"reports"/("v118_native_full_"+name+".audit.json")
    if any(p.exists() or p.is_symlink() for p in (raw,log,audit)):
        raise FileExistsError("EVIDENCE_ALREADY_EXISTS:"+name)
    cmd=[PYTHON,str(PROBE),"--device","cuda:0","--report-mode",mode,
        "--layout",layout,"--capture",capture,
        "--steps","180","--output",str(raw)]
    started=time.monotonic();timed=False
    with log.open("x") as f:
        f.write("SOURCE="+git("rev-parse","HEAD")+"\n")
        f.write("PROBE_SHA="+sha(PROBE)+"\n");f.flush()
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                     env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
                     OMP_NUM_THREADS="1"),start_new_session=True)
        try:rc=child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timed=True
            os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=6)
            f.write("CHILD_ONLY_TIMEOUT_120S\n")
    result={}
    if raw.exists():
        try:result=json.loads(raw.read_text())
        except Exception:pass
    good=(rc==0 and not timed and result.get("actual_runtime_frames")==180
          and "exception" not in result)
    row=dict(name=name,report_mode=mode,layout=layout,certified_isolated_physx=good,
          actual_true_simulation_steps=result.get("actual_runtime_frames",0) if good else 0,
          native_body_pair_witness=result.get("native_pair_witness",False) if good else False,
          native_contact_entries=result.get("native_counts",{}).get("native_pair_events",0) if good else 0,
          native_pair_geometry_manifold_certified=False,source_sha=git("rev-parse","HEAD"),
          raw_sha=sha(raw) if raw.exists() else None,log_sha=sha(log),exit_code=rc,
          timeout=timed,elapsed_seconds=round(time.monotonic()-started,2),
          errors=result.get("native_full_counts",{}).get("errors",[]))
    with audit.open("x") as f:json.dump(row,f,indent=2);f.write("\n")
    print("V118_NATIVE_RAW_V2_TRIAL",json.dumps(row),flush=True)
    return row
def main():
    free=preflight()
    lock=ROOT.parent/"kinetickick/experiments/kinetickick_physics_gpu.lock"
    if not lock.exists():raise FileNotFoundError("MISSING_GPU_LOCK")
    completed=[]
    with lock.open("r+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GPU_LOCK_HELD")
        for case in CASES:
            if shutil.disk_usage(ROOT).free/1024**3<20:break
            if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
                 "--format=csv,noheader"],text=True).strip():break
            v=run(*case);completed.append(v)
            if not v["certified_isolated_physx"]:break
    # Never mistake a zero native report or zero filtered force for noncollision.
    off=ROOT/"reports/v118_native_full_positive_off.json"
    on=ROOT/"reports/v118_native_full_positive_on.json"
    pose_parity=False
    if off.is_file() and on.is_file():
        a=json.loads(off.read_text()).get("runtime_z_vz")
        b=json.loads(on.read_text()).get("runtime_z_vz")
        pose_parity=bool(a and a==b)
    neg=ROOT/"reports/v118_native_full_separated_on.json"
    negative_witness=bool(neg.is_file() and json.loads(neg.read_text()).get("observed_noncollision_dynamics"))
    positive_witness=bool(on.is_file() and json.loads(on.read_text()).get("observed_collision_dynamics"))
    native_pair=bool(on.is_file() and json.loads(on.read_text()).get("native_pair_witness"))
    result=dict(version="v0.1.118",source_commit=git("rev-parse","HEAD"),
                off_on_recorded_pose_parity=pose_parity,
                actual_positive_contact_dynamics=positive_witness,
                actual_negative_separation_dynamics=negative_witness,
                full_subscription_native_pair_witness=native_pair,
                completed_calibration_runs=sum(x["certified_isolated_physx"] for x in completed),
                actual_isolated_physx_steps=sum(x["actual_true_simulation_steps"] for x in completed),
                true_G1_shots_in_this_calibration=0,disk_free_pre_gib=free,
                native_actor_collider_paths_fully_certified=bool(
                  pose_parity and positive_witness and negative_witness and native_pair),
                cases=completed,not_executed=[x[0] for x in CASES[len(completed):]])
    with MASTER.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V118_NATIVE_RAW_V2_BATCH",json.dumps({k:v for k,v in result.items() if k!="cases"}),flush=True)
if __name__=="__main__":main()
