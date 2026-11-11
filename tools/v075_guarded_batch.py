#!/usr/bin/env python3
"""Immutable GPU mutex + bounded real Isaac two-case shutdown experiment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=Path("/home/you/projects/kinetickick")
PYTHON="/home/you/conda_env/env_isaaclab/bin/python"
PROTO=ROOT/"configs/v075_frozen_shutdown_native_pair.json"
FREEZE_COMMIT="501f29919214996a9cfdcd57c8e195aade3ef073"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def classify(log,code,timed):
    if timed:
        if "APP_CLOSE_ENTER" in log and "APP_CLOSE_RETURN" not in log:
            return "APP_CLOSE_TIMEOUT"
        if "SIM_STOP_ENTER" in log and "SIM_STOP_RETURN" not in log:
            return "STOP_TIMEOUT"
        if "APP_CLOSE_RETURN" in log:
            return "EXIT_AFTER_CLOSE_TIMEOUT"
        return "INIT_OR_STEPS_TIMEOUT"
    if code==0 and "PYTHON_MAIN_RETURN" in log and "APP_CLOSE_RETURN" in log and "EIGHT_REAL_SIM_STEPS_DONE_NOT_G1" in log:
        return "EIGHT_STEPS_CLEAN_NORMAL_EXIT"
    return "INCOMPLETE_OR_NONZERO"

def preflight(config):
    assert config["version"]=="v0.1.75"
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
    if branch!="version/v0.1.75": raise RuntimeError("WRONG_BRANCH")
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip():
        raise RuntimeError("TRACKED_DIRTY")
    for relative,sha in config["source_sha256"].items():
        if digest(ROOT/relative)!=sha: raise RuntimeError("SOURCE_DRIFT:"+relative)
    for relative,sha in config["original_v54_sha256"].items():
        if digest(ORIGINAL/relative)!=sha: raise RuntimeError("ORIGINAL_DATA_MUTATED:"+relative)
    fs=os.statvfs(ROOT)
    free_gib=fs.f_bavail*fs.f_frsize/(1024**3)
    if free_gib <config["minimum_disk_gib"]: raise RuntimeError("LOW_DISK")
    gpu=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name",
                        "--format=csv,noheader"],text=True,capture_output=True,check=True,timeout=12)
    if gpu.stdout.strip(): raise RuntimeError("GPU_OCCUPIED:"+gpu.stdout)
    return round(free_gib,2)

def run_one(variant,config,free_gib):
    prefix=ROOT/("reports/v075_"+variant)
    logpath=Path(str(prefix)+".log")
    resultpath=Path(str(prefix)+".json")
    if logpath.exists() or resultpath.exists():raise FileExistsError("WRITE_ONCE_EVIDENCE")
    env=dict(os.environ,PYTHONUNBUFFERED="1",CUDA_VISIBLE_DEVICES="0",
             OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    cmd=[PYTHON,str(ROOT/"tools/v075_guard_probe.py"),"--variant",variant]
    timed=False
    start=time.monotonic()
    with logpath.open("x") as out:
        out.write("WRAPPER_START "+repr(cmd)+"\n")
        out.flush()
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code=process.wait(timeout=config["smoke_timeout_s"])
        except subprocess.TimeoutExpired:
            timed=True
            try:os.kill(process.pid,signal.SIGUSR1)
            except ProcessLookupError:pass
            time.sleep(2)
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                code=process.wait(timeout=5)
            out.write("V075_CHILD_WATCHDOG_EXPIRED_NOT_G1\n")
    txt=logpath.read_text(errors="replace")
    verdict=classify(txt,code,timed)
    row=dict(version="v0.1.75",variant=variant,classification=verdict,
             clean_exit=verdict=="EIGHT_STEPS_CLEAN_NORMAL_EXIT",
             simulated_steps=8 if "EIGHT_REAL_SIM_STEPS_DONE_NOT_G1" in txt else None,
             real_G1_episodes=0, child_pid=process.pid,child_code=code,
             watchdog=timed,elapsed_wall_s=round(time.monotonic()-start,3),
             original_sources_unchanged=True,free_disk_GiB=free_gib,
             config_sha256=digest(PROTO),freeze_commit=FREEZE_COMMIT,
             source_probe_sha256=digest(ROOT/"tools/v075_guard_probe.py"),
             logfile_sha256=digest(logpath),logfile_bytes=logpath.stat().st_size)
    with resultpath.open("x") as f:json.dump(row,f,indent=2);f.write("\n")
    print("V075_GUARDED_RESULT",json.dumps(row),flush=True)
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--only",choices=("all","post_physics_guarded_close","post_physics_guarded_stop"),
                    default="all")
    a=ap.parse_args()
    c=json.loads(PROTO.read_text())
    with Path(c["gpu_lock"]).open("r+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=[]
        for variant in c["variants"] if a.only=="all" else [a.only]:
            rows.append(run_one(variant,c,preflight(c)))
        print("V075_ALL_RESULT",json.dumps(rows),flush=True)

if __name__=="__main__":main()
