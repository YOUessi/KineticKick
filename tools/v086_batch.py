#!/usr/bin/env python3
"""SHA-frozen, serial, GPU-locked isolated PhysX contact calibration matrix.

This is not original G1 football physics; do not pool its events into kick scores.
"""
from __future__ import annotations
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v086_physx_contact_calibration.json"
REPORT=ROOT/"reports/v086_isolated_calibration.audit.json"
FILES=["configs/v086_physx_contact_calibration.json","docs/V086_FROZEN_PROTOCOL.md",
       "tools/v086_isolated_physx_contact.py","tools/v086_batch.py",
       "tests/test_v086_contact_calibration.py"]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def preflight():
    if git("branch","--show-current")!="version/v0.1.86":raise RuntimeError("WRONG_BRANCH")
    if git("status","--porcelain","--untracked-files=no"):raise RuntimeError("TRACKED_DIRTY")
    if REPORT.exists():raise FileExistsError("WRITE_ONCE_BATCH")
    cfg=json.loads(CFG.read_text())
    if cfg["parent_sha"] not in git("log","--format=%H"):raise RuntimeError("PARENT_DRIFT")
    if cfg["version"]!="v0.1.86" or len(cfg["cases"])!=4:raise RuntimeError("MATRIX_CHANGED")
    if sorted(x["name"] for x in cfg["cases"])!=sorted(set(x["name"] for x in cfg["cases"])):raise RuntimeError("DUPLICATE_CASE")
    for f in FILES:
        if (ROOT/f).read_bytes()!=subprocess.check_output(["git","show","HEAD:"+f],cwd=ROOT):
            raise RuntimeError("FROZEN_SOURCE_MODIFIED:"+f)
    free=shutil.disk_usage(ROOT).free/1024**3
    if free<cfg["min_free_gib"]:raise RuntimeError("DISK_LOW")
    return cfg
def gpu_idle():
    rows=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
                                  "--format=csv,noheader"],text=True).strip()
    if rows:raise RuntimeError("GPU_OCCUPIED:"+rows)
def one(cfg,item,protocol,commit):
    base="v086_"+item["name"]
    raw=ROOT/"reports"/(base+".json")
    log=ROOT/"reports"/(base+".log")
    side=ROOT/"reports"/(base+".audit.json")
    if any(p.exists() for p in (raw,log,side)):raise FileExistsError("WRITE_ONCE:"+base)
    if shutil.disk_usage(ROOT).free/1024**3<cfg["min_free_gib"]:raise RuntimeError("DISK_LOW_DURING_BATCH")
    gpu_idle()
    cmd=[cfg["python"],str(ROOT/"tools/v086_isolated_physx_contact.py"),
         "--device",item["device"],"--mode",item["mode"],
         "--steps",str(cfg["steps"]),"--output",str(raw)]
    env=dict(os.environ,PYTHONUNBUFFERED="1",CUDA_VISIBLE_DEVICES="0",
             PYTHONPATH=str(ROOT/"tools"),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    start=time.monotonic()
    with log.open("x") as f:
        f.write("FROZEN_COMMIT="+commit+"\nPROTOCOL_SHA256="+protocol+"\nCMD="+repr(cmd)+"\n")
        f.flush()
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,
                              start_new_session=True)
        timeout=False
        try:rc=proc.wait(timeout=cfg["timeout_s"])
        except subprocess.TimeoutExpired:
            timeout=True
            os.killpg(proc.pid,signal.SIGTERM)
            try:rc=proc.wait(timeout=7)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                rc=proc.wait(timeout=7)
            f.write("\nV086_WATCHDOG_CHILD_TERMINATION\n")
    output={}
    if raw.exists():
        try:output=json.loads(raw.read_text())
        except Exception as e:output={"decode_error":repr(e)}
    txt=log.read_text(errors="replace")
    certified=(rc==0 and not timeout and "V086_SIM_RESET" in txt and
                "V086_PHYSICS_RESULT" in txt and "V086_APP_CLOSED" in txt and
                output.get("steps")==cfg["steps"] and
                output.get("device")==item["device"] and
                output.get("mode")==item["mode"] and
                output.get("original_G1_trial") is False and
                output.get("original_USD_modified") is False and
                output.get("usd_inspection",{}).get("ball_api") is True and
                output.get("counts",{}).get("poll_calls")==cfg["steps"])
    status="CERTIFIED_ISOLATED_TEST" if certified else "UNSCORED_ATTEMPT"
    row={"case":item["name"],"device":item["device"],"mode":item["mode"],
         "certified_isolated_physics":certified,"status":status,
         "positive_native_contact":bool(certified and output.get("calibrated_native_pair")),
         "original_G1_trials":0,"rc":rc,"watchdog":timeout,
         "elapsed_s":round(time.monotonic()-start,2),
         "raw_sha256":sha(raw) if raw.exists() else None,"log_sha256":sha(log),
         "counts":output.get("counts",{}),"errors":output.get("errors",[]),
         "last_lines":txt.splitlines()[-10:]}
    with side.open("x") as f:json.dump(row,f,indent=2);f.write("\n")
    print("V086_ISOLATED",item["name"],status,"POS",row["positive_native_contact"],"RC",rc,
          "HEADERS",row["counts"].get("callback_headers"),row["counts"].get("poll_headers"),flush=True)
    return row
def main():
    cfg=preflight()
    commit=git("rev-parse","HEAD");protocol=sha(CFG)
    rows=[]
    lock=Path(cfg["gpu_lock"])
    if not lock.is_file():raise RuntimeError("GLOBAL_LOCK_MISSING")
    with lock.open("r+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GPU_LOCK_BUSY")
        for case in cfg["cases"]:
            row=one(cfg,case,protocol,commit);rows.append(row)
            # Do not continue after an uncertified/timeout experiment: machine state uncertain.
            if not row["certified_isolated_physics"]:break
    audit={"version":cfg["version"],"parent_sha":cfg["parent_sha"],
           "frozen_commit":commit,"protocol_sha256":protocol,
           "isolated_physx_executions":sum(x["certified_isolated_physics"] for x in rows),
           "uncertified_attempts":sum(not x["certified_isolated_physics"] for x in rows),
           "native_positive_controls":sum(x["positive_native_contact"] for x in rows),
           "real_original_29DoF_G1_shot_trials":0,
           "unexecuted": [x["name"] for x in cfg["cases"][len(rows):]],
           "original_G1_wrist_hip_manifold_certified":False,"rows":rows}
    with REPORT.open("x") as f:json.dump(audit,f,indent=2);f.write("\n")
    print("V086_BATCH_FINISHED",json.dumps({k:v for k,v in audit.items() if k!="rows"}),flush=True)
if __name__=="__main__":main()
