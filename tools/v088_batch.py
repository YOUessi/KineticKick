#!/usr/bin/env python3
"""Frozen v88 isolated PhysX positive-control; no original G1 trials."""
from __future__ import annotations
import fcntl, hashlib, json, os, shutil, signal, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v088_frozen_matrix.json"
FILES=("configs/v088_frozen_matrix.json","tools/v088_batch.py",
       "tools/v088_physx_positive.py","tests/test_v088_evidence.py",
       "docs/V088_FROZEN_PROTOCOL.md")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def check_config():
    c=json.loads(CFG.read_text())
    assert c["version"]=="v0.1.88" and git("branch","--show-current")=="version/v0.1.88"
    assert c["parent_sha"] in git("log","--format=%H")
    assert not git("status","--porcelain","--untracked-files=no")
    assert c["steps"]==180 and len(c["matrix"])==4
    assert len({r["case"] for r in c["matrix"]})==4
    assert c["original_G1_gpu_shots_scheduled"]==0
    for f in FILES:
        if (ROOT/f).read_bytes()!=subprocess.check_output(["git","show","HEAD:"+f],cwd=ROOT):
            raise RuntimeError("FROZEN_SOURCE_DRIFT:"+f)
    if (ROOT/"reports/v088_batch.audit.json").exists():raise FileExistsError("WRITE_ONCE_BATCH")
    return c
def gpu_idle():
    txt=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
        "--format=csv,noheader"],text=True).strip()
    if txt:raise RuntimeError("GPU_OCCUPIED:"+txt)
def trial(cfg,row,protocol,commit):
    case=row["case"]
    base=ROOT/"reports"/("v088_"+case)
    raw=base.with_suffix(".json"); log=base.with_suffix(".log"); audit=ROOT/"reports"/("v088_"+case+".audit.json")
    if any(p.exists() for p in (raw,log,audit)):raise FileExistsError("WRITE_ONCE:"+case)
    if shutil.disk_usage(ROOT).free/1024**3<cfg["min_free_gib"]:raise RuntimeError("DISK_GUARD")
    gpu_idle()
    cmd=[cfg["python"],str(ROOT/"tools/v088_physx_positive.py"),"--device",row["device"],
         "--mode",row["mode"],"--steps",str(cfg["steps"]),"--output",str(raw)]
    t=time.monotonic()
    env=dict(os.environ,PYTHONPATH=str(ROOT/"tools"),PYTHONUNBUFFERED="1",
             CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    with log.open("x") as o:
        o.write("FROZEN_COMMIT="+commit+"\nPROTOCOL_SHA256="+protocol+"\nCMD="+repr(cmd)+"\n");o.flush()
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=o,stderr=subprocess.STDOUT,
                           start_new_session=True)
        watchdog=False
        try:code=p.wait(timeout=cfg["timeout_s"])
        except subprocess.TimeoutExpired:
            watchdog=True
            os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=8)
            o.write("\nV088_OWN_CHILD_WATCHDOG\n")
    data={}
    if raw.exists():
        try:data=json.loads(raw.read_text())
        except Exception as e:data={"read_error":repr(e)}
    txt=log.read_text(errors="replace")
    lifecycle=(code==0 and not watchdog and "V088_SIM_RESET" in txt and
               "V088_PHYSICS_RESULT" in txt and "V088_APP_CLOSED" in txt)
    physical=(lifecycle and data.get("physically_certified") is True and
              data.get("steps")==180 and data.get("original_G1_trial") is False and
              data.get("device")==row["device"] and data.get("mode")==row["mode"] and
              data.get("counts",{}).get("poll_calls")==180 and
              data.get("force_max_N",0)>10)
    record={"case":case,"device":row["device"],"mode":row["mode"],"rc":code,
       "watchdog":watchdog,"elapsed_s":round(time.monotonic()-t,2),
       "lifecycle_ok":lifecycle,"isolated_physics_certified":physical,
       "native_positive":bool(physical and data.get("native_pair_calibrated")),
       "original_G1_shots":0,"force_max_N":data.get("force_max_N"),
       "contact_steps":data.get("contact_steps"),"z":data.get("tensor_z_samples",[]),
       "native_counts":data.get("counts",{}),"errors":data.get("errors",[]),
       "raw_sha256":sha(raw) if raw.exists() else None,"log_sha256":sha(log),
       "last_lines":txt.splitlines()[-14:]}
    with audit.open("x") as f:json.dump(record,f,indent=2);f.write("\n")
    print("V088_TRIAL",case,"PHYSICS",physical,"NATIVE",record["native_positive"],
          "LIFECYCLE",lifecycle,"CODE",code,flush=True)
    return record
def main():
    c=check_config(); protocol=sha(CFG); commit=git("rev-parse","HEAD");rows=[]
    lock=Path(c["global_gpu_lock"])
    if not lock.is_file():raise RuntimeError("GLOBAL_GPU_LOCK_NOT_FOUND")
    with lock.open("r+") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in c["matrix"]:
            r=trial(c,row,protocol,commit);rows.append(r)
            if c["stop_on_infrastructure_error"] and not r["isolated_physics_certified"]:
                break
    result={"version":c["version"],"parent_sha":c["parent_sha"],
       "frozen_commit":commit,"protocol_sha256":protocol,
       "certified_isolated_physx":sum(x["isolated_physics_certified"] for x in rows),
       "uncertified_attempts":sum(not x["isolated_physics_certified"] for x in rows),
       "native_positive_controls":sum(x["native_positive"] for x in rows),
       "new_original_29dof_g1_shots":0,
       "original_g1_wrist_hip_pair_certified":False,
       "unexecuted":[x["case"] for x in c["matrix"][len(rows):]],"rows":rows}
    with (ROOT/"reports/v088_batch.audit.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V088_BATCH_FINISHED",json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)
if __name__=="__main__":main()
