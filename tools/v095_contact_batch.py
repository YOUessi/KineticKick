#!/usr/bin/env python3
"""v92 committed-SHA frozen serial isolated PhysX matrix and immutable audit."""
from __future__ import annotations
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/v095_frozen_matrix.json"
REPORT=ROOT/"reports/v095_frozen_contact_batch.audit.json"
FROZEN=("configs/v095_frozen_matrix.json","docs/V095_FROZEN_PROTOCOL.md",
        "tools/v095_contact_probe.py","tools/v095_contact_batch.py",
        "tests/test_v095_contact_contract.py")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()
def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def preflight():
    c=json.loads(CONFIG.read_text())
    if git("branch","--show-current")!="version/v0.1.95":raise RuntimeError("WRONG_BRANCH")
    if git("status","--porcelain","--untracked-files=no"):raise RuntimeError("DIRTY_TRACKED")
    if c["version"]!="v0.1.95" or c["parent_sha"] not in git("log","--format=%H"):
        raise RuntimeError("INVALID_PARENT_SHA")
    if len(c["matrix"])!=9 or len({r["case"] for r in c["matrix"]})!=9:
        raise RuntimeError("MATRIX_DRIFT")
    if c["original_G1_shots_scheduled"]!=0 or c["physical_G1_actuation_new"]:
        raise RuntimeError("UNREGISTERED_G1_ACTUATION")
    for f in FROZEN:
        if (ROOT/f).read_bytes()!=subprocess.check_output(
             ["git","show","HEAD:"+f],cwd=ROOT):
            raise RuntimeError("FROZEN_FILE_CHANGED:"+f)
    if REPORT.exists():raise FileExistsError("WRITE_ONCE_AGGREGATE")
    space=shutil.disk_usage(ROOT).free/1024**3
    if space<c["required_free_gib"]:raise RuntimeError("DISK_BELOW_20GIB")
    if not Path(c["gpu_lock"]).is_file():raise RuntimeError("PROJECT_GLOBAL_MUTEX_MISSING")
    return c
def guard(c):
    free=shutil.disk_usage(ROOT).free/1024**3
    if free<c["required_free_gib"]:raise RuntimeError("DISK_BELOW_20GIB")
    gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
           "--format=csv,noheader"],text=True).strip()
    if gpu:raise RuntimeError("OTHER_GPU_WORK:"+gpu)
    return round(free,2)
def launch(c,row,commit,protocol):
    stem="v095_"+row["case"]
    out=ROOT/"reports"/(stem+".json")
    log=ROOT/"reports"/(stem+".log")
    audit=ROOT/"reports"/(stem+".audit.json")
    if any(p.exists() or p.is_symlink() for p in (out,log,audit)):
        raise FileExistsError("WRITE_ONCE_TRIAL:"+stem)
    free=guard(c)
    args=[c["python"],str(ROOT/"tools/v095_contact_probe.py"),
          "--device",row["device"],"--report-mode",row["report_mode"],
          "--layout",row["layout"],"--capture",row["capture"],
          "--steps",str(c["steps"]),"--subscribe-phase",row.get("subscribe_phase","pre_reset"),"--output",str(out)]
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
             MKL_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
             PYTHONPATH=str(ROOT/"tools"))
    start=time.monotonic()
    with log.open("x") as o:
        o.write("V095_COMMIT="+commit+"\nPROTOCOL_SHA256="+protocol+
                "\nCOMMAND="+repr(args)+"\n");o.flush()
        proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=o,
                              stderr=subprocess.STDOUT,start_new_session=True)
        timeout=False
        try:rc=proc.wait(timeout=c["timeout_s"])
        except subprocess.TimeoutExpired:
            timeout=True
            os.killpg(proc.pid,signal.SIGTERM)
            try:rc=proc.wait(timeout=7)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                rc=proc.wait(timeout=7)
            o.write("\nV095_OWN_CHILD_TIMEOUT_TERMINATION\n")
    txt=log.read_text(errors="replace")
    raw={}
    if out.exists():
        try:raw=json.loads(out.read_text())
        except Exception as exc:raw={"read_error":repr(exc)}
    ok=(rc==0 and not timeout and raw.get("actual_runtime_frames")==180
        and raw.get("device")==row["device"]
        and raw.get("report_mode")==row["report_mode"]
        and raw.get("layout")==row["layout"] and raw.get("capture")==row["capture"]
        and "V095_RESET_OK" in txt and "V095_RESULT" in txt)
    stage=raw.get("stage",{})
    correct_api=(stage.get("ball_report") is (row["report_mode"]=="on") and
                 stage.get("box_report") is (row["report_mode"]=="on"))
    ok=bool(ok and correct_api)
    physically_confirmed=bool(ok and (raw.get("observed_collision_dynamics")
          if row["layout"]=="colliding" else raw.get("observed_noncollision_dynamics")))
    record=dict(**row,rc=rc,timeout=timeout,elapsed_s=round(time.monotonic()-start,2),
      free_gib_at_start=free,raw_sha256=sha(out) if out.exists() else None,
      log_sha256=sha(log),lifecycle_marker_app_closed="V095_APP_CLOSED" in txt,
      recorded_runtime_evidence=ok,physically_confirmed=physically_confirmed,
      correct_report_api_contract=correct_api,child_exception=raw.get("exception"),
      force_positive=bool(raw.get("tensor_pair_witness")),
      native_positive=bool(raw.get("native_pair_witness")),
      native_counts=raw.get("native_counts"),tensor_peak_N=raw.get("tensor_peak_force_N"),
      tensor_contact_counts=raw.get("tensor_contact_counts"),
      tensor_sensor_filter_paths=raw.get("contact_view_shape"),
      errors=raw.get("errors",[]),z_first_last=[raw.get("runtime_z_vz",[])[i][1]
          for i in (0,-1)] if raw.get("runtime_z_vz") else None,
      original_G1_shots=0)
    with audit.open("x") as f:json.dump(record,f,indent=2);f.write("\n")
    print("V095_TRIAL",stem,"OBSERVED",ok,"PHYSICALLY_CONFIRMED",physically_confirmed,
          "TENSOR",record["force_positive"],"NATIVE",record["native_positive"],
          "RC",rc,"APP_CLOSED",record["lifecycle_marker_app_closed"],flush=True)
    return record,raw
def parity(off,on):
    a=off.get("runtime_z_vz");b=on.get("runtime_z_vz")
    if not a or not b or len(a)!=len(b):return {"valid":False,"reason":"TRACES_MISSING"}
    max_abs=max(abs(float(x[j])-float(y[j]))
                for x,y in zip(a,b) for j in (1,2))
    return {"valid":True,"frames":len(a),"max_abs_z_or_vz":max_abs,
            "exact_at_saved_9_digits":max_abs==0,
            "within_1e_minus_6":max_abs<=1e-6}
def main():
    c=preflight();rows=[];raws={};commit=git("rev-parse","HEAD")
    protocol=sha(CONFIG)
    lock=Path(c["gpu_lock"])
    with lock.open("r+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GLOBAL_GPU_MUTEX_BUSY")
        for row in c["matrix"]:
            a,raw=launch(c,row,commit,protocol)
            rows.append(a);raws[row["case"]]=raw
            if (c["stop_on_child_crash"] and (a["rc"]!=0 or a["timeout"])) or (
                  c["stop_on_runtime_infrastructure_failure"] and
                  (not a["recorded_runtime_evidence"] or
                   not a["physically_confirmed"])):
                break
    p={}
    for device in ("cpu","gpu"):
        base=device+"_pose_off"
        for capture in ("native","tensor","both","native_after"):
            candidate=device+"_"+capture+"_on"
            if base in raws and candidate in raws:
                p[device+"_pose_vs_"+capture]=parity(raws[base],raws[candidate])
    result=dict(version="v0.1.95",parent=c["parent_sha"],frozen_commit=commit,
       protocol_sha256=protocol,isolated_physx_attempts=len(rows),
       physically_confirmed=sum(x["physically_confirmed"] for x in rows),
       native_positive=sum(x["native_positive"] for x in rows),
       tensor_positive=sum(x["force_positive"] for x in rows),
       off_on_kinematic_parity=p,original_G1_shots=0,
       direct_original_G1_wrist_hip_manifold=False,
       unexecuted=[x["case"] for x in c["matrix"][len(rows):]],trials=rows)
    with REPORT.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V095_BATCH_DONE",json.dumps({k:v for k,v in result.items() if k!="trials"}),flush=True)
if __name__=="__main__":main()
