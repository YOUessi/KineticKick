#!/usr/bin/env python3
"""v76 preregistered real G1 native-contact paired trial; never modifies old JSON."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
ORIG=Path("/home/you/projects/kinetickick")
V75=Path("/home/you/projects/kinetickick-wt-v075")
CONF=ROOT/"configs/v076_frozen_g1_native_shutdown.json"
PYTHON="/home/you/conda_env/env_isaaclab/bin/python"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def preflight(c):
    if c["version"]!="v0.1.76":raise RuntimeError("BAD_PROTOCOL")
    b=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
    if b!="version/v0.1.76":raise RuntimeError("BAD_BRANCH")
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip():
        raise RuntimeError("TRACKED_DIRTY")
    for s,expected in c["immutable_inputs_sha256"].items():
        if sha(ROOT/s)!=expected:raise RuntimeError("SOURCE_DRIFT:"+s)
    for s,expected in c["base_sources_sha256"].items():
        if sha(ORIG/s)!=expected:raise RuntimeError("ORIGINAL_PHYSICS_CHANGED:"+s)
    if sha(c["app_runtime_source"])!=c["app_runtime_sha256"]:
        raise RuntimeError("ISAAC_SIM_VERSION_DRIFT")
    for name in ("post_physics_guarded_close","post_physics_guarded_stop"):
        result=json.loads((V75/("reports/v075_"+name+".json")).read_text())
        if (result["watchdog"] or result["child_code"]!=0 or
            result["simulated_steps"]!=8):
            raise RuntimeError("PRIOR_KIT_GUARD_NO_PROCESS_LEVEL_EXIT")
    free=os.statvfs(ROOT)
    gib=free.f_bavail*free.f_frsize/1024**3
    if gib<c["minimum_free_GiB"]:raise RuntimeError("DISK_BELOW20_GIB")
    r=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],
                     capture_output=True,text=True,check=True,timeout=12)
    if r.stdout.strip():raise RuntimeError("GPU_BUSY:"+r.stdout)
    return round(gib,2)

def physical_fields(a):
    keys=("traces","records","sequence_records","contacts","termination_reasons",
          "initial_robot_state","joint_velocity_limits","joint_names","joint_effort_limits",
          "automatic_resets_suppressed","post_initialization_state_writes","torque_evidence")
    return {k:a.get(k) for k in keys}

def parity(off,on):
    if off.get("seed")!=on.get("seed"):return False,["seed"]
    left=physical_fields(off);right=physical_fields(on)
    bad=[k for k in left if left[k]!=right[k]]
    return not bad,bad

def single(case,mode,c,free,run_id):
    label=case["name"]
    pre=ROOT/("reports/v076_g1_"+label+"_"+mode+"_"+run_id)
    logfile=Path(str(pre)+".log")
    dest=Path(str(pre)+".json")
    sidecar=Path(str(pre)+"_audit.json")
    if any(p.exists() for p in (logfile,dest,sidecar)):
        raise FileExistsError("FROZEN_WRITE_ONCE:"+str(pre))
    command=[PYTHON,str(ROOT/"tools/v076_run_placement_strike.py"),
             "--config",str(ROOT/case["config"]),"--output",str(dest),
             "--contact-diagnostics","--eval-seed","1201"]
    if mode=="on":command+=["--native-readonly-pair-v056"]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",PYTHONPATH=":".join([
             str(ROOT/"tools"),str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),
             OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    start=time.monotonic();timed=False
    with logfile.open("x") as out:
        out.write("V076_FROZEN_CMD "+repr(command)+"\n")
        out.flush()
        proc=subprocess.Popen(command,env=env,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
                              start_new_session=True)
        try:code=proc.wait(timeout=c["gpu_max_seconds_per_g1_trial"])
        except subprocess.TimeoutExpired:
            timed=True
            try:os.kill(proc.pid,signal.SIGUSR1)
            except ProcessLookupError:pass
            time.sleep(2)
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                code=proc.wait(timeout=5)
            out.write("\nV076_OWNED_G1_CHILD_WATCHDOG\n")
    valid=False;payload={};error=None
    if dest.exists():
        try:
            payload=json.loads(dest.read_text())
            valid=bool(payload.get("traces") and payload.get("joint_names") and
                       len(payload["joint_names"])==29 and payload.get("records"))
        except Exception as exc:
            error=type(exc).__name__+":"+str(exc)[:150]
    text=logfile.read_text(errors="replace")
    row=dict(case=label,mode=mode,code=code,watchdog=timed,
             real_G1_episode=bool(valid and code==0 and not timed),
             step_count=len(payload["traces"][0]) if valid else None,
             source_diagnostics=payload.get("termination_reasons"),
             native_report=payload.get("physx_native_report_v056",{}) if mode=="on" else None,
             guard_marker_present="V076_POST_PHYSICS_STOP_CALLBACK_GUARD_TRUE" in text,
             child_elapsed_s=round(time.monotonic()-start,2),
             full_raw_sha256=sha(dest) if dest.exists() else None,
             full_log_sha256=sha(logfile),config_sha256=sha(CONF),
             controller_source_unchanged=True,free_gib_before=free,error=error)
    with sidecar.open("x") as f:json.dump(row,f,ensure_ascii=False,indent=2);f.write("\n")
    print("V076_G1_RESULT",json.dumps({k:v for k,v in row.items() if k!="native_report"},
                                     ensure_ascii=False),flush=True)
    return row,payload

def main():
    args=argparse.ArgumentParser()
    args.add_argument("--cases",default="all",help="all or one frozen case name")
    args.add_argument("--run-id",default="r1",choices=("r1","r2",),help="write-once namespace for repairing missing local source link")
    opts=args.parse_args()
    c=json.loads(CONF.read_text())
    requested=c["physical_trials_order"] if opts.cases=="all" else [
        x for x in c["physical_trials_order"] if x["name"]==opts.cases]
    if not requested:raise RuntimeError("UNKNOWN_FROZEN_CASE")
    lock=Path(c["gpu_lock"])
    with lock.open("r+") as fp:
        fcntl.flock(fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case in requested:
            first,off=single(case,"off",c,preflight(c),opts.run_id)
            if not first["real_G1_episode"] or not first["guard_marker_present"]:
                raise RuntimeError("ORIGINAL_G1_INCOMPLETE_STOP_BEFORE_ON:"+case["name"])
            second,on=single(case,"on",c,preflight(c),opts.run_id)
            if not second["real_G1_episode"] or not second["guard_marker_present"]:
                raise RuntimeError("NATIVE_G1_INCOMPLETE_STOP:"+case["name"])
            match,keys=parity(off,on)
            marker=ROOT/("reports/v076_g1_"+case["name"]+"_parity_"+opts.run_id+".json")
            with marker.open("x") as out:
                json.dump(dict(case=case["name"],full_physical_fields_equal=match,
                               differing_fields=keys,original_off_sha=first["full_raw_sha256"],
                               native_on_sha=second["full_raw_sha256"],
                               native_event_headers=second["native_report"].get("native_headers"),
                               pair_direct_certification=second["native_report"].get("rigid_pair_certified",False),
                               case_is_real_gpu_physx=True),out,indent=2)
                out.write("\n")
            print("V076_PAIR_RESULT",case["name"],match,keys,flush=True)
            if not match:raise RuntimeError("NATIVE_INSTRUMENTATION_CHANGED_PHYSICS_STOP")
    print("V076_FROZEN_BATCH_FINISHED",len(requested),"pairs",flush=True)

if __name__=="__main__":main()
