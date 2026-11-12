#!/usr/bin/env python3
"""One-shot, source-frozen, flock-guarded G1 pre-step diagnostic, fail closed."""
from __future__ import annotations
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from v083_run_physics_batch import PYTHON,certified_physics
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v105_frozen_protocol.json"
MASTER=ROOT/"reports/v105_L_v32_exception_diagnosis.audit.json"
RAW=ROOT/"reports/v105_L_v32_exception_diagnosis.raw.json"
LOG=ROOT/"reports/v105_L_v32_exception_diagnosis.log"
FROZEN=("configs/v105_frozen_protocol.json","docs/V105_FROZEN_PROTOCOL.md",
        "tools/v105_g1_root_cause_batch.py","tools/v105_trace_original_exception.py",
        "tools/v080_run_poll_strike.py","tools/v076_eval_guarded.py")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def git(*argv):
    return subprocess.check_output(["git",*argv],cwd=ROOT,text=True).strip()
def preflight_resources(c):
    free=shutil.disk_usage(ROOT).free/1024**3
    if free<c["allow_gpu_if_disk_free_gib_ge"]:raise RuntimeError("DISK_BELOW_20_GIB")
    gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
         "--format=csv,noheader"],text=True).strip()
    if gpu:raise RuntimeError("GPU_OTHER_COMPUTE_APP:"+gpu)
    return round(free,3)
def preflight():
    c=json.loads(CFG.read_text())
    if git("branch","--show-current")!="version/v0.1.105":raise RuntimeError("WRONG_BRANCH")
    if git("status","--porcelain","--untracked-files=no"):raise RuntimeError("TRACKED_SOURCE_DIRTY")
    if c["gpu_attempt_limit"]!=1 or c["gpu_child_timeout_s"]!=75:raise ValueError("BAD_WATCHDOG")
    if c["gpu_scene"]!="L" or c["gpu_seed"]!=1601 or c["gpu_ball_offset_xy"]!=[-.095,-.035] or c["gpu_control"]!="v32":raise ValueError("PRE_REG_DRIFT")
    if c["parent_commit"] not in git("log","--format=%H"):raise ValueError("PARENT_MISSING")
    if any(p.exists() or p.is_symlink() for p in (MASTER,RAW,LOG)):raise FileExistsError("WRITE_ONCE_EVIDENCE")
    for x in FROZEN:
        want=subprocess.check_output(["git","show","HEAD:"+x],cwd=ROOT)
        if (ROOT/x).read_bytes()!=want:raise RuntimeError("TRACKED_SOURCE_DRIFT:"+x)
    control=json.loads((ROOT/c["gpu_control_config"]).read_text())
    for x,h in control["frozen_files_sha256"].items():
        if sha(ROOT/x)!=h:raise RuntimeError("ORIGINAL_CONTROLLER_DRIFT:"+x)
    for x in ("template","reference","expert_plan"):
        if sha(ROOT/control[x])!=control[x+"_sha256"]:raise RuntimeError("HUMAN_PROXY_ASSET_DRIFT:"+x)
    for p in (ROOT/"configs/paid_original/paid_main_resolved.urdf",
              ROOT/".pretrained_checkpoints/paid_official/policy_30000.onnx",
              ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"):
        if not p.exists():raise FileNotFoundError(p)
    return c
def run():
    c=preflight()
    args=[PYTHON,str(ROOT/"tools/v105_trace_original_exception.py"),
          str(ROOT/"tools/v080_run_poll_strike.py"),"--",
          "--config",str(ROOT/c["gpu_control_config"]),"--output",str(RAW),
          "--ball-offset",*[str(x) for x in c["gpu_ball_offset_xy"]],
          "--velocity-delta","0","--eval-seed",str(c["gpu_seed"]),"--contact-diagnostics"]
    env=dict(os.environ,PYTHONUNBUFFERED="1",CUDA_VISIBLE_DEVICES="0",
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
             PYTHONPATH=":".join([str(ROOT/"tools"),str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]))
    start=time.monotonic();lock=Path(c["global_gpu_lock"])
    if not lock.exists():raise FileNotFoundError("GLOBAL_GPU_LOCK_NOT_FOUND:"+str(lock))
    with lock.open("r+") as lf:
        try:fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GLOBAL_GPU_ALREADY_LOCKED")
        free=preflight_resources(c)
        with LOG.open("x") as out:
            out.write("V105_FROZEN_COMMIT="+git("rev-parse","HEAD")+"\n")
            out.write("V105_SOURCE_SHA="+sha(ROOT/"tools/v080_run_poll_strike.py")+"\n")
            out.write("V105_PROTOCOL_SHA="+sha(CFG)+"\n")
            out.write("V105_COMMAND="+repr(args)+"\n");out.flush()
            child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            timed=False
            try:rc=child.wait(timeout=c["gpu_child_timeout_s"])
            except subprocess.TimeoutExpired:
                timed=True
                os.killpg(child.pid,signal.SIGTERM)
                try:rc=child.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=6)
                out.write("\nV105_CHILD_ONLY_WATCHDOG_TIMEOUT\n")
    txt=LOG.read_text(errors="replace")
    raw={}
    if RAW.exists():
        try:raw=json.loads(RAW.read_text())
        except Exception:pass
    trial=dict(name="L_v32_off",scene="L",control="v32",observer=False,ball_offset_xy=[-.095,-.035],velocity_delta=0,seed=1601)
    qualified,reason=certified_physics(raw,txt,trial,rc,timed)
    result=dict(version=c["version"],frozen_commit=git("rev-parse","HEAD"),
           original_controller_sha256=sha(ROOT/"tools/v080_run_poll_strike.py"),
           original_eval_sha256=sha(ROOT/"tools/v076_eval_guarded.py"),
           protocol_sha256=sha(CFG),G1_original_physics_attempts=1,
           original_200Hz_G1_episode_certified=bool(qualified),
           new_scored_G1_shooting_denominator=int(qualified),
           new_physx_200hz_steps=len(raw.get("traces",[[]])[0]) if qualified else 0,
           provenance="FAILURE_ONLY_PYTHON_TRACE_AFFECTS_TIMING_NO_PARITY_CLAIM",
           root_cause_exception_captured="V105_ORIGINAL_PRE_CLOSE_CAUSE" in txt,
           shutdown_guard_interrupt_fired="V105_ABORT_BLOCKING_TEARDOWN_ONLY_ON_EXISTING_EXCEPTION" in txt,
           root_exception_lines=[x for x in txt.splitlines() if "V105_ORIGINAL_PRE_CLOSE_CAUSE" in x],
           rc=rc,timeout=timed,elapsed_s=round(time.monotonic()-start,3),
           log_sha256=sha(LOG),raw_sha256=sha(RAW) if RAW.exists() else None,
           certification=reason,foreign_gpu_work_not_launched=True,
           native_collision_manifold_certified=False,new_policy_motor_actuation=False,
           disk_free_before_gib=free)
    with MASTER.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V105_G1_DIAGNOSTIC",json.dumps(result),flush=True)
    return result
if __name__=="__main__":run()
