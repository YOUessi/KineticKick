#!/usr/bin/env python3
"""One write-once G1 pre-shutdown original exception capture; unscored."""
import fcntl, hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from v083_run_physics_batch import PYTHON
ROOT=Path(__file__).resolve().parents[1]
NAME="v115_error_capture_AG_v32_off"
FILES=("tools/v115_eval_guarded_diagnostic.py","tools/v115_diagnostic_poll_strike.py",
"docs/V115_ORIGINAL_EXCEPTION_CAPTURE_FROZEN.md","tests/test_v115_original_exception_forensic.py","tools/v115_exception_probe_batch.py")
def hashfile(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda:f.read(1048576),b""):h.update(buf)
    return h.hexdigest()
def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def main():
    if git("branch","--show-current")!="version/v0.1.115":raise ValueError("WRONG_SOURCE_BRANCH")
    if git("status","--porcelain","--untracked-files=no"):raise RuntimeError("TRACKED_SOURCE_DIRTY")
    for p in FILES:
        if (ROOT/p).read_bytes()!=subprocess.check_output(["git","show","HEAD:"+p],cwd=ROOT):
            raise RuntimeError("UNFROZEN_SOURCE_"+p)
    if shutil.disk_usage(ROOT).free/1024**3<20:raise RuntimeError("DISK_GATE")
    if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True).strip():
        raise RuntimeError("GPU_BUSY")
    report=ROOT/"reports";raw=report/(NAME+".json");log=report/(NAME+".log");audit=report/(NAME+".audit.json")
    if any(p.exists() or p.is_symlink() for p in (raw,log,audit)):raise FileExistsError("WRITE_ONCE_DIAGNOSTIC")
    lock=Path("/home/you/projects/kinetickick/experiments/kinetickick_physics_gpu.lock")
    cmd=[PYTHON,str(ROOT/"tools/v115_diagnostic_poll_strike.py"),
         "--config",str(ROOT/"configs/cmu_real_120hz_10_02_physx_zero_v052.json"),
         "--output",str(raw),"--ball-offset","-.042",".013",
         "--velocity-delta","-.008","--eval-seed","2731","--contact-diagnostics"]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
             PYTHONPATH=":".join([str(ROOT/"tools/v115_startup_readonly"),str(ROOT/"tools"),
                str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]))
    t=time.monotonic()
    with lock.open("r+") as mutex:
        try:fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("OTHER_GPU_JOB_ALREADY_OWNS_MUTEX")
        with log.open("x") as f:
            f.write("V115_DIAG_SOURCE_SHA="+git("rev-parse","HEAD")+"\n"
                    "V115_GUARD_SHA="+hashfile(ROOT/FILES[0])+"\n")
            f.flush()
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            timed=False
            try:rc=child.wait(timeout=125)
            except subprocess.TimeoutExpired:
                timed=True
                os.killpg(child.pid,signal.SIGTERM)
                try:rc=child.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=6)
                f.write("\nV115_G1_DIAGNOSTIC_CHILD_WATCHDOG\n")
    tlog=log.read_text(errors="replace")
    lines=tlog.splitlines()
    matches=[(i,x) for i,x in enumerate(lines) if "V115_UNMASKED_PRE_SHUTDOWN_ORIGINAL_EXCEPTION" in x]
    result=dict(version="v0.1.115",certified_original_controller_rollout=False,
        original_exception_captured=bool(matches),
        first_unmasked_exception=lines[matches[0][0]] if matches else None,
        traceback_excerpt="\n".join(lines[matches[0][0]:matches[0][0]+30]) if matches else None,
        stopped_in_Kit_close=bool("simulation_app.py" in tlog and "in close" in tlog),
        source_sha=git("rev-parse","HEAD"),source_runner_sha=hashfile(ROOT/FILES[1]),
        rc=rc,timeout=timed,elapsed_s=round(time.monotonic()-t,2),
        raw_created=raw.exists(),log_sha256=hashfile(log),
        new_scored_G1_physics_runs=0)
    with audit.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V115_UNMASKED_DIAGNOSTIC",json.dumps({k:v for k,v in result.items() if k!="traceback_excerpt"}))
if __name__=="__main__":main()
