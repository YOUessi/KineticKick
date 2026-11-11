#!/usr/bin/env python3
"""v84 write-once 7 original-G1 CUDA/PhysX episodes. No new policy activation."""
from __future__ import annotations
import fcntl, hashlib, json, os, subprocess, signal, time, shutil
from pathlib import Path
from v083_run_physics_batch import certified_physics, sha, PYTHON
from audit_v067_hard_gates import physically_scored

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v084_frozen_generalization.json"
OUTPUT=ROOT/"reports/v084_g1_physics_final.audit.json"
FROZEN=["configs/v084_frozen_generalization.json","docs/V084_FROZEN_PROTOCOL.md",
        "tools/v084_batch.py","tools/v084_evidence.py",
        "tests/test_v084_independent_evidence.py"]
def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT).decode().strip()

def resources(c):
    free=shutil.disk_usage(ROOT).free/1024**3
    if free<c["settings"]["min_free_gib"]:
        raise RuntimeError("DISK_BELOW_20GIB")
    gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
                                  "--format=csv,noheader"],text=True).strip()
    if gpu:raise RuntimeError("GPU_BUSY:"+gpu)
    return round(free,3)

def run_child(c,stem,args,timeout):
    out=ROOT/"reports"/(stem+".json")
    log=ROOT/"reports"/(stem+".log")
    side=ROOT/"reports"/(stem+".audit.json")
    if any(x.exists() or x.is_symlink() for x in (out,log,side)):
        raise FileExistsError("WRITE_ONCE:"+stem)
    disk=resources(c)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
        PYTHONPATH=":".join([str(ROOT/"tools"),str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    start=time.monotonic()
    with log.open("x") as f:
        f.write("V084_FROZEN_COMMIT="+c["_frozen_commit"]+"\n")
        f.write("V084_PROTOCOL_SHA="+c["_protocol_sha256"]+"\nCMD="+repr(args)+"\n")
        f.flush()
        proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=f,
                              stderr=subprocess.STDOUT,start_new_session=True)
        timed=False
        try:rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed=True
            os.killpg(proc.pid,signal.SIGTERM)
            try:rc=proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                rc=proc.wait(timeout=5)
            f.write("\nV084_CHILD_ONLY_WATCHDOG_STOP\n")
    return ({"run":stem,"rc":rc,"watchdog":timed,
        "elapsed_s":round(time.monotonic()-start,2),"disk_free_preflight_gib":disk,
        "output_exists":out.exists(),"raw_sha256":sha(out) if out.exists() else None,
        "log_sha256":sha(log),"last_log_lines":log.read_text(errors="replace").splitlines()[-8:]},
        out,log,side)
def preflight():
    if git("branch","--show-current")!="version/v0.1.84":
        raise RuntimeError("WRONG_VERSION_BRANCH")
    if git("status","--porcelain","--untracked-files=no"):
        raise RuntimeError("TRACKED_DIRTY")
    for f in FROZEN:
        actual=(ROOT/f).read_bytes()
        committed=subprocess.check_output(["git","show","HEAD:"+f],cwd=ROOT)
        if actual!=committed:
            raise RuntimeError("UNFROZEN_PROTOCOL_OR_EXECUTOR:"+f)
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    c=json.loads(CFG.read_text())
    if c["version"]!="v0.1.84" or c["parent_commit"] not in git("log","--format=%H"):
        raise RuntimeError("PARENT_OR_VERSION_DRIFT")
    if len(c["cases"])!=7 or not c["actual_reset_perturbation"] is False:
        raise RuntimeError("TRIAL_MATRIX_CHANGED")
    for source in c["controls"].values():
        spec=json.loads((ROOT/source).read_text())
        for path,expect in spec["frozen_files_sha256"].items():
            if not (ROOT/path).is_file() or sha(ROOT/path)!=expect:
                raise RuntimeError("SOURCE_DRIFT:"+path)
        for k in ("template","reference","expert_plan"):
            if sha(ROOT/spec[k])!=spec[k+"_sha256"]:
                raise RuntimeError("ASSET_DRIFT:"+k)
    if not (ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking/soccer").is_dir():
        raise RuntimeError("FROZEN_UPSTREAM_CONTROLLER_MISSING")
    if not (ROOT/"experiments/runtime/onnxruntime/onnxruntime").is_dir():
        raise RuntimeError("FROZEN_ONNX_RUNTIME_MISSING")
    c["_frozen_commit"]=git("rev-parse","HEAD")
    c["_protocol_sha256"]=sha(CFG)
    resources(c)
    return c
def run_trial(c,trial):
    stem="v084_"+trial["name"]
    out=ROOT/"reports"/(stem+".json")
    args=[PYTHON,str(ROOT/"tools/v080_run_poll_strike.py"),
          "--config",str(ROOT/c["controls"][trial["control"]]),
          "--output",str(out),"--ball-offset",
          *[str(v) for v in trial["ball_offset_xy"]],
          "--velocity-delta",str(trial["velocity_delta"]),
          "--eval-seed",str(trial["seed"]),"--contact-diagnostics"]
    if trial["native_poll"]:args+=["--native-readonly-poll-v080"]
    row,rawpath,log,side=run_child(c,stem,args,c["settings"]["timeout_s"])
    raw={}
    if rawpath.exists():
        try:raw=json.loads(rawpath.read_text())
        except Exception as e:row["decode_error"]=repr(e)
    valid,reason=certified_physics(raw,log.read_text(errors="replace"),
                                    trial,row["rc"],row["watchdog"])
    row.update(name=trial["name"],scene=trial["scene"],control=trial["control"],
        seed=trial["seed"],native_poll=trial["native_poll"],
        true_original_200Hz_G1_CUDA_PhysX=valid,validation=reason,
        frames=len(raw.get("traces",[[]])[0]) if raw else 0)
    if valid:
        try:row["gates"]=physically_scored(raw,wrapper_certified=True)
        except Exception as exc:row["gates"]={"hard_gate_pass":False,
                 "failed_gates":["INDEPENDENT_SCORER_ERROR"],"exception":repr(exc)}
    else:row["gates"]={"hard_gate_pass":False,
                "failed_gates":["NO_CERTIFIED_ORIGINAL_CUDA_G1_PHYSICS"]}
    with side.open("x") as f:json.dump(row,f,indent=2);f.write("\n")
    print("V084_TRIAL",trial["name"],"VALID",valid,"FRAMES",row["frames"],
          "PASS",row["gates"]["hard_gate_pass"],
          "FAILED",row["gates"]["failed_gates"],flush=True)
    return row

def parity(root,off="C_v32_off",on="C_v32_on"):
    a=json.loads((root/"reports"/("v084_"+off+".json")).read_text())
    b=json.loads((root/"reports"/("v084_"+on+".json")).read_text())
    ta,tb=a["traces"][0],b["traces"][0]
    different=[]
    if len(ta)!=len(tb):different.append("FRAME_COUNT")
    for i,(x,y) in enumerate(zip(ta,tb)):
        if set(x)!=set(y):
            different.append("FRAME_KEYS:"+str(i));break
        for k in x:
            if x[k]!=y[k]:
                different.append("frame:"+str(i)+":"+k)
                if len(different)>=12:break
        if len(different)>=12:break
    for field in ("records","contacts","joint_names","termination_reasons","sequence_records"):
        if a.get(field)!=b.get(field):different.append("global:"+field)
    native=b.get("physx_native_poll_v080")
    if not isinstance(native,dict):
        different.append("MISSING_NATIVE_READONLY_POLL_RESULT")
    return {"pass":not different,"common_physics_exact":not different,
        "mismatches_first":different,"frames_compared":min(len(ta),len(tb)),
        "read_only_poll":native,"native_wrist_hip_contact_certified":
            bool(native and native.get("wrist_hip_direct_contact_certified")),
        "zero_native_headers_means":"UNKNOWN_NOT_ABSENCE"}

def main():
    c=preflight()
    lock=Path(c["settings"]["gpu_lock"])
    rows=[]
    with lock.open("r+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("CONCURRENT_GPU_PHYSICS")
        for t in c["cases"]:
            resources(c)
            row=run_trial(c,t);rows.append(row)
            if not row["true_original_200Hz_G1_CUDA_PhysX"]:break
    pair=None
    if all(any(x["name"]==k and x["true_original_200Hz_G1_CUDA_PhysX"]
               for x in rows) for k in ("C_v32_off","C_v32_on")):
        pair=parity(ROOT)
    report={
        "version":"v0.1.84","frozen_commit":c["_frozen_commit"],
        "protocol_sha256":c["_protocol_sha256"],
        "real_original_G1_CUDA_PhysX_episodes":sum(
            x["true_original_200Hz_G1_CUDA_PhysX"] for x in rows),
        "strict_pass":sum(x["true_original_200Hz_G1_CUDA_PhysX"]
                           and x["gates"]["hard_gate_pass"] for x in rows),
        "strict_fail":sum(x["true_original_200Hz_G1_CUDA_PhysX"]
                           and not x["gates"]["hard_gate_pass"] for x in rows),
        "unscored_gpu_attempts":sum(not x["true_original_200Hz_G1_CUDA_PhysX"]
                                    for x in rows),
        "unexecuted_frozen_trials":[x["name"] for x in c["cases"][len(rows):]],
        "paired_readonly_physical_parity":pair,
        "human_professional_kinetic_impulse_learned":False,
        "native_direct_wrist_hip_manifold_certified":bool(
            pair and pair["native_wrist_hip_contact_certified"]),
        "new_wholebody_policy_physx_executed":False,
        "runs":rows}
    with OUTPUT.open("x") as f:json.dump(report,f,indent=2);f.write("\n")
    print("V084_DONE",json.dumps({k:v for k,v in report.items()
          if k not in ("runs","paired_readonly_physical_parity")}),flush=True)
if __name__=="__main__":main()
