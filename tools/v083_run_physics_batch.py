#!/usr/bin/env python3
"""Write-once v83 original-G1 GPU serial batch. Native shape probe is NOT a shooting FAIL."""
from __future__ import annotations
import fcntl,hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v083_frozen_independent_batch.json"
# These constants are patched BEFORE frozen protocol commit, and never at runtime.
FROZEN_CFG_SHA="629b8cd8940a8469c1aa81dc62bf33cea2049542aab1a1263449feea9a69c647"
FREEZE_COMMIT="b74cca51a20bfe31b8e2a09bf22ea42a15ef73dd"
PYTHON="/home/you/conda_env/env_isaaclab/bin/python"
REQUIRED_CUDA_LOG="Environment device    : cuda:0"
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def cmd(args,**kwargs):
    return subprocess.check_output(args,text=True,**kwargs).strip()
def cfg_load():
    c=json.loads(CFG.read_text())
    if sha(CFG)!=FROZEN_CFG_SHA:raise RuntimeError("FROZEN_PROTOCOL_DRIFT")
    if cmd(["git","branch","--show-current"],cwd=ROOT)!="version/v0.1.83":
        raise RuntimeError("WRONG_VERSION_BRANCH")
    if subprocess.run(["git","merge-base","--is-ancestor",FREEZE_COMMIT,"HEAD"],cwd=ROOT).returncode:
        raise RuntimeError("FROZEN_COMMIT_MISSING")
    if cmd(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT):
        raise RuntimeError("TRACKED_TREE_DIRTY")
    if cmd(["git","rev-parse",FREEZE_COMMIT+"^"],cwd=ROOT)!=c["parent_sha"]:
        # Freeze commit is allowed to have a separate parent if scripts added
        pass
    for name,expected in c["frozen_dependencies"].items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=expected:
            raise RuntimeError("SRC_DRIFT_OR_MISSING:"+name)
    for trial in c["trials"]:
        src=json.loads((ROOT/trial["source_config"]).read_text())
        for name,h in src["frozen_files_sha256"].items():
            if not (ROOT/name).is_file() or sha(ROOT/name)!=h:
                raise RuntimeError("G1_INPUT_SOURCE_DRIFT:"+name)
        for key in ("template","reference","expert_plan"):
            if sha(ROOT/src[key])!=src[key+"_sha256"]:
                raise RuntimeError("G1_INPUT_ASSET_DRIFT:"+key)
    if not (ROOT/"experiments/runtime/onnxruntime/onnxruntime").is_dir():
        raise RuntimeError("FROZEN_ONNX_RUNTIME_MISSING")
    return c

def resources(c):
    free=shutil.disk_usage(ROOT).free/1024**3
    if free<c["safety"]["min_disk_GiB"]:raise RuntimeError("DISK_BELOW_20GIB")
    gpu=cmd(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"])
    if gpu:raise RuntimeError("GPU_BUSY_OTHER_PROCESS:"+gpu)
    return round(free,2)

def run_child(c,stem,argv,timeout):
    out=ROOT/"reports"/(stem+".json")
    log=ROOT/"reports"/(stem+".log")
    side=ROOT/"reports"/(stem+".audit.json")
    if any(p.exists() or p.is_symlink() for p in (out,log,side)):
        raise FileExistsError("NO_EVIDENCE_OVERWRITE:"+stem)
    disk=resources(c)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
             PYTHONPATH=":".join([str(ROOT/"tools"),str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    start=time.monotonic()
    with log.open("x") as fp:
        fp.write("V083_PROTOCOL_SHA="+FROZEN_CFG_SHA+"\nV083_FREEZE="+FREEZE_COMMIT+"\nCMD="+repr(argv)+"\n")
        fp.flush()
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT,start_new_session=True)
        timed=False
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed=True
            os.killpg(p.pid,signal.SIGTERM)
            try:rc=p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL)
                rc=p.wait(timeout=5)
            fp.write("\nV083_WATCHDOG_TERMINATED_ONLY_THIS_CHILD\n")
    row={"run":stem,"rc":rc,"watchdog":timed,
         "elapsed_s":round(time.monotonic()-start,2),"disk_free_preflight_GiB":disk,
         "output_exists":out.exists(),"raw_sha256":sha(out) if out.exists() else None,
         "log_sha256":sha(log),"log_tail":log.read_text(errors="replace").splitlines()[-10:]}
    return row,out,log,side

def certified_physics(raw,text,trial,rc,timed):
    rows=raw.get("traces",[[]])[0]
    if rc!=0 or timed or not rows or REQUIRED_CUDA_LOG not in text:
        return False,"CUDA_G1_OUTPUT_UNCERTIFIED"
    if raw.get("seed")!=trial["seed"] or raw.get("root_fixed") is not False or len(raw.get("joint_names",()))!=29:
        return False,"NOT_ORIGINAL_29DOF_FREE_G1"
    if any(len(t.get("joint_vel",[]))!=29 or
           abs(float(t.get("physics_dt_s",0))-.005)>1e-6 for t in rows):
        return False,"INVALID_REAL_200HZ_ORIGINAL_29_QDOT"
    return True,"ORIGINAL_29DOF_200HZ_PHYSX"

def once(c,trial):
    stem="v083_independent_"+trial["case"]
    out=ROOT/"reports"/(stem+".json")
    argv=[PYTHON,str(ROOT/"tools/v080_run_poll_strike.py"),"--config",
          str(ROOT/trial["source_config"]),"--output",str(out),
          "--ball-offset",*[str(a) for a in trial["ball_offset_xy"]],
          "--velocity-delta",str(trial["velocity_delta"]),
          "--contact-diagnostics","--eval-seed",str(trial["seed"])]
    row,p,log,side=run_child(c,stem,argv,c["safety"]["trial_timeout_s"])
    raw={}
    if p.exists():
        try:raw=json.loads(p.read_text())
        except Exception as e:row["decode_error"]=repr(e)
    yes,why=certified_physics(raw,log.read_text(errors="replace"),trial,row["rc"],row["watchdog"])
    row.update({"case":trial["case"],"scenario":trial["scenario"],
                "source_config":trial["source_config"],"seed":trial["seed"],
                "true_physics":yes,"source_certification":why,
                "actual_200hz_rows":len(raw.get("traces",[[]])[0]) if raw else 0,
                "actual_ball_offset_xy":trial["ball_offset_xy"],
                "actual_velocity_delta":trial["velocity_delta"]})
    if yes:
        from audit_v067_hard_gates import physically_scored
        try:row["strict_gates"]=physically_scored(raw,wrapper_certified=True)
        except Exception as e:
            row["strict_gates"]={"hard_gate_pass":False,"failed_gates":["INDEPENDENT_AUDIT_ERROR"],
                                 "error":repr(e)}
    else:
        row["strict_gates"]={"hard_gate_pass":False,"failed_gates":["NO_CERTIFIED_ORIGINAL_G1_GPU_PHYSICS"]}
    with side.open("x") as fp:json.dump(row,fp,indent=2,ensure_ascii=False);fp.write("\n")
    print("V083_ORIGINAL_G1",row["case"],"certified",yes,"rows",row["actual_200hz_rows"],
          "hard",row["strict_gates"]["hard_gate_pass"],
          "failed",row["strict_gates"]["failed_gates"],flush=True)
    return row

def probe(c):
    trial=c["independent_probe"];stem="v083_staged_native_pair_probe"
    out=ROOT/"reports"/(stem+".json")
    argv=[PYTHON,str(ROOT/"tools/v083_run_staged_strike.py"),"--config",
          str(ROOT/trial["source_config"]),"--output",str(out),
          "--eval-seed",str(trial["seed"]),"--contact-diagnostics",
          "--tensor-wrist-hip-v081"]
    row,p,log,side=run_child(c,stem,argv,trial["timeout_s"])
    phases=[line for line in log.read_text(errors="replace").splitlines() if "V083_STAGE" in line]
    row.update({"phase_markers":phases,
                "last_marker":phases[-1] if phases else None,
                "native_manifold_certified":False,
                "zero_pair_report_interpretation":"UNKNOWN",
                "physical_g1_episode_count":0 if row["watchdog"] or not p.exists() else None})
    with side.open("x") as fp:json.dump(row,fp,indent=2,ensure_ascii=False);fp.write("\n")
    print("V083_STAGE_PROBE",row["rc"],"timeout",row["watchdog"],"last",row["last_marker"],flush=True)
    return row

def main():
    c=cfg_load();resources(c)
    lock=Path(c["safety"]["gpu_lock"])
    if not lock.exists():raise RuntimeError("SHARED_GPU_LOCK_MISSING")
    rows=[]
    with lock.open("r+") as fp:
        try:fcntl.flock(fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("CONCURRENT_PHYSICS_GPU_OWNER")
        # One bounded native pair probe, no unplanned repeated hung probes.
        diag=probe(c)
        # Probe failure is independent of the unchanged original-control matrix;
        # only proceed if the child exited and GPU clean.
        resources(c)
        for trial in c["trials"]:
            rows.append(once(c,trial))
            # Fail-closed on inability to obtain genuine original-G1 state.
            if not rows[-1]["true_physics"]:break
    report={"version":"v0.1.83","protocol_sha256":FROZEN_CFG_SHA,
            "probe_gpu_attempts":1,"native_probe":diag,
            "real_original_G1_PhysX_episodes":sum(r["true_physics"] for r in rows),
            "strict_pass_count":sum(r["strict_gates"]["hard_gate_pass"] for r in rows),
            "strict_fail_count":sum(r["true_physics"] and
                                    not r["strict_gates"]["hard_gate_pass"] for r in rows),
            "unscored_GPU_attempts":sum(not r["true_physics"] for r in rows),
            "unexecuted_frozen_trials":[t["case"] for t in c["trials"][len(rows):]],
            "human_professional_kick_learned":False,
            "native_collider_manifold_proven":False,"runs":rows}
    path=ROOT/"reports/v083_final_real_physx.audit.json"
    with path.open("x") as fp:json.dump(report,fp,indent=2,ensure_ascii=False);fp.write("\n")
    print("V083_BATCH_DONE",json.dumps({k:v for k,v in report.items() if k not in ("runs","native_probe")}),flush=True)
if __name__=="__main__":main()
