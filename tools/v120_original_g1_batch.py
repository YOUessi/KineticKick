#!/usr/bin/env python3
"""Write-once frozen original G1 200Hz OFF/ON PhysX matrix with full shared-field parity.
This code NEVER trains a new policy or certifies native contact manifold.
"""
from __future__ import annotations
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from v083_run_physics_batch import certified_physics,PYTHON
from audit_v067_hard_gates import physically_scored
from v116_asset_qualification import assert_all_assets
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v120_g1_frozen.json"
MASTER=ROOT/"reports/v120_original_g1_batch.audit.json"
FROZEN=("configs/v120_g1_frozen.json","docs/V120_G1_FROZEN_PROTOCOL.md",
        "tools/v120_original_g1_batch.py","tests/test_v120_g1_protocol.py",
        "tools/v080_run_poll_strike.py","tools/v098_run_pair_strike.py",
        "tools/v098_g1_tensor_pair.py","tools/v116_200hz_coupled_audit.py",
        "tools/v116_asset_qualification.py")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def resources(c):
    free=shutil.disk_usage(ROOT).free/(1024**3)
    if free<c["min_free_gib"]:raise RuntimeError("DISK_BELOW_20_GIB")
    gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
        "--format=csv,noheader"],text=True).strip()
    if gpu:raise RuntimeError("FOREIGN_GPU_COMPUTE_PROCESS:"+gpu)
    return round(free,3)
def preflight():
    c=json.loads(CFG.read_text())
    if git("branch","--show-current")!="version/v0.1.118":raise RuntimeError("WRONG_BRANCH")
    if git("status","--porcelain","--untracked-files=no"):raise RuntimeError("TRACKED_SOURCE_DIRTY")
    if MASTER.exists() or MASTER.is_symlink():raise FileExistsError("EXISTING_MASTER_AUDIT")
    if c["parent_commit"] not in git("log","--format=%H"):raise ValueError("WRONG_PARENT_SHA")
    if c["per_child_watchdog_s"]!=140 or c["min_free_gib"]!=20:raise ValueError("SAFETY_CHANGED")
    if c["off_original_runner"]!="tools/v080_run_poll_strike.py" or c["on_original_runner"]!="tools/v098_run_pair_strike.py" or c["on_only_flag"]!="--tensor-pair-v098":raise ValueError("CHANGED_ORIGINAL_RUNNER")
    expected=[]
    for scene,offset,vel,seed in (
      ("AO",[.041,-.038],.004,3119),
      ("AP",[-.036,.073],-.004,3167)):
        for ctl in ("v32","cmu02","cmu05"):
            for ob in ((False,True) if ctl=="v32" and scene=="AO" else (False,)):
                expected.append((scene,ctl,ob,offset,vel,seed))
    got=[(r["scene"],r["control"],r["observer"],r["ball_offset_xy"],r["velocity_delta"],r["seed"]) for r in c["matrix"]]
    if got!=expected:raise ValueError("UNREGISTERED_RUN_MATRIX")
    if c["learned_policy_actuation"] or not c["stop_after_first_unverified_or_parity_fail"]:raise ValueError("UNSAFE_PROTOCOL")
    for path in FROZEN:
        if (ROOT/path).read_bytes()!=subprocess.check_output(["git","show","HEAD:"+path],cwd=ROOT):
            raise ValueError("TRACKED_FROZEN_SOURCE_DRIFT:"+path)
    for p in c["control_configs"].values():
        d=json.loads((ROOT/p).read_text())
        for name,want in d["frozen_files_sha256"].items():
            if sha(ROOT/name)!=want:raise ValueError("CONTROLLER_SHA_DRIFT:"+name)
        for n in ("template","reference","expert_plan"):
            if sha(ROOT/d[n])!=d[n+"_sha256"]:raise ValueError("HUMAN_PROXY_ASSET_SHA_DRIFT:"+n)
    for p in ("third_party/HumanoidSoccer/source/whole_body_tracking/soccer",
              "experiments/runtime/onnxruntime/onnxruntime",
              ".pretrained_checkpoints/paid_official/policy_30000.onnx",
              "configs/paid_original/paid_main_resolved.urdf"):
        if not (ROOT/p).exists():raise FileNotFoundError(p)
    assert_all_assets(ROOT, c)
    resources(c)
    return c
def _max_abs(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):return float("inf")
        return max((_max_abs(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return float("inf")
        return max((_max_abs(x,y) for x,y in zip(a,b)),default=0.)
    if isinstance(a,bool) != isinstance(b,bool):return float("inf")
    if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
        return abs(float(a)-float(b))
    return 0. if type(a) is type(b) and a==b else float("inf")
def parity(off,on):
    if not (off["certified_original_G1_cuda_200Hz"] and on["certified_original_G1_cuda_200Hz"]):
        return {"passed":False,"reason":"UNQUALIFIED_G1_PAIR"}
    a=json.loads((ROOT/"reports"/("v120_"+off["name"]+".json")).read_text())
    b=json.loads((ROOT/"reports"/("v120_"+on["name"]+".json")).read_text())
    tr0=a.get("traces",[[]])[0];tr1=b.get("traces",[[]])[0]
    md=_max_abs(tr0,tr1);rd=_max_abs(a.get("records"),b.get("records"))
    extra={k:_max_abs(a.get(k),b.get(k)) for k in (
        "contacts","termination_reasons","kinematics","sequence_records",
        "stride_shot_evidence","torque_evidence","ground_collision_paths")}
    matching=(bool(tr0) and len(tr0)==len(tr1) and md==0 and rd==0 and
              all(v==0 for v in extra.values()))
    return {"passed":matching,"scene":off["scene"],"control":off["control"],
            "seed":off["seed"],"all_true_shared_physical_trace_fields_exact":matching,
            "max_abs_full_recorded_200Hz_physical_delta":md,
            "max_abs_original_shot_record_delta":rd,
            "other_original_physical_and_event_record_max_abs":extra,
            "true_recorded_trace_rows":len(tr0),
            "native_actor_collider_manifold_certified":False}
def once(c,trial):
    prefix="v120_"+trial["name"]
    raw=ROOT/"reports"/(prefix+".json")
    log=ROOT/"reports"/(prefix+".log")
    audit=ROOT/"reports"/(prefix+".audit.json")
    if any(p.exists() or p.is_symlink() for p in (raw,log,audit)):
        raise FileExistsError("WRITE_ONCE_EVIDENCE:"+prefix)
    source=c["on_original_runner"] if trial["observer"] else c["off_original_runner"]
    cmd=[PYTHON,str(ROOT/source),
         "--config",str(ROOT/c["control_configs"][trial["control"]]),
         "--output",str(raw),"--ball-offset",*[str(x) for x in trial["ball_offset_xy"]],
         "--velocity-delta",str(trial["velocity_delta"]),"--eval-seed",str(trial["seed"]),
         "--contact-diagnostics"]
    if trial["observer"]:cmd.append(c["on_only_flag"])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
        PYTHONPATH=":".join((str(ROOT/"tools/v116_startup_readonly"),str(ROOT/"tools"),str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"))))
    free=resources(c)
    start=time.monotonic()
    with log.open("x") as out:
        out.write("V120_SOURCE_COMMIT="+git("rev-parse","HEAD")+"\n")
        out.write("V120_PROTOCOL_SHA="+sha(CFG)+"\n")
        out.write("V120_RUNNER_SHA="+sha(ROOT/source)+"\n")
        out.write("V120_ORIGINAL_CMD="+repr(cmd)+"\n");out.flush()
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        timed=False
        try:rc=child.wait(timeout=c["per_child_watchdog_s"])
        except subprocess.TimeoutExpired:
            timed=True
            os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=6)
            out.write("\nV120_CHILD_ONLY_WATCHDOG_TIMEOUT\n")
    payload={}
    if raw.exists():
        try:payload=json.loads(raw.read_text())
        except Exception:pass
    valid,reason=certified_physics(payload,log.read_text(errors="replace"),trial,rc,timed)
    gates={"hard_gate_pass":False,"failed_gates":["NO_CERTIFIED_COMPLETE_G1_200HZ"]}
    if valid:
        try:gates=physically_scored(payload,wrapper_certified=True)
        except Exception as e:gates={"hard_gate_pass":False,"failed_gates":["INDEPENDENT_AUDIT_ERROR"],"error":repr(e)}
    evidence=payload.get("g1_tensor_pair_v098")
    record=dict(**trial,certified_original_G1_cuda_200Hz=valid,
        certification=reason,hard_gates=gates,
        actual_traced_200hz_rows=len(payload.get("traces",[[]])[0]) if valid else 0,
        observer_positive_reports=evidence,
        native_colliders_manifold_certified=False,
        source_runner_sha256=sha(ROOT/source),raw_sha256=sha(raw) if raw.exists() else None,
        log_sha256=sha(log),exit_code=rc,timeout=timed,
        elapsed_s=round(time.monotonic()-start,3),free_disk_pre_gib=free)
    with audit.open("x") as out:json.dump(record,out,indent=2);out.write("\n")
    print("V120_EPISODE",trial["name"],"certified",valid,"steps",record["actual_traced_200hz_rows"],
        "hard",gates["hard_gate_pass"],"fail",gates["failed_gates"],"time",record["elapsed_s"],flush=True)
    return record
def main():
    c=preflight();runs=[];pairs=[]
    lock=Path(c["global_gpu_lock"])
    if not lock.exists():raise FileNotFoundError("GLOBAL_GPU_LOCK_MISSING")
    with lock.open("r+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GLOBAL_GPU_ALREADY_OWNED")
        for i,trial in enumerate(c["matrix"]):
            row=once(c,trial);runs.append(row)
            if not row["certified_original_G1_cuda_200Hz"]:break
            if trial["observer"]:
                compare=parity(runs[-2],runs[-1]);pairs.append(compare)
                print("V120_PARITY",compare,flush=True)
                if not compare["passed"]:break
    result=dict(version="v0.1.118",source_commit=git("rev-parse","HEAD"),
       protocol_sha256=sha(CFG),true_completed_scored_original_G1_runs=sum(x["certified_original_G1_cuda_200Hz"] for x in runs),
       new_strict_pass=sum(x["certified_original_G1_cuda_200Hz"] and x["hard_gates"]["hard_gate_pass"] for x in runs),
       completed_200hz_original_G1_physics_steps=sum(x["actual_traced_200hz_rows"] for x in runs),
       all_shared_physical_state_parity_pass=sum(p["passed"] for p in pairs),
       new_G1_attempts=len(runs),
       not_executed_due_to_fail_closed=[x["name"] for x in c["matrix"][len(runs):]],
       native_wrist_hip_manifold_certified=False,learned_policy_actuation=False,
       runs=runs,parity=pairs)
    with MASTER.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V120_BATCH_SUMMARY",json.dumps({k:v for k,v in result.items() if k not in ("runs","parity")}),flush=True)
    return result
if __name__=="__main__":main()
