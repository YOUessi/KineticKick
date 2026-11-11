#!/usr/bin/env python3
"""v79 frozen original-29DoF G1 native PhysX callback off/on paired trials.

This is diagnostic parity, NOT training nor professional human dynamics evidence.
All real G1 trials are serialized under the existing global GPU mutex.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ORIG=Path("/home/you/projects/kinetickick")
V078=Path("/home/you/projects/kinetickick-wt-v078")
CONF=ROOT/"configs/v079_frozen_native_pair_matrix.json"
CONF_SHA="8ca3de101c8df85bbd505ae8ee6f5a2ba75d78d5814619fac10d18c6088fe793"
FREEZE_COMMIT="3a5579a27c53d1e7070abbdfb7ddf0c99ef2973a"
PYTHON="/home/you/conda_env/env_isaaclab/bin/python"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as fp:
        for block in iter(lambda:fp.read(1048576),b""):h.update(block)
    return h.hexdigest()

def read_config():
    if sha(CONF)!=CONF_SHA:raise RuntimeError("PREREGISTERED_PROTOCOL_SHA_DRIFT")
    return json.loads(CONF.read_text())

def run_cmd(cmd, **kw):
    return subprocess.check_output(cmd,text=True,**kw).strip()

def check_sources(c):
    branch=run_cmd(["git","branch","--show-current"],cwd=ROOT)
    if branch!="version/v0.1.79":raise RuntimeError("WRONG_BRANCH")
    if subprocess.run(["git","merge-base","--is-ancestor",FREEZE_COMMIT,"HEAD"],cwd=ROOT).returncode:
        raise RuntimeError("FROZEN_PROTOCOL_NOT_IN_GIT_HISTORY")
    if run_cmd(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT):
        raise RuntimeError("TRACKED_DIRTY")
    for name,digest in c["source_sha256"].items():
        if sha(ROOT/name)!=digest:raise RuntimeError("FROZEN_INPUT_SHA_DRIFT:"+name)
    for name,digest in c["original_v054_sha256"].items():
        if sha(ORIG/name)!=digest:raise RuntimeError("LEGACY_PHYSX_SHA_DRIFT:"+name)
    if sha(V078/"reports/v078_g1_original_v32_liveness_1.physx.json")!=c["existing_v078_real_episode_sha256"]:
        raise RuntimeError("V078_REAL_G1_EVIDENCE_DRIFT")
    if not (ROOT/"experiments/runtime/onnxruntime/onnxruntime").is_dir():
        raise RuntimeError("ONNX_RUNTIME_MISSING")
    from targeted_shooting import controller_provenance
    provenance=json.loads((ROOT/"configs/v078_frozen_boot_diagnostics.json").read_text())
    if controller_provenance(ROOT)!=provenance["full_controller_provenance"]:
        raise RuntimeError("FULL_SOURCE_OR_MODEL_PROVENANCE_CHANGED")
    for trial in c["trial_order"]:
        cfg=json.loads((ROOT/trial["source_config"]).read_text())
        for name,expected in cfg["frozen_files_sha256"].items():
            if sha(ROOT/name)!=expected:raise RuntimeError("FROZEN_CONFIG_SOURCE:"+name)
        for key in ("template","reference","expert_plan"):
            if sha(ROOT/cfg[key])!=cfg[key+"_sha256"]:
                raise RuntimeError("FROZEN_SOURCE_CHANGED:"+key)
    return True

def resources(c):
    free=os.statvfs(ROOT)
    gib=free.f_bavail*free.f_frsize/1024**3
    if gib<c["minimum_free_gib"]:raise RuntimeError("DISK_BELOW20_GIB")
    gpu=run_cmd(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"])
    if gpu:raise RuntimeError("GPU_ALREADY_OWNED:"+gpu)
    return round(gib,3)

def valid_physics(payload,log,trial):
    rows=payload.get("traces",[[]])[0]
    if payload.get("seed")!=trial["seed"] or payload.get("root_fixed") is not False:
        return False,"WRONG_G1_RESET_OR_ROOT"
    if len(payload.get("joint_names",[]))!=29 or not rows or len(rows)<100:
        return False,"NOT_COMPLETE_G1_200HZ"
    if not payload.get("records") or not payload.get("sequence_records"):
        return False,"NO_G1_PHYSICS_RECORD"
    if "Environment device    : cuda:0" not in log:
        return False,"NO_CUDA_ENVIRONMENT_LOG"
    if not rows[-1]["time_s"]>=13.995:
        return False,"NOT_FULL_14S"
    if any(len(row.get("joint_vel",[]))!=29 for row in rows):
        return False,"G1_QDOT_MISSING"
    if any(abs(float(row.get("physics_dt_s",0))-0.005)>0.000001 for row in rows):
        return False,"NOT_200HZ"
    return True,"COMPLETE_ORIGINAL_G1_200HZ"

def evaluate(payload,certified):
    from audit_v067_hard_gates import physically_scored
    if not certified:return {"hard_gate_pass":False,"failed_gates":["no_certified_physics"]}
    return physically_scored(payload,wrapper_certified=True)

def native_summary(report):
    if not isinstance(report,dict):
        return {"status":"NOT_PRESENT","wrist_hip_direct_contact_certified":False}
    api=report.get("report_api_present_on_original_bodies",{})
    events=report.get("saved_interesting_events",[])
    def complete_pair(e):
        paths=e.get("actor_paths",[])+e.get("collider_paths",[])
        matching=("left_wrist_yaw_link","left_hip_roll_link")
        if not all(any(target in p for p in paths) for target in matching):
            return False
        points=e.get("point_sample",[])
        return bool(e.get("reported_points",0)>0 and any(
            cp.get("point_w") is not None and cp.get("normal_w") is not None
            for cp in points))
    candidate=[e for e in events if complete_pair(e)]
    api_present=any(bool(v) and ("left_wrist_yaw_link" in k or "left_hip_roll_link" in k)
                    for k,v in api.items())
    return dict(status="CALLBACK_UNVERIFIED_UNTIL_PARITY",headers=report.get("native_headers"),
                callback_calls=report.get("callback_calls"),errors=report.get("errors"),
                dropped=report.get("dropped_events"),report_api_paths=api,
                native_complete=report.get("report_capture_complete") is True,
                wrist_hip_path_events=report.get("wrist_hip_direct_path_events",0),
                sample_paths=[e.get("collider_paths") for e in candidate[:4]],
                wrist_hip_manifold_candidate=bool(candidate),
                wrist_hip_direct_contact_certified=False,
                original_prim_report_api_present=api_present,
                limitation="No callback events is UNKNOWN. Certification needs parity and all source gates.")

def one(c,trial,mode):
    name=trial["case"]
    prefix=ROOT/"reports"/(c["run_namespace"]+"_"+name+"_"+mode)
    logfile=Path(str(prefix)+".log")
    output=Path(str(prefix)+".json")
    audit_file=Path(str(prefix)+".audit.json")
    if any(p.exists() for p in (logfile,output,audit_file)):
        raise FileExistsError("WRITE_ONCE_OUTPUT_ALREADY_EXISTS:"+str(prefix))
    free=resources(c)
    argv=[PYTHON,str(ROOT/"tools/v076_run_placement_strike.py"),
          "--config",str(ROOT/trial["source_config"]),"--output",str(output),
          "--contact-diagnostics","--eval-seed",str(trial["seed"])]
    if mode=="on":argv.append(c["native_on_arg"])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
             PYTHONPATH=":".join([str(ROOT/"tools"),
                                 str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    t0=time.monotonic()
    timeout=False
    with logfile.open("x") as outputlog:
        outputlog.write("V079_FROZEN_PROTOCOL_SHA="+CONF_SHA+"\n")
        outputlog.write("V079_PARENT_SHA="+FREEZE_COMMIT+"\n")
        outputlog.write("V079_ARGV="+repr(argv)+"\n")
        outputlog.flush()
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=outputlog,
                               stderr=subprocess.STDOUT,start_new_session=True)
        try:
            exit_code=child.wait(timeout=c["gpu_timeout_seconds_per_trial"])
        except subprocess.TimeoutExpired:
            timeout=True
            os.killpg(child.pid,signal.SIGTERM)
            try:exit_code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                exit_code=child.wait(timeout=5)
            outputlog.write("\nV079_KILLED_OWN_GROUP_ON_FROZEN_WATCHDOG\n")
    elapsed=round(time.monotonic()-t0,2)
    text=logfile.read_text(errors="replace")
    payload={}
    parse_error=None
    if output.exists():
        try:payload=json.loads(output.read_text())
        except (ValueError,OSError) as error:parse_error=repr(error)[:180]
    certified,reason=valid_physics(payload,text,trial) if payload else (False,"NO_JSON_OR_PARSE_ERROR")
    certified=certified and exit_code==0 and not timeout
    if not certified and exit_code!=0:reason+=";NONZERO_EXIT"
    scoring=evaluate(payload,certified)
    report=payload.get("physx_native_report_v056",None) if mode=="on" else None
    summary=dict(case=name,mode=mode,source_config=trial["source_config"],seed=trial["seed"],
                 exit_code=exit_code,watchdog=timeout,elapsed_s=elapsed,
                 original_g1_full_physics_certified=certified,certification_reason=reason,
                 rows_200hz=len(payload.get("traces",[[]])[0]) if payload else 0,
                 raw_sha256=sha(output) if output.exists() else None,
                 log_sha256=sha(logfile),protocol_sha256=CONF_SHA,
                 disk_free_gib_preflight=free,parse_error=parse_error,
                 independent_hard_gates=scoring,native_callback=native_summary(report))
    with audit_file.open("x") as fh:json.dump(summary,fh,indent=2,ensure_ascii=False);fh.write("\n")
    print("V079_INDEPENDENT_TRIAL",json.dumps({k:v for k,v in summary.items() if k not in
        ("independent_hard_gates","native_callback")},ensure_ascii=False),flush=True)
    return summary,payload

def compare(c,trial,off,on,off_json,on_json):
    keys=c["parity_keys"]
    mismatches=[k for k in keys if off_json.get(k)!=on_json.get(k)]
    callback=on["native_callback"]
    pair_cert=bool(not mismatches and
        callback.get("wrist_hip_manifold_candidate") and
        callback.get("original_prim_report_api_present") and
        callback.get("native_complete") and
        callback.get("errors")==[])
    state=dict(case=trial["case"],seed=trial["seed"],raw_off_sha=off["raw_sha256"],
               raw_on_sha=on["raw_sha256"],shared_physical_fields_exactly_equal=not mismatches,
               compared_keys=keys,exact_mismatched_keys=mismatches,
               matched_29dof_original_200hz=bool(off["original_g1_full_physics_certified"] and on["original_g1_full_physics_certified"]),
               off_hard_gates=off["independent_hard_gates"],
               on_hard_gates=on["independent_hard_gates"],
               native_on_summary=callback,
               original_readonly_native_wrist_hip_collider_path_certified=pair_cert,
               native_negative_interpretation="UNKNOWN" if not pair_cert else "DIRECT_PATH_WITH_CONTACT_POINT",
               caveat="Native collider pair is NOT certified by equal-opposite net sensor forces; 200Hz foot forces are not ground-truth GRF.")
    path=ROOT/"reports"/(c["run_namespace"]+"_"+trial["case"]+"_parity.audit.json")
    with path.open("x") as fp:json.dump(state,fp,indent=2,ensure_ascii=False);fp.write("\n")
    print("V079_ORIGINAL_G1_PARITY",trial["case"],not mismatches,mismatches,"native_hip_wrist",pair_cert,flush=True)
    return state

def main():
    c=read_config(); check_sources(c); resources(c)
    lock=Path(c["gpu_lock"])
    if not lock.exists():raise RuntimeError("GLOBAL_GPU_MUTEX_MISSING")
    results=[]
    with lock.open("r+") as fp:
        try:fcntl.flock(fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GPU_MUTEX_ALREADY_LOCKED")
        for trial in c["trial_order"]:
            off,off_json=one(c,trial,"off")
            results.append(off)
            if not off["original_g1_full_physics_certified"]:
                raise RuntimeError("STOP_FIRST_OFF_INCOMPLETE:"+trial["case"])
            on,on_json=one(c,trial,"on")
            results.append(on)
            if not on["original_g1_full_physics_certified"]:
                raise RuntimeError("STOP_ON_INCOMPLETE:"+trial["case"])
            comparison=compare(c,trial,off,on,off_json,on_json)
            if not comparison["shared_physical_fields_exactly_equal"]:
                raise RuntimeError("STOP_NATIVE_READONLY_PHYSICS_PARITY_DRIFT:"+trial["case"])
    final={"version":c["version"],"gpu_real_original_G1_episodes":len(results),
           "complete_native_pairs":len(results)//2,
           "strict_hard_gate_pass":sum(r["independent_hard_gates"]["hard_gate_pass"] for r in results),
           "all_pair_parity_pass":True,
           "frozen_protocol_sha":CONF_SHA,
           "native_pair_contact_truth": "SEE_INDIVIDUAL_PARITY_AUDITS"}
    path=ROOT/"reports/v079_g1_final_batch_summary.audit.json"
    with path.open("x") as fp:json.dump(final,fp,indent=2);fp.write("\n")
    print("V079_ALL_FROZEN_TRIALS_COMPLETED",json.dumps(final),flush=True)

if __name__=="__main__":
    main()

