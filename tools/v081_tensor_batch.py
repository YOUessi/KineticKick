#!/usr/bin/env python3
"""SHA-frozen paired six real G1 runs: off vs dual filtered tensor views."""
from __future__ import annotations
import fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
from v080_native_poll_batch import sha,call,validate,resource_check
from audit_v067_hard_gates import physically_scored

ROOT=Path(__file__).resolve().parents[1]
ORIG=Path("/home/you/projects/kinetickick")
PREV=Path("/home/you/projects/kinetickick-wt-v079")
CFG=ROOT/"configs/v081_frozen_rigid_pair_tensor.json"
CFG_SHA="20fb3893988022e40ccacc5056a3b2a2c995750b1ed80aa838bb278b817ff38f"
FREEZE_COMMIT="377943cccda8bc2976cae32bdabfdd6a4443f2f3"
PYTHON="/home/you/conda_env/env_isaaclab/bin/python"

def preflight(c):
    assert sha(CFG)==CFG_SHA,"PROTOCOL_CHANGED"
    assert call(["git","branch","--show-current"],cwd=ROOT)=="version/v0.1.81"
    assert subprocess.run(["git","merge-base","--is-ancestor",FREEZE_COMMIT,"HEAD"],cwd=ROOT).returncode==0
    assert not call(["git","status","--porcelain"],cwd=ROOT),"DIRTY_TREE"
    assert c["max_physics_episodes"]==6
    for name,h in c["source_unchanged_sha256"].items():
        assert sha(ROOT/name)==h,"ORIGINAL_CONTROLLER_SOURCE_DRIFT:"+name
    for name,h in c["new_source_baseline_sha256"].items():
        assert sha(ROOT/name)==h,"V80_IMMUTABLE_SOURCE_DRIFT:"+name
    assert sha(ROOT/"configs/v079_frozen_native_pair_matrix.json")==c["v079_frozen_config_sha256"]
    for name,h in c["original_v054_sha256"].items():
        assert sha(ORIG/name)==h,"V54_PHYSX_EVIDENCE_DRIFT:"+name
    for case,h in c["real_v079_episode_off_sha256"].items():
        assert sha(PREV/"reports"/f"v079_original_g1_native_pair_{case}_off.json")==h
    for t in c["trial_order"]:
        sc=json.loads((ROOT/t["source_config"]).read_text())
        for name,h in sc["frozen_files_sha256"].items():
            assert sha(ROOT/name)==h,"TRIAL_DEPENDENCY_DRIFT:"+name
        for k in ("template","reference","expert_plan"):
            assert sha(ROOT/sc[k])==sc[k+"_sha256"],"INPUT_CHANGED:"+k
    assert (ROOT/"experiments/runtime/onnxruntime/onnxruntime").is_dir()
    for name in ("tools/v081_run_tensor_strike.py","tools/v081_tensor_pair.py",
                 "tools/v081_tensor_batch.py","tests/test_v081_tensor_pair.py"):
        # Only committed bytes may run, even if an investigator modified the file.
        committed=subprocess.check_output(["git","show","HEAD:"+name],cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest()==sha(ROOT/name),"NEW_CODE_UNCOMMITTED_DRIFT:"+name

def once(c,t,mode):
    case=t["case"];prefix=ROOT/"reports"/f"{c['run_namespace']}_{case}_{mode}"
    out=Path(str(prefix)+".json");log=Path(str(prefix)+".log");audit=Path(str(prefix)+".audit.json")
    if any(p.exists() for p in (out,log,audit)):
        raise FileExistsError("WRITE_ONCE_COLLISION:"+str(prefix))
    free=resource_check(c)
    args=[PYTHON,str(ROOT/"tools/v081_run_tensor_strike.py"),
          "--config",str(ROOT/t["source_config"]),"--output",str(out),
          "--contact-diagnostics","--eval-seed",str(t["seed"])]
    if mode=="tensor":args.append("--tensor-wrist-hip-v081")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
             PYTHONPATH=":".join([str(ROOT/"tools"),
                str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    start=time.monotonic();timedout=False
    with log.open("x") as fp:
        fp.write("V081_CFG_SHA="+CFG_SHA+"\nV081_FREEZE="+FREEZE_COMMIT+"\n"+repr(args)+"\n")
        fp.flush()
        p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=fp,
                           stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=c["gpu_trial_timeout_seconds"])
        except subprocess.TimeoutExpired:
            timedout=True;os.killpg(p.pid,signal.SIGTERM)
            try:rc=p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);rc=p.wait(timeout=5)
            fp.write("\nV081_WATCHDOG_TERMINATED_OWN_CHILD\n")
    raw={};parse_err=None
    if out.exists():
        try:raw=json.loads(out.read_text())
        except Exception as e:parse_err=repr(e)
    certified,complete,reason=(validate(raw,log.read_text(errors="replace"),t)
                                  if raw else (False,False,"NO_JSON"))
    certified=certified and rc==0 and not timedout
    try:score=physically_scored(raw,wrapper_certified=True) if certified else {
        "hard_gate_pass":False,"failed_gates":["NO_CERTIFIED_ORIGINAL_G1"]}
    except Exception as e:score={"hard_gate_pass":False,"failed_gates":["AUDIT_EXCEPTION"],
                                  "audit_error":repr(e)}
    evidence=raw.get("physx_tensor_selfpair_v081",{})
    summary={k:evidence.get(k) for k in ("physics_frames_seen","sampled_frames",
        "raw_contact_counts","sampled_max_reported_impulse",
        "foot_ball_positive_control_observed","direct_filtered_rigid_pair_witness",
        "native_collider_pair_manifold_certified","scientific_conclusion","errors","overflow",
        "view_shapes","parent_collision_prim_inventory")}
    row=dict(case=case,mode=mode,rc=rc,timeout=timedout,
        original_200Hz_G1_certified=certified,complete_14s=complete,
        reason=reason,physics_rows=len(raw.get("traces",[[]])[0]) if raw else 0,
        strict_hard_gates=score,tensor_summary=summary,
        source_config=t["source_config"],seed=t["seed"],
        json_sha256=sha(out) if out.exists() else None,log_sha256=sha(log),
        cfg_sha256=CFG_SHA,parse_error=parse_err,elapsed_s=round(time.monotonic()-start,2),
        free_disk_gib_preflight=free)
    with audit.open("x") as f:json.dump(row,f,indent=2,ensure_ascii=False);f.write("\n")
    print("V081_G1",case,mode,"physical",certified,"rows",row["physics_rows"],
          "14s",complete,"hard_pass",score.get("hard_gate_pass"),
          "tensor",summary.get("raw_contact_counts"),"positive",
          summary.get("foot_ball_positive_control_observed"),flush=True)
    return row,raw

def compare(c,off,on,a,b):
    changed=[k for k in c["compare_physical_fields_exact"] if a.get(k)!=b.get(k)]
    parity=not changed and off["original_200Hz_G1_certified"] and on["original_200Hz_G1_certified"]
    pair={"case":off["case"],"exact_physical_parity":parity,"changed_physical_keys":changed,
       "baseline_sha":off["json_sha256"],"tensor_sha":on["json_sha256"],
       "same_strict_hard_gates":off["strict_hard_gates"].get("hard_gate_pass")==
                                  on["strict_hard_gates"].get("hard_gate_pass"),
       "tensor_positive_control":on["tensor_summary"]["foot_ball_positive_control_observed"],
       "tensor_filtered_pair_witness":on["tensor_summary"]["direct_filtered_rigid_pair_witness"],
       "native_collider_manifold_proven":False,
       "absence_interpretation":"UNKNOWN"},
    pair=pair[0]
    with (ROOT/"reports"/f"v081_{off['case']}_parity.audit.json").open("x") as f:
        json.dump(pair,f,indent=2);f.write("\n")
    print("V081_PARITY",off["case"],parity,"CHANGED",changed,
        "TENSOR_PAIR",pair["tensor_filtered_pair_witness"],flush=True)
    return pair

def main():
    c=json.loads(CFG.read_text());preflight(c);resource_check(c)
    lock=Path(c["gpu_lock"])
    if not lock.exists():raise RuntimeError("GPU_MUTEX_MISSING")
    rows=[];pairs=[]
    with lock.open("r+") as fp:
        try:fcntl.flock(fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GPU_MUTEX_ALREADY_LOCKED")
        for t in c["trial_order"]:
            baseline,a=once(c,t,"off");rows.append(baseline)
            if not baseline["original_200Hz_G1_certified"]:raise RuntimeError("BASELINE_UNSCORED_STOP")
            instrumented,b=once(c,t,"tensor");rows.append(instrumented)
            if not instrumented["original_200Hz_G1_certified"]:raise RuntimeError("TENSOR_UNSCORED_STOP")
            if instrumented["tensor_summary"]["errors"]:
                raise RuntimeError("PHYSX_TENSOR_OBSERVER_ERROR_STOP:"+str(instrumented["tensor_summary"]["errors"]))
            p=compare(c,baseline,instrumented,a,b);pairs.append(p)
            if not p["exact_physical_parity"]:raise RuntimeError("READ_ONLY_PARITY_VIOLATION_STOP")
            if not p["tensor_positive_control"]:
                raise RuntimeError("TENSOR_KNOWN_FOOTBALL_CONTACT_POSITIVE_CONTROL_FAILED")
    report=dict(version="v0.1.81",protocol_sha256=CFG_SHA,
      real_original_29dof_g1_episodes=len(rows),
      hard_pass=sum(r["strict_hard_gates"].get("hard_gate_pass",False) for r in rows),
      early_end=sum(not r["complete_14s"] for r in rows),
      exact_pair_parity=sum(p["exact_physical_parity"] for p in pairs),
      positive_tensor_rigidpair=sum(bool(p["tensor_filtered_pair_witness"]) for p in pairs),
      native_collider_pair_manifold_certified=0,
      independent_controls=len(pairs),runs=rows,pairs=pairs)
    with (ROOT/"reports/v081_original_g1_tensor_final.audit.json").open("x") as f:
        json.dump(report,f,indent=2,ensure_ascii=False);f.write("\n")
    print("V081_BATCH_DONE",json.dumps({k:v for k,v in report.items() if k not in ("runs","pairs")}),flush=True)

if __name__=="__main__":main()
