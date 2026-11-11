#!/usr/bin/env python3
"""Frozen six-shot original G1 off/on native poll parity, original controller unchanged."""
from __future__ import annotations
import fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ORIG=Path("/home/you/projects/kinetickick")
PREV=Path("/home/you/projects/kinetickick-wt-v079")
CFG=ROOT/"configs/v080_frozen_native_poll_matrix.json"
CFG_SHA="2898b8aca9c168e87a07967d7ccfce14448ffbf693e1bd3265e8bbf7327e978f"
FREEZE_COMMIT="2301a4b93bf21d7a9cdd9456cc95b3f02593dca2"
PYTHON="/home/you/conda_env/env_isaaclab/bin/python"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def call(args,**kw):
    return subprocess.check_output(args,text=True,**kw).strip()
def preflight(cfg):
    assert sha(CFG)==CFG_SHA,"PROTOCOL_CHANGED"
    assert call(["git","branch","--show-current"],cwd=ROOT)=="version/v0.1.80"
    assert subprocess.run(["git","merge-base","--is-ancestor",FREEZE_COMMIT,"HEAD"],cwd=ROOT).returncode==0
    assert not call(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT),"DIRTY_TRACKED_SOURCE"
    assert call(["git","rev-parse","HEAD"],cwd=ROOT)!=cfg["parent_sha"]
    for name,h in cfg["source_unchanged_sha256"].items():
        assert sha(ROOT/name)==h,"ORIGINAL_CONTROLLER_SOURCE_DRIFT:"+name
    assert sha(ROOT/"configs/v079_frozen_native_pair_matrix.json")==cfg["v079_frozen_config_sha256"]
    for case,h in cfg["real_v079_episode_off_sha256"].items():
        assert sha(PREV/"reports"/f"v079_original_g1_native_pair_{case}_off.json")==h
    for name,h in cfg["original_v054_sha256"].items():
        assert sha(ORIG/name)==h,"OLD_PHYSX_EVIDENCE_CHANGED:"+name
    for t in cfg["trial_order"]:
        sc=json.loads((ROOT/t["source_config"]).read_text())
        for name,expected in sc["frozen_files_sha256"].items():
            assert sha(ROOT/name)==expected,"TRIAL_DEPENDENCY_DRIFT:"+name
        for key in ("template","reference","expert_plan"):
            assert sha(ROOT/sc[key])==sc[key+"_sha256"],"INPUT_CHANGED:"+key
    # Original controller can be run from an isolated frozen copy only:
    import difflib
    before=(ROOT/"tools/v076_run_placement_strike.py").read_text().splitlines()
    after=(ROOT/"tools/v080_run_poll_strike.py").read_text().splitlines()
    added=[l[1:] for l in difflib.ndiff(before,after) if l.startswith("+ ")]
    removed=[l[1:] for l in difflib.ndiff(before,after) if l.startswith("- ")]
    assert not removed and len(added)==7,("UNREGISTERED_CONTROLLER_DIFF",removed,added)
    assert all("v080" in x or "self.pair_poll_v080" in x or
               "ExistingNativeContactPoll" in x or
               "from v080_contact_poll" in x for x in added),added
    assert (ROOT/"experiments/runtime/onnxruntime/onnxruntime").is_dir()
def resource_check(cfg):
    v=os.statvfs(ROOT);free=v.f_bavail*v.f_frsize/(1024**3)
    if free<cfg["minimum_free_gib"]:raise RuntimeError("DISK_BELOW_20GIB")
    gpu=call(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"])
    if gpu:raise RuntimeError("OTHER_CUDA_GPU_OWNER:"+gpu)
    return round(free,2)
def validate(raw,log,t):
    rows=raw.get("traces",[[]])[0]
    if (raw.get("seed")!=t["seed"] or raw.get("root_fixed") is not False
        or len(raw.get("joint_names",[]))!=29 or len(rows)<100
        or not raw.get("records") or not raw.get("sequence_records")
        or "Environment device    : cuda:0" not in log):
        return False,False,"UNCERTIFIED_ORIGINAL_CUDA_G1"
    if any(len(row.get("joint_vel",[]))!=29 or abs(float(row.get("physics_dt_s",0))-0.005)>1e-6 for row in rows):
        return False,False,"INVALID_ORIGINAL_200HZ_29QDOT"
    final=float(rows[-1].get("time_s",0))
    if final<0.5 or final>14.005:return False,False,"INVALID_TIME"
    complete=final>=13.995
    return True,complete,"COMPLETE_14S" if complete else "TRUE_EARLY_TERMINATION"
def once(cfg,t,mode):
    case=t["case"]
    prefix=ROOT/"reports"/f"{cfg['run_namespace']}_{case}_{mode}"
    p=Path(str(prefix)+".json");log=Path(str(prefix)+".log");audit=Path(str(prefix)+".audit.json")
    if any(x.exists() for x in (p,log,audit)):
        raise FileExistsError("WRITE_ONCE_COLLISION:"+str(prefix))
    free=resource_check(cfg)
    argv=[PYTHON,str(ROOT/"tools/v080_run_poll_strike.py"),
          "--config",str(ROOT/t["source_config"]),"--output",str(p),
          "--contact-diagnostics","--eval-seed",str(t["seed"])]
    if mode=="poll":argv+=["--native-readonly-poll-v080"]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
             PYTHONPATH=":".join([str(ROOT/"tools"),
                    str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    start=time.monotonic();timeout=False
    with log.open("x") as fp:
        for key,val in (("PROTOCOL_SHA",CFG_SHA),("PROTOCOL_COMMIT",FREEZE_COMMIT),
                        ("RUN_ARGV",argv)):
            fp.write(f"V080_{key}={val}\n")
        fp.flush()
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=fp,
                               stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=child.wait(timeout=cfg["gpu_trial_timeout_seconds"])
        except subprocess.TimeoutExpired:
            timeout=True
            os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                rc=child.wait(timeout=5)
            fp.write("\nV080_OWN_CHILD_WATCHDOG_TERMINATED\n")
    raw={};parse_err=None
    if p.exists():
        try:raw=json.loads(p.read_text())
        except Exception as e:parse_err=repr(e)
    certified,complete,reason=validate(raw,log.read_text(errors="replace"),t) if raw else (False,False,"NO_JSON")
    certified=certified and not timeout and rc==0
    from audit_v067_hard_gates import physically_scored
    if certified:
        try:score=physically_scored(raw,wrapper_certified=True)
        except Exception as exc:
            score={"hard_gate_pass":False,"failed_gates":["AUDIT_EXCEPTION"],
                   "audit_error":repr(exc)}
    else:score={"hard_gate_pass":False,"failed_gates":["NO_CERTIFIED_G1_PHYSICS"]}
    observer=raw.get("physx_native_poll_v080",{})
    row=dict(case=case,mode=mode,rc=rc,watchdog=timeout,
             source_config=t["source_config"],seed=t["seed"],
             certified_original_200hz_G1=certified,complete_14s=complete,
             certification_reason=reason,physics_rows=len(raw.get("traces",[[]])[0]) if raw else 0,
             elapsed_s=round(time.monotonic()-start,2),disk_free_gib_preflight=free,
             json_sha256=sha(p) if p.exists() else None,log_sha256=sha(log),
             protocol_sha256=CFG_SHA,parse_error=parse_err,
             strict_hard_gates=score,native_poll_summary={
               key:observer.get(key) for key in (
                 "poll_calls","native_headers","reported_native_contact_points",
                 "events_with_pair_witness","wrist_hip_direct_contact_certified",
                 "dropped_events","errors","times_polled_first_last",
                 "negative_interpretation")},
             callback_semantics="zero native events means UNKNOWN, not no collision")
    with audit.open("x") as f:json.dump(row,f,indent=2,ensure_ascii=False);f.write("\n")
    print("V080_REAL_G1",case,mode,"rc",rc,"real",certified,"14s",complete,
          "pass",score.get("hard_gate_pass"),"headers",observer.get("native_headers"),
          "poll_calls",observer.get("poll_calls"),"sha",row["json_sha256"],flush=True)
    return row,raw
def compare(cfg,case,off,on,a,b):
    differences=[key for key in cfg["physical_parity_keys"] if a.get(key)!=b.get(key)]
    ispair=not differences and off["certified_original_200hz_G1"] and on["certified_original_200hz_G1"]
    pair_event=on["native_poll_summary"].get("wrist_hip_direct_contact_certified") is True
    row={"case":case,"original_29dof_200hz_physics_parity":ispair,
         "differing_physics_fields":differences,
         "off_json_sha256":off["json_sha256"],"on_json_sha256":on["json_sha256"],
         "off_strict_hard_pass":off["strict_hard_gates"]["hard_gate_pass"],
         "on_strict_hard_pass":on["strict_hard_gates"]["hard_gate_pass"],
         "early_termination_does_not_count_as_success":not off["complete_14s"] or not on["complete_14s"],
         "direct_wrist_hip_contact_certified":bool(ispair and pair_event),
         "negative_interpretation":"UNKNOWN" if not pair_event else "NATIVE_PAIR_WITNESS",
         "off_termination_reasons":a.get("termination_reasons"),
         "on_termination_reasons":b.get("termination_reasons")}
    with (ROOT/"reports"/f"v080_original_g1_poll_pair_{case}_parity.audit.json").open("x") as f:
        json.dump(row,f,indent=2,ensure_ascii=False);f.write("\n")
    print("V080_PARITY",case,ispair,"DIFF",differences,"CERT_PAIR",row["direct_wrist_hip_contact_certified"],flush=True)
    return row
def main():
    cfg=json.loads(CFG.read_text());preflight(cfg);resource_check(cfg)
    lock=Path(cfg["gpu_lock"])
    if not lock.exists():raise RuntimeError("GPU_MUTEX_MISSING")
    results=[];pairs=[]
    with lock.open("r+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GPU_MUTEX_ALREADY_LOCKED")
        for t in cfg["trial_order"]:
            off,a=once(cfg,t,"off");results.append(off)
            if not off["certified_original_200hz_G1"]:
                raise RuntimeError("UNSCORED_BASELINE_STOP:"+t["case"])
            on,b=once(cfg,t,"poll");results.append(on)
            if not on["certified_original_200hz_G1"]:
                raise RuntimeError("UNSCORED_POLL_STOP:"+t["case"])
            cp=compare(cfg,t["case"],off,on,a,b);pairs.append(cp)
            if not cp["original_29dof_200hz_physics_parity"]:
                raise RuntimeError("READ_ONLY_PARITY_VIOLATED_STOP:"+t["case"])
    final={"version":"v0.1.80","frozen_config_sha256":CFG_SHA,
           "real_original_g1_episodes":len(results),
           "complete_14s_episodes":sum(x["complete_14s"] for x in results),
           "strict_hard_pass_episodes":sum(x["strict_hard_gates"]["hard_gate_pass"] for x in results),
           "native_poll_pairs_exact_parity":sum(x["original_29dof_200hz_physics_parity"] for x in pairs),
           "collider_pair_manifold_proven":sum(x["direct_wrist_hip_contact_certified"] for x in pairs),
           "distinct_control_configurations":len(pairs),
           "caution":"Paired repeats are not independent seeds or generalization evaluations",
           "runs":results,"pairs":pairs}
    with (ROOT/"reports/v080_original_g1_final.audit.json").open("x") as f:
        json.dump(final,f,indent=2,ensure_ascii=False);f.write("\n")
    print("V080_FROZEN_BATCH_DONE",json.dumps({k:v for k,v in final.items() if k not in ("runs","pairs")}),flush=True)
if __name__=="__main__":main()
