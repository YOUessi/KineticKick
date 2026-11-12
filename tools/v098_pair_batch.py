#!/usr/bin/env python3
"""Six real G1 200Hz PhysX center OFF/ON pairs, read-only G1 filtered tensor observer.

Frozen source SHA; global GPU flock, child-only bounded watchdog, source hashes
and strict all-common-physics-field parity, no overwrites. Never call filtered
tensor force a native collider manifold.
"""
from __future__ import annotations
import os,json,subprocess,hashlib,fcntl,signal,time,shutil
from pathlib import Path
from v083_run_physics_batch import certified_physics,sha,PYTHON
from audit_v067_hard_gates import physically_scored
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v098_pair_protocol.json"
MAIN=ROOT/"reports/v098_pair_final.audit.json"
FROZEN=("configs/v098_pair_protocol.json","tools/v098_pair_batch.py",
        "tools/v098_run_pair_strike.py","tools/v098_g1_tensor_pair.py",
        "docs/V098_PAIR_FREEZE.md","tests/test_v098_pair_protocol.py")
def git(*args):
 return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def resources(c):
 free=shutil.disk_usage(ROOT).free/1024**3
 if free<c["resources"]["min_free_gib"]:raise RuntimeError("DISK_LOW_20GiB")
 apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
           "--format=csv,noheader"],text=True).strip()
 if apps:raise RuntimeError("FOREIGN_CUDA_APPS:"+apps)
 return free
def preflight():
 if git("branch","--show-current")!="version/v0.1.98":raise RuntimeError("WRONG_BRANCH")
 if git("status","--porcelain","--untracked-files=no"):raise RuntimeError("TRACKED_DIRTY")
 if MAIN.exists():raise FileExistsError("WRITE_ONCE_PAIR_BATCH")
 for p in FROZEN:
  if (ROOT/p).read_bytes()!=subprocess.check_output(["git","show","HEAD:"+p],cwd=ROOT):
   raise RuntimeError("PAIR_PROTOCOL_NOT_SHA_FROZEN:"+p)
 c=json.loads(CFG.read_text())
 if c["parent_frozen_physics_sha"] not in git("log","--format=%H"):raise RuntimeError("PARENT_SHA_NOT_ANCESTOR")
 expected=[(control,mode) for control in ("v32","cmu02","cmu05") for mode in (False,True)]
 if [(z["control"],z["observer"]) for z in c["pair_matrix"]]!=expected:raise RuntimeError("CHANGED_MATRIX")
 for path in c["original_control_configs"].values():
  conf=json.loads((ROOT/path).read_text())
  for name,expected_sha in conf["frozen_files_sha256"].items():
   if sha(ROOT/name)!=expected_sha:raise RuntimeError("UPSTREAM_SHA_DRIFT:"+name)
  for k in ("template","reference","expert_plan"):
   if sha(ROOT/conf[k])!=conf[k+"_sha256"]:raise RuntimeError("FROZEN_REFERENCE_DRIFT")
 for d in ("third_party/HumanoidSoccer/source/whole_body_tracking/soccer","experiments/runtime/onnxruntime/onnxruntime"):
  if not (ROOT/d).is_dir():raise RuntimeError("UNAVAILABLE_ORIGINAL_UPSTREAM:"+d)
 resources(c)
 c["_frozen"]=git("rev-parse","HEAD")
 c["_protocol_sha"]=sha(CFG)
 return c
def run(c,trial):
 name="v098_pair_"+trial["name"]
 rawpath=ROOT/"reports"/(name+".json")
 logfile=ROOT/"reports"/(name+".log")
 auditfile=ROOT/"reports"/(name+".audit.json")
 if any(p.exists() or p.is_symlink() for p in (rawpath,logfile,auditfile)):
  raise FileExistsError("WRITE_ONCE:"+name)
 free=resources(c)
 args=[PYTHON,str(ROOT/"tools/v098_run_pair_strike.py"),
       "--config",str(ROOT/c["original_control_configs"][trial["control"]]),
       "--output",str(rawpath),"--ball-offset","0.0","0.0",
       "--velocity-delta","0.0","--eval-seed","1201","--contact-diagnostics"]
 if trial["observer"]:args.append("--tensor-pair-v098")
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
          PYTHONPATH=":".join((str(ROOT/"tools"),str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"))),
          OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
 started=time.monotonic()
 with logfile.open("x") as f:
  f.write("FROZEN_SHA="+c["_frozen"]+"\nPROTOCOL_SHA="+c["_protocol_sha"]+"\nCMD="+repr(args)+"\n");f.flush()
  child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  timeout=False
  try:rc=child.wait(timeout=c["resources"]["per_child_timeout_s"])
  except subprocess.TimeoutExpired:
   timeout=True
   os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=5)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=5)
   f.write("\nOWN_CHILD_ONLY_TIMEOUT\n")
 raw={}
 if rawpath.is_file():
  try:raw=json.loads(rawpath.read_text())
  except Exception:pass
 valid,reason=certified_physics(raw,logfile.read_text(errors="replace"),
      {"seed":1201,"ball_offset_xy":[0.,0.],"velocity_delta":0.,"name":trial["name"]},rc,timeout)
 g={"hard_gate_pass":False,"failed_gates":["UNSCORED"]}
 if valid:
  try:g=physically_scored(raw,wrapper_certified=True)
  except Exception as e:g={"hard_gate_pass":False,"failed_gates":["AUDIT_ERROR"],"error":repr(e)}
 pair=raw.get("g1_tensor_pair_v098")
 row=dict(name=trial["name"],control=trial["control"],observer=trial["observer"],
          elapsed_s=round(time.monotonic()-started,3),rc=rc,timeout=timeout,
          certified_original_G1_cuda_200hz=valid,certification=reason,
          raw_sha256=sha(rawpath) if rawpath.is_file() else None,
          log_sha256=sha(logfile),true_raw_trace_rows=len(raw.get("traces",[[]])[0]),
          disk_free_pre_gib=round(free,3),hard_gate=g,
          observer_present=pair is not None,pair_observation=pair)
 with auditfile.open("x") as f:json.dump(row,f,indent=2);f.write("\n")
 print("PAIR_TRIAL",row["name"],valid,row["true_raw_trace_rows"],g["hard_gate_pass"],
       "tensor",pair and {k:(v["nonzero_force_frames"],v["peak_force_N"],v["status"]) for k,v in pair["pairs"].items()},flush=True)
 return row
def _max_abs(a,b):
 """No omission of shared primitive physical fields, exact equality required."""
 if isinstance(a,dict) and isinstance(b,dict):
  if set(a)!=set(b):return float("inf")
  return max((_max_abs(a[k],b[k]) for k in a),default=0.)
 if isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):return float("inf")
  return max((_max_abs(x,y) for x,y in zip(a,b)),default=0.)
 if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
  return abs(float(a)-float(b))
 return 0. if a==b else float("inf")
def parity(c,off,on):
 r0=json.loads((ROOT/"reports"/("v098_pair_"+off["name"]+".json")).read_text())
 r1=json.loads((ROOT/"reports"/("v098_pair_"+on["name"]+".json")).read_text())
 a,b=r0["traces"][0],r1["traces"][0]
 # All shared physical 200Hz trace keys, including q, dq, root, COM, support,
 # raw contacts, torque, termination and ball, not just a selected subset.
 maxdelta=_max_abs(a,b)
 records_delta=_max_abs(r0.get("records"),r1.get("records"))
 fields=sorted(set(a[0])&set(b[0])) if a and b else []
 valid=(maxdelta==0 and records_delta==0 and fields
    and off["certified_original_G1_cuda_200hz"] and on["certified_original_G1_cuda_200hz"])
 return dict(control=off["control"],identical_all_shared_trace_fields=bool(valid),
       max_absolute_all_physical_trace_difference=maxdelta,
       max_absolute_original_shot_record_difference=records_delta,
       compared_true_raw_rows=len(a),trace_fields=fields,
       native_actor_collider_manifold_certified=False)
def main():
 c=preflight()
 rows=[];pairs=[]
 lock=Path(c["resources"]["gpu_lock"])
 with lock.open("r+") as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError("GLOBAL_GPU_LOCK_BUSY")
  for i,t in enumerate(c["pair_matrix"]):
   row=run(c,t);rows.append(row)
   if not row["certified_original_G1_cuda_200hz"]:break
   if i%2==1:
    match=parity(c,rows[-2],rows[-1]);pairs.append(match)
    print("PAIR_PARITY",match["control"],match["identical_all_shared_trace_fields"],
          match["max_absolute_all_physical_trace_difference"],flush=True)
    if not match["identical_all_shared_trace_fields"]:break
 result=dict(version="v0.1.98",frozen_commit=c["_frozen"],protocol_sha256=c["_protocol_sha"],
   completed_certified_original_G1_cuda=sum(x["certified_original_G1_cuda_200hz"] for x in rows),
   strict_pass=sum(x["certified_original_G1_cuda_200hz"] and x["hard_gate"]["hard_gate_pass"] for x in rows),
   no_drift_pair_count=sum(x["identical_all_shared_trace_fields"] for x in pairs),
   no_drift_pairs=pairs,not_executed=[x["name"] for x in c["pair_matrix"][len(rows):]],
   native_wrist_hip_manifold_verified=False,eligible_new_learned_policy_trials=0,runs=rows)
 with MAIN.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
 print("PAIR_FINAL",json.dumps({k:v for k,v in result.items() if k not in ("runs","no_drift_pairs")}),flush=True)
if __name__=="__main__":main()
