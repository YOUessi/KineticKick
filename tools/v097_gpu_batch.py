#!/usr/bin/env python3
"""v97 frozen untouched-controller original G1 GPU batch: 4 write-once scenes."""
from __future__ import annotations
import fcntl,hashlib,json,os,subprocess,signal,time,shutil
from pathlib import Path
from v083_run_physics_batch import certified_physics,sha,PYTHON
from audit_v067_hard_gates import physically_scored

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/v097_frozen_matrix.json"
OUTPUT=ROOT/"reports/v097_g1_physics_final.audit.json"
FROZEN=("configs/v097_frozen_matrix.json","docs/V097_FROZEN_PROTOCOL.md",
        "tools/v097_gpu_batch.py","tests/test_v097_protocol.py")
def git(*args):
 return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def resources(settings):
 free=shutil.disk_usage(ROOT).free/1024**3
 if free<settings["min_free_gib"]:raise RuntimeError("DISK_BELOW_20GIB")
 gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True).strip()
 if gpu:raise RuntimeError("GPU_BUSY:"+gpu)
 return round(free,3)
def preflight():
 if git("branch","--show-current")!="version/v0.1.91":raise RuntimeError("WRONG_VERSION_BRANCH")
 if git("status","--porcelain","--untracked-files=no"):raise RuntimeError("TRACKED_DIRTY")
 for f in FROZEN:
  if (ROOT/f).read_bytes()!=subprocess.check_output(["git","show","HEAD:"+f],cwd=ROOT):
   raise RuntimeError("UNFROZEN_EXECUTOR_OR_PROTOCOL:"+f)
 if OUTPUT.exists():raise FileExistsError("WRITE_ONCE_MAIN_AUDIT")
 c=json.loads(CFG.read_text())
 if c["version"]!="v0.1.91" or c["parent_commit"] not in git("log","--format=%H"):
  raise RuntimeError("VERSION_SOURCE_DRIFT")
 if len(c["cases"])!=6 or len(set(t["name"] for t in c["cases"]))!=4:raise RuntimeError("MATRIX_CHANGED")
 if c["causal_contract"]["new_pre_registered_scenes"]!=["G","H"]:raise RuntimeError("PRE_REGISTERED_HOLDOUT_CHANGED")
 for specfile in c["controls"].values():
  conf=json.loads((ROOT/specfile).read_text())
  for p,expected in conf["frozen_files_sha256"].items():
   if not (ROOT/p).is_file() or sha(ROOT/p)!=expected:raise RuntimeError("SOURCE_SHA_DRIFT:"+p)
  for part in ("template","reference","expert_plan"):
   if sha(ROOT/conf[part])!=conf[part+"_sha256"]:raise RuntimeError("ASSET_SHA_DRIFT:"+part)
 for p in ("third_party/HumanoidSoccer/source/whole_body_tracking/soccer","experiments/runtime/onnxruntime/onnxruntime"):
  if not (ROOT/p).is_dir():raise RuntimeError("UPSTREAM_ASSET_UNAVAILABLE:"+p)
 c["_frozen_commit"]=git("rev-parse","HEAD")
 c["_protocol_sha256"]=sha(CFG)
 resources(c["settings"])
 return c
def trial(c,t):
 name="v097_"+t["name"]
 out=ROOT/"reports"/(name+".json")
 log=ROOT/"reports"/(name+".log")
 side=ROOT/"reports"/(name+".audit.json")
 if any(p.exists() or p.is_symlink() for p in (out,log,side)):raise FileExistsError("WRITE_ONCE:"+name)
 space=resources(c["settings"])
 args=[PYTHON,str(ROOT/"tools/v080_run_poll_strike.py"),"--config",str(ROOT/c["controls"][t["control"]]),
       "--output",str(out),"--ball-offset",*[str(v) for v in t["ball_offset_xy"]],
       "--velocity-delta",str(t["velocity_delta"]),"--eval-seed",str(t["seed"]),"--contact-diagnostics"]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
          PYTHONPATH=":".join((str(ROOT/"tools"),str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking"))),
          OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
 start=time.monotonic()
 with log.open("x") as f:
  f.write("V097_FROZEN_COMMIT="+c["_frozen_commit"]+"\nV097_PROTOCOL_SHA="+c["_protocol_sha256"]+"\nCMD="+repr(args)+"\n")
  f.flush()
  proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  timeout=False
  try:rc=proc.wait(timeout=c["settings"]["timeout_s"])
  except subprocess.TimeoutExpired:
   timeout=True
   os.killpg(proc.pid,signal.SIGTERM)
   try:rc=proc.wait(timeout=5)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait(timeout=5)
   f.write("\nV097_CHILD_ONLY_WATCHDOG_TERMINATION\n")
 row={"name":t["name"],"scene":t["scene"],"control":t["control"],"seed":t["seed"],
      "elapsed_s":round(time.monotonic()-start,3),"rc":rc,"watchdog":timeout,
      "disk_free_preflight_gib":space,"output_exists":out.exists(),
      "raw_sha256":sha(out) if out.exists() else None,
      "log_sha256":sha(log)}
 raw={}
 if out.exists():
  try:raw=json.loads(out.read_text())
  except Exception as e:row["decode_error"]=repr(e)
 valid,reason=certified_physics(raw,log.read_text(errors="replace"),t,rc,timeout)
 row.update(true_original_200Hz_G1_CUDA_PhysX=valid,validation=reason,
            true_physics_frames=len(raw.get("traces",[[]])[0]) if raw else 0)
 if valid:
  try:row["gates"]=physically_scored(raw,wrapper_certified=True)
  except Exception as e:row["gates"]={"hard_gate_pass":False,"failed_gates":["SCORER_EXCEPTION"],"error":repr(e)}
 else:row["gates"]={"hard_gate_pass":False,"failed_gates":["UNSCORED_PHYSICS"]}
 with side.open("x") as f:json.dump(row,f,indent=2);f.write("\n")
 print("V097_TRIAL",row["name"],"VALID",valid,"FRAMES",row["true_physics_frames"],"STRICT",row["gates"]["hard_gate_pass"],"REASON",row["gates"]["failed_gates"],flush=True)
 return row
def main():
 c=preflight()
 rows=[]
 lock=Path(c["settings"]["gpu_lock"])
 with lock.open("r+") as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError("OTHER_PHYSX_GPU_BATCH_HOLDS_MUTEX")
  for t in c["cases"]:
   resources(c["settings"])
   row=trial(c,t)
   rows.append(row)
   if not row["true_original_200Hz_G1_CUDA_PhysX"]:break
 report={"version":"v0.1.91","frozen_commit":c["_frozen_commit"],
         "protocol_sha256":c["_protocol_sha256"],
         "real_original_G1_PhysX_episodes":sum(z["true_original_200Hz_G1_CUDA_PhysX"] for z in rows),
         "strict_pass":sum(z["true_original_200Hz_G1_CUDA_PhysX"] and z["gates"]["hard_gate_pass"] for z in rows),
         "strict_fail":sum(z["true_original_200Hz_G1_CUDA_PhysX"] and not z["gates"]["hard_gate_pass"] for z in rows),
         "unscored_gpu_attempts":sum(not z["true_original_200Hz_G1_CUDA_PhysX"] for z in rows),
         "not_executed":[t["name"] for t in c["cases"][len(rows):]],
         "actuation":"UNCHANGED_FROZEN_v32_CMU02_CMU05","human_true_impulse_learned":False,
         "direct_wrist_hip_manifold_certified":False,"runs":rows}
 with OUTPUT.open("x") as f:json.dump(report,f,indent=2);f.write("\n")
 print("V097_DONE",json.dumps({k:v for k,v in report.items() if k!="runs"}),flush=True)
if __name__=="__main__":main()
