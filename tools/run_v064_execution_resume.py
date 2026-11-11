# v64 NON-PHYSICS safety fix: nullable baseline progress print only. Frozen protocol runner file untouched.
#!/usr/bin/env python3
"""Frozen 4 physical conditions x 3 controls = 12 serial original-G1 PhysX trials.

v32 / 120Hz CMU-sourced fitted 29DoF temporal phase / same human phase reversed.
Real human contact, elite power and GRF are not present: geometry proxy only.
Original PD, sphere, floor, physical terminations and recovery remain unchanged.
Every large JSON is local, write-once, and hashed. No repeated seed pass counting.
"""
from __future__ import annotations
import fcntl,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"configs/g1_human_phase_batch_v064.json"
LOCK=ROOT/"reports/v064_protocol_lock.json"
PROGRESS=ROOT/"reports/v064_batch_progress.json"
MUTEX=ROOT/"experiments/kinetickick_physics_gpu.lock"
PYTHON=Path("/home/you/conda_env/env_isaaclab/bin/python")
BACKEND=re.compile(r"Environment device\s*:\s*(cpu|cuda:\d+)")

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def guard_gpu():
 fs=os.statvfs(ROOT)
 available=fs.f_bavail*fs.f_frsize/1024**3
 if available<20:raise RuntimeError("Disk protective stop below 20GiB")
 s=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],
                  capture_output=True,text=True,timeout=15)
 if s.returncode or s.stdout.strip():raise RuntimeError("Other GPU compute process: "+s.stdout)
 return available

def config():
 c=json.loads(PROTOCOL.read_text())
 if c["version"]!="v0.1.64" or c["parent_sha"]!="d18ea81964a9a6b4e276c029b2bddd4e323268d4":
  raise ValueError("v64 scientific ancestry drift")
 if c.get("human_training_clips")!=["10_02"] or c.get("human_foot_ball_contact_truth") is not False:
  raise ValueError("Do not invent true impact supervision")
 if len(c["cases"])!=12 or len({x["id"] for x in c["cases"]})!=12:
  raise ValueError("12 originally registered runs expected")
 if set(c["policies"])!={"v32","human_ordered","human_reversed"}:
  raise ValueError("v64 policy control matrix drift")
 for name,h in c["source_sha256"].items():
  if sha(ROOT/name)!=h:raise RuntimeError("SOURCE_DRIFT "+name)
 for name,h in c["policy_sha256"].items():
  if sha(ROOT/name)!=h:raise RuntimeError("POLICY_DRIFT "+name)
 if set(x["seed"] for x in c["cases"])!={1201,6407,6419,6431}:
  raise ValueError("Only registered source-protected context seeds")
 for cse in c["cases"]:
  if cse["policy"] not in c["policies"] or not cse["output"].startswith("reports/v064_") or not cse["output"].endswith(".json"):
   raise ValueError("Unsafe physical result path")
  if cse["policy"]=="v32" and cse["seed"]==1201 and cse["perturb_reset"]:
   raise ValueError("Nominal baseline cannot have hidden reset")
 return c

def check_physics(case,raw,log,rc,c):
 from audit_v060_observer_factorial import assert_complete_physics_fields
 from wholebody_safety_interface_v055 import G1_JOINT_ORDER
 if rc or "V058_NATIVE_RETURN" not in log:raise ValueError("Missing clean original G1 PhysX result")
 backs=BACKEND.findall(log)
 if not backs or backs[0]!="cuda:0" or raw.get("v057_requested_physics_backend")!="cuda:0":
  raise ValueError("Wrong physics backend")
 if raw.get("root_fixed") is not False or raw.get("seed")!=case["seed"]:
  raise ValueError("Not original free-root G1 seeded state")
 if tuple(raw.get("joint_names",()))!=G1_JOINT_ORDER:raise ValueError("29DoF G1 ordering changed")
 if raw.get("policy_sha256")!=c["policy_sha256"][c["policies"][case["policy"]]]:
  raise ValueError("Frozen policy source SHA mismatch")
 trace=raw.get("traces",[[]])[0]
 assert_complete_physics_fields(trace)
 obs=raw.get("safe_body_v062")
 if not obs or obs.get("mode")!="safe_full" or len(obs.get("rows",[]))!=len(trace):
  raise ValueError("Full safe 200Hz G1 physical observer absent")
 if len(trace)<30:raise ValueError("No physical time series")
 for idx,(rawframe,frame) in enumerate(zip(trace,obs["rows"])):
  if (len(rawframe.get("joint_pos",[]))!=29 or
      len(frame.get("joint_pos_rad",[]))!=29 or
      max(abs(a-b) for a,b in zip(rawframe["joint_pos"],frame["joint_pos_rad"]))>1e-6 or
      max(abs(a-b) for a,b in zip(rawframe["joint_vel"],frame["joint_vel_rad_s"]))>1e-6 or
      abs(frame["state_aligned_time_s"]-rawframe["time_s"])>1e-7 or
      frame["state_alignment_source_index"]!=idx):
   raise ValueError("Real G1 200Hz q/dq/time alignment not certified")
 if type(raw.get("stride_shot_success")) is not bool:raise ValueError("Strict original physics verdict absent")
 if case["policy"]=="v32":
  if "human_phase_v064_summary" in raw:raise ValueError("Human candidate activated in baseline")
 else:
  summary=raw.get("human_phase_v064_summary")
  if not summary or summary["mode"]!=case["policy"] or summary["physical_original_29dof_joint_actuator_limits_unchanged"] is not True:
   raise ValueError("Original full-body human prior NOT actually active/verified")
  if summary["nonzero_20ms_control_steps"]<1:
   raise ValueError("Human learned geometric prior had zero measured physical actions")
  if summary["max_action_delta_rad"]>.0550001:
   raise ValueError("Original action deviations exceeded frozen range")
 return dict(status="VALID_G1",strict_pass=raw["stride_shot_success"],
  trace_rows=len(trace),physical_end_s=trace[-1]["time_s"],
  contact_time_s=raw["records"][0].get("contact_time_s"),
  termination=raw["termination_reasons"],
  reached_goal=raw["records"][0].get("goal"),
  measured_surface=raw["records"][0].get("measured_surface"),
  instep_fraction=raw["records"][0].get("requested_surface_impulse_fraction"),
  physically_recovered=raw["records"][0].get("recovered"),
  applied=raw.get("human_phase_v064_summary"))

def main():
 if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError("Pinned isaaclab environment required")
 current=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
 if current!="version/v0.1.64":raise RuntimeError("Cannot change another active version's experiments")
 if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip():
  raise RuntimeError("Frozen tracked code dirty before physics")
 c=config()
 proto_sha=sha(PROTOCOL)
 if PROGRESS.exists():
  state=json.loads(PROGRESS.read_text())
  if state["protocol_sha256"]!=proto_sha:raise ValueError("Previously registered protocol drift")
 else:state=dict(protocol_sha256=proto_sha,version="v0.1.64",runs=[])
 MUTEX.parent.mkdir(exist_ok=True,parents=True)
 with MUTEX.open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  guard_gpu()
  if LOCK.exists():
   if json.loads(LOCK.read_text())["protocol_sha256"]!=proto_sha:raise RuntimeError("Protocol lock drift")
  else:
   LOCK.write_text(json.dumps(dict(protocol_sha256=proto_sha,
     frozen_git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
     preregistered_before_first_real_G1_PhysX=True,
     cases=len(c["cases"])),indent=2)+"\n")
  if not PROGRESS.exists():PROGRESS.write_text(json.dumps(state,indent=2)+"\n")
  for idx,case in enumerate(c["cases"]):
   config();guard_gpu()
   dest=ROOT/case["output"];log=dest.with_suffix(".log")
   if idx<len(state["runs"]):
    old=state["runs"][idx]
    if (old["id"]!=case["id"] or not log.exists() or sha(log)!=old["log_sha256"] or
       dest.exists()!=bool(old.get("raw_sha256")) or
       dest.exists() and sha(dest)!=old["raw_sha256"]):
     raise RuntimeError("PREVIOUS_EVIDENCE_DRIFT: "+case["id"])
    print("PRESERVED",case["id"],old["status"],flush=True)
    continue
   if dest.exists() or log.exists():raise FileExistsError("Already-existing write-once PhysX evidence "+case["id"])
   command=[str(PYTHON),str(ROOT/"tools/run_placement_strike_v064.py"),
    "--config",str(ROOT/c["policies"][case["policy"]]),
    "--output",str(dest),"--eval-seed",str(case["seed"]),
    "--sim-device","cuda:0","--safe-body-v062","safe_full",
    "--ball-offset",*[str(x) for x in case["offset_xy"]],
    "--velocity-delta",str(case["velocity_delta_mps"])]
   if case["perturb_reset"]:command.append("--perturb-reset-v056")
   env=dict(os.environ,PYTHONPATH=":".join(["tools",str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),
    OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
   print("V064_BEGIN",idx+1,len(c["cases"]),case["id"],flush=True)
   started=time.monotonic()
   with log.open("x") as out:
    try:
     process=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=250)
     rc=process.returncode
    except subprocess.TimeoutExpired:
     rc=124
     out.write("\nV064_TIMEOUT_250_UNSCORABLE\n")
   logtext=log.read_text(errors="replace")
   row=dict(id=case["id"],case=case,return_code=rc,elapsed_wall_s=round(time.monotonic()-started,2),
      log_sha256=sha(log))
   if dest.exists():
    row["raw_sha256"]=sha(dest)
    try:row.update(check_physics(case,json.loads(dest.read_text()),logtext,rc,c))
    except Exception as e:row.update(status="INVALID_G1",reason=str(e))
   else:
    row.update(status="NO_G1_OUTPUT",reason="No physical G1 output: never score PASS/FAIL",
       last_console=logtext[-1000:])
   state["runs"].append(row)
   temp=PROGRESS.with_suffix(".tmp")
   temp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+"\n")
   os.replace(temp,PROGRESS)
   print("V064_END",row["id"],row["status"],row.get("strict_pass"),(row.get("applied") or {}).get("max_action_delta_rad"),flush=True)
 print("V064_COMPLETE",len(state["runs"]),sum(x["status"]=="VALID_G1" for x in state["runs"]),flush=True)
if __name__=="__main__":main()
