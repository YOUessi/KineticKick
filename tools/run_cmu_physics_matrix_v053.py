#!/usr/bin/env python3
"""Run entire frozen v53 three-controller x four-condition genuine G1 PhysX batch.

Runs one IsaacLab process at a time to avoid GPU OOM. Never overwrite original
results, never adapt controller gains after viewing outputs, and never turn a
failed kick into success because the ball crossed the goal plane.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/cmu_g1_physics_matrix_v053.json"
PROGRESS=ROOT/"reports/v053_physics_progress.json"
ORDER=("frozen_v32","cmu_10_02","cmu_10_05")

def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for part in iter(lambda:f.read(1<<20),b""):h.update(part)
 return h.hexdigest()

def check(c):
 assert c["version"]=="0.1.53" and c["n_conditions"]==4
 assert c["no_policy_retraining"] is True and c["no_adaptive_tuning"] is True
 assert list(c["policies"])==list(ORDER)
 for f,h in c["source_sha256"].items():
  if digest(ROOT/f)!=h:raise ValueError("Frozen original physical/CMU source drift: "+f)
 seen=set()
 for case in c["trials"]:
  xy=case["ball_offset_xy_m"];v=case["velocity_y_delta_mps"]
  t=(*xy,v)
  if (t in seen or len(xy)!=2 or max(map(abs,xy))>.15 or abs(v)>.30):
   raise ValueError("Invalid or duplicate original G1 physical task condition")
  seen.add(t)
  if list(case["outputs"])!=list(ORDER):raise ValueError("Missing predeclared results")
 return True

def run():
 c=json.loads(CONFIG.read_text());check(c)
 if "env_isaaclab" not in sys.executable:
  raise RuntimeError("Only original Tang Isaac Lab isolated Python may run the physical chain")
 tasks=[(i,case,slot) for i,case in enumerate(c["trials"]) for slot in ORDER]
 # Support resuming an interrupted run, never repeating a successfully saved
 # original physical artifact and never skipping a missing intermediate one.
 history=[]
 if PROGRESS.exists():
  saved=json.loads(PROGRESS.read_text())
  if saved["protocol_sha256"]!=digest(CONFIG):raise ValueError("Progress belongs to different frozen study")
  history=saved["completed"]
  for n,r in enumerate(history):
   i,case,slot=tasks[n]
   if (r["condition"]!=case["id"] or r["policy"]!=slot or
       r["sha256"]!=digest(ROOT/case["outputs"][slot])):
    raise ValueError("Cannot resume with mismatched, tampered, or reordered physical evidence")
  print("V053_RESUME_ALREADY_VERIFIED",len(history),"of",len(tasks),flush=True)
 for k,(i,case,slot) in enumerate(tasks):
  if k<len(history):continue
  check(c)
  dest=ROOT/case["outputs"][slot]
  if dest.exists():raise FileExistsError("Cannot overwrite unregistered physical evidence "+str(dest))
  cfg=ROOT/c["policies"][slot]["config"]
  log=dest.with_suffix(".log")
  cmd=[sys.executable, str(ROOT/"tools/run_placement_strike.py"),
       "--config",str(cfg),"--output",str(dest),"--ball-offset",
       *map(str,case["ball_offset_xy_m"]),"--velocity-delta",str(case["velocity_y_delta_mps"])]
  env=dict(os.environ,PYTHONPATH=":".join(["tools",str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),
           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
  started=time.monotonic()
  print("V053_PHYSX_BEGIN",k+1,"of",len(tasks),case["id"],slot,flush=True)
  with log.open("w") as output:
   proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
  rec=dict(condition=case["id"],policy=slot,result=case["outputs"][slot],
      wall_seconds=round(time.monotonic()-started,2),exit_code=proc.returncode,
      bytes=dest.stat().st_size if dest.exists() else 0,
      sha256=digest(dest) if dest.exists() else None)
  if proc.returncode!=0 or not dest.exists():
   print("V053_RUN_FAILED_STOP",json.dumps(rec),flush=True)
   raise RuntimeError("Original PhysX process failed; preserve full log and stop without auto retuning")
  history.append(rec)
  PROGRESS.write_text(json.dumps(dict(version="0.1.53",protocol_sha256=digest(CONFIG),
        planned=len(tasks),completed=history),indent=2)+"\n")
  print("V053_PHYSX_FINISH",json.dumps(rec),flush=True)
 print("V053_ALL_FROZEN_PHYSICS_COMPLETE",len(history),flush=True)
 return history

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--run",action="store_true")
 a=p.parse_args()
 c=json.loads(CONFIG.read_text());check(c)
 if a.run:run()
 else:print("V053_FROZEN_SOURCE_CHECK_OK")
