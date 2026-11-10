#!/usr/bin/env python3
"""Three original center-ball physical replay diagnostics, no changed policy.

One isolated GPU IsaacLab process at a time, writes immutable original
complete PhysX traces with --contact-diagnostics only. Source artifacts and
per-run CLI status are pinned; no in-run retuning or inferred contact labels.
"""
import json,hashlib,sys,subprocess,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/cmu_named_nonfoot_diagnostics_v054.json"
PROGRESS=ROOT/"reports/v054_contact_diag_progress.json"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def freeze(c):
 if c["version"]!="0.1.54" or len(c["original_three_center_policies"])!=3:
  raise ValueError("Wrong original v54 study")
 if not c["no_policy_or_ball_state_changes"] or not c["no_adaptive_retraining"]:
  raise ValueError("Must not modify previous G1 physical policy")
 for k,h in c["immutable_sha256"].items():
  if sha(ROOT/k)!=h:raise ValueError("Original frozen G1/PhysX source drift: "+k)
 return True
def run():
 c=json.loads(CONFIG.read_text());freeze(c)
 if "env_isaaclab" not in sys.executable:raise ValueError("Original Isaac Lab Python required")
 cases=list(c["original_three_center_policies"].items())
 if PROGRESS.exists():
  record=json.loads(PROGRESS.read_text())
  if record["protocol_sha256"]!=sha(CONFIG):raise ValueError("Progress source drift")
  history=record["runs"]
 else:history=[]
 for i,(name,spec) in enumerate(cases):
  out=ROOT/spec["instrumented_output"]
  if i<len(history):
   h=history[i]
   if h["name"]!=name or sha(out)!=h["sha256"]:raise ValueError("Cannot skip mismatched existing physical evidence")
   continue
  freeze(c)
  if out.exists():raise FileExistsError("Do not overwrite original contact diagnostics")
  log=out.with_suffix(".log")
  env=dict(os.environ,PYTHONPATH=":".join(["tools",str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
  cmd=[sys.executable,str(ROOT/"tools/run_placement_strike.py"),
       "--config",str(ROOT/spec["config"]),"--output",str(out),"--contact-diagnostics"]
  print("V054_INSTRUMENTED_PHYSX_BEGIN",i+1,"of",len(cases),name,flush=True)
  start=time.monotonic()
  with log.open("w") as f:
   process=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  if process.returncode!=0 or not out.exists():
   raise RuntimeError("Original physical process stopped: no policy edit or automatic retry")
  row=dict(name=name,sha256=sha(out),result=spec["instrumented_output"],
           bytes=out.stat().st_size,wall_seconds=round(time.monotonic()-start,2))
  history.append(row)
  PROGRESS.write_text(json.dumps(dict(version="0.1.54",protocol_sha256=sha(CONFIG),runs=history),indent=2)+"\n")
  print("V054_INSTRUMENTED_PHYSX_END",json.dumps(row),flush=True)
 print("V054_ALL_NAMED_CONTACT_DIAGNOSTICS_DONE",len(history),flush=True)
if __name__=="__main__":run()
