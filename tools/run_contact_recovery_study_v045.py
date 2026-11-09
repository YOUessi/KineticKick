#!/usr/bin/env python3
"""v45 one original-G1 PhysX shot per prespecified task/controller.

Sequence: strict zero-parity first (already audited), 2 known diagnostics,
then 6 NEW exact ball / velocity parameter pairs v41 vs v45. No case tuning,
no early stopping on scorer failure (all registered outcomes must be reported).
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/contact_recovery_governor_v045.json"

def digest(path):
 h=hashlib.sha256()
 with (ROOT/path).open("rb") as f:
  for p in iter(lambda:f.read(1<<20),b""):h.update(p)
 return h.hexdigest()

def run():
 study=json.loads((ROOT/STUDY).read_text())
 if study["version"]!="0.1.45" or study["total_prespecified_physx_runs"]!=15:
  raise ValueError("Different original study or trial budget")
 for path,expected in study["source_sha256"].items():
  if digest(path)!=expected:raise ValueError("Frozen source drift: "+path)
 zero=json.loads((ROOT/"reports/v045_governor_zero_center_audit.json").read_text())
 assert zero["zero_exact"]["original_actions_identical"]
 assert zero["zero_exact"]["original_shared_28_physical_channels_identical_every_5ms"]
 assert zero["zero_exact"]["zero_added_governor_control_steps"]==0
 options=[]
 for case in study["known_development_diagnostics"]:
  options.append(dict(case=case["id"],controller="governor_v45",runner=study["governor_runner"],
      config=study["governor_enabled_config"],offset=case["ball_offset_xy_m"],
      velocity=case["velocity_y_delta_mps"],output=case["output"],status="previously_observed_development"))
 for case in study["six_new_parameter_conditions"]:
  for ctrl in ("style_v41","governor_v45"):
   options.append(dict(case=case["id"],controller=ctrl,
      runner=study["baseline_runner"] if ctrl=="style_v41" else study["governor_runner"],
      config=study["original_baseline_config"] if ctrl=="style_v41" else study["governor_enabled_config"],
      offset=case["ball_offset_xy_m"],velocity=case["velocity_y_delta_mps"],
      output=case["outputs"][ctrl],status="exact_new_study_parameter_combination"))
 assert len(options)==14
 for t in options:
  if (ROOT/t["output"]).exists():
   raise FileExistsError("Never overwrite PhysX evidence, even after failure: "+t["output"])
 if "/env_isaaclab/" not in sys.executable:
  raise ValueError("Must use the source frozen IsaacLab environment")
 progress=ROOT/"reports/v045_physical_study_progress.json"
 if progress.exists():raise FileExistsError(progress)
 finished=[]
 for i,job in enumerate(options,1):
  for path,h in study["source_sha256"].items():
   if digest(path)!=h:raise ValueError("Source changed during original predeclared experiment: "+path)
  out=ROOT/job["output"]
  log=out.with_suffix(".log")
  cmd=[sys.executable,str(ROOT/job["runner"]),"--config",str(ROOT/job["config"]),
       "--ball-offset",str(job["offset"][0]),str(job["offset"][1]),
       "--velocity-delta",str(job["velocity"]),"--output",str(out)]
  print("V045_REAL_PHYSX_START",i,len(options),job["case"],job["controller"],
    job["offset"],job["velocity"],flush=True)
  start=time.monotonic()
  with log.open("w") as f:
   result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
  row=dict(index=i,case=job["case"],controller=job["controller"],
    physical_json=job["output"],cli_exit=int(result.returncode),
    seconds=round(time.monotonic()-start,3),
    bytes=out.stat().st_size if out.exists() else 0,
    sha256=digest(job["output"]) if out.exists() else None)
  finished.append(row)
  progress.write_text(json.dumps(dict(
      study_sha256=digest(STUDY),
      intended_additional_runs=len(options),
      completed=len(finished),trials=finished),indent=2)+"\n")
  print("V045_REAL_PHYSX_DONE",json.dumps(row),flush=True)
  if result.returncode!=0 or not out.is_file():
   raise RuntimeError("PhysX program execution failed; preserve output and stop, do NOT modify success thresholds or adjust controller")
 print("V045_FULL_PRESPECIFIED_PHYSX_FINISHED",len(finished),flush=True)
if __name__=="__main__":run()
