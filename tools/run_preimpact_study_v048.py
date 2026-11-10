#!/usr/bin/env python3
"""v48: one frozen pre-contact 29DoF learned actor vs original v32 G1.

Only do real 14s G1 free-root Isaac PhysX with unchanged 29 joint velocity/
effort limits, legal right instep ball force, full soccer ball goal and recovery.
The zero-disabled case must first match original v32 actual physical stream.
Six exact test ball/initial-velocity triples declared before any PPO training.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/learned_preimpact_wholebody_study_v048"
STUDY_JSON=STUDY+".json"
def sha(path):
 h=hashlib.sha256()
 with (ROOT/path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def preflight(s):
 if s["version"]!="0.1.48" or len(s["planned_6_new_exact_soccer_conditions"])!=6:
  raise ValueError("Only the pre-registered true G1 v48 study")
 for f,h in s["source_sha256"].items():
  if sha(f)!=h:raise ValueError("Frozen original PhysX source drift: "+f)
 result=ROOT/s["training_spec"]["output"]/"result.json"
 if not result.exists():raise FileNotFoundError("One-shot preimpact PPO training has not completed")
 r=json.loads(result.read_text())
 ckpt=s["training_spec"]["output"]+"/last_model.pt"
 if r["updates"]<1 or r["checkpoint_sha256"]!=sha(ckpt):
  raise ValueError("Trained 29DoF preimpact actor checkpoint is missing/drifted")
 return ckpt,r

def prepare_eval():
 p=ROOT/STUDY_JSON
 study=json.loads(p.read_text())
 ckpt,result=preflight(study)
 path=ROOT/study["candidate_evaluation_config"]
 if path.exists():raise FileExistsError("Do not overwrite frozen G1 test config")
 training=json.loads((ROOT/study["training_config"]).read_text())
 assert training["preimpact_wholebody_v048"]["checkpoint"] is None
 training["preimpact_wholebody_v048"]["checkpoint"]=ckpt
 training["preimpact_wholebody_v048"]["checkpoint_sha256"]=sha(ckpt)
 training["v048_training"]=dict(training["v048_training"],
    frozen_finished_updates=result["updates"],
    final_checkpoint_sha256=sha(ckpt),
    no_policy_reselection_using_eval_tasks=True,
    eval_started_before_this_write=False)
 training["status"]="research_only_sha_locked_preimpact_actor_14s_original_PhysX_evaluation"
 path.write_text(json.dumps(training,ensure_ascii=False,indent=2)+"\n")
 original_bytes=p.read_bytes()
 previous=ROOT/"experiments/preimpact_wholebody_v048/study_initial_before_trained_actor_metadata.json"
 if previous.exists():raise FileExistsError("Preserve protocol history once")
 previous.parent.mkdir(parents=True,exist_ok=True)
 previous.write_bytes(original_bytes)
 study["post_training_final_only_frozen"]={
    "initial_preregister_sha256":hashlib.sha256(original_bytes).hexdigest(),
    "original_preserved":str(previous.relative_to(ROOT)),
    "final_checkpoint_sha256":sha(ckpt),
    "final_PPO_updates":result["updates"],
    "eval_config_sha256":sha(study["candidate_evaluation_config"]),
    "runner_sha256":sha("tools/run_preimpact_study_v048.py"),
    "physx_auditor_sha256":sha("tools/audit_preimpact_wholebody_v048.py"),
    "no_eval_trial_run_before_model_lock":True}
 p.write_text(json.dumps(study,ensure_ascii=False,indent=2)+"\n")
 print("V048_EVAL_CONFIG_FROZEN",sha(STUDY_JSON),
     "PPO",sha(ckpt),"updates",result["updates"],
     "paired_trials",len(study["planned_6_new_exact_soccer_conditions"]),flush=True)

def evaluate():
 study=json.loads((ROOT/STUDY_JSON).read_text())
 ckpt,result=preflight(study)
 now=study.get("post_training_final_only_frozen")
 if not now or now["final_checkpoint_sha256"]!=sha(ckpt) or now["eval_config_sha256"]!=sha(study["candidate_evaluation_config"]):
  raise ValueError("New learned policy must be frozen before real PhysX testing")
 if sha("tools/run_preimpact_study_v048.py")!=now["runner_sha256"] or sha("tools/audit_preimpact_wholebody_v048.py")!=now["physx_auditor_sha256"]:
  raise ValueError("Original evaluator/auditor source drift")
 if "/env_isaaclab/" not in sys.executable:raise ValueError("Original IsaacLab interpreter required")
 if not (ROOT/study["zero_result"]).exists():
  raise ValueError("Run original v48 zero physics before enabling candidate")
 if not (ROOT/study["zero_audit"]).exists():
  raise ValueError("Independent exact-zero auditor must complete")
 zero=json.loads((ROOT/study["zero_audit"]).read_text())
 if not zero["zero_parity_pass"]:
  raise ValueError("Zero trial not exactly original G1 physical results")
 jobs=[]
 for c in study["planned_6_new_exact_soccer_conditions"]:
  for who in ("original_v32","learned_preimpact_v48"):
   jobs.append(dict(case=c["id"],controller=who,
      config=study["original_v32_baseline_config"] if who=="original_v32" else study["candidate_evaluation_config"],
      runner="tools/run_placement_strike.py" if who=="original_v32" else "tools/run_preimpact_wholebody_v048.py",
      output=c["baseline_output"] if who=="original_v32" else c["candidate_output"],
      offset=c["ball_offset_xy_m"],v=c["velocity_y_delta_mps"]))
 if len(jobs)!=12:raise ValueError("Only 12 predeclared original G1 PhysX runs allowed")
 for j in jobs:
  if (ROOT/j["output"]).exists():raise FileExistsError("Prior test result cannot be overwritten: "+j["output"])
 progress=ROOT/"reports/v048_real_physics_progress.json"
 if progress.exists():raise FileExistsError("Original physical study already started")
 completed=[]
 for i,j in enumerate(jobs,1):
  s=json.loads((ROOT/STUDY_JSON).read_text())
  preflight(s)
  if sha(s["candidate_evaluation_config"])!=now["eval_config_sha256"]:
   raise ValueError("G1 physical policy changed during v48 testing")
  args=[sys.executable,str(ROOT/j["runner"]),"--config",str(ROOT/j["config"]),
      "--ball-offset",str(j["offset"][0]),str(j["offset"][1]),
      "--velocity-delta",str(j["v"]),"--output",str(ROOT/j["output"])]
  start=time.monotonic()
  print("V048_REAL_G1_PHYSX_BEGIN",i,len(jobs),j["case"],j["controller"],j["offset"],j["v"],flush=True)
  with (ROOT/j["output"]).with_suffix(".log").open("w") as out:
   proc=subprocess.run(args,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=False)
  dest=ROOT/j["output"]
  rec=dict(case=j["case"],control=j["controller"],original_result_file=j["output"],
      exit=proc.returncode,elapsed_s=round(time.monotonic()-start,3),
      data_bytes=dest.stat().st_size if dest.exists() else 0,
      SHA256=sha(j["output"]) if dest.exists() else None)
  completed.append(rec)
  progress.write_text(json.dumps(dict(version="0.1.48",study_sha256=sha(STUDY_JSON),
      completed=len(completed),expected=len(jobs),sources=completed),indent=2)+"\n")
  print("V048_REAL_G1_PHYSX_COMPLETED",json.dumps(rec),flush=True)
  if proc.returncode!=0 or not dest.exists():
   raise RuntimeError("An original G1 PhysX process failed; do not rerun or choose another ball state")
 print("V048_ALL_TWELVE_NEW_G1_PHYSX_COMPLETE",len(completed),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--freeze",action="store_true")
 p.add_argument("--evaluate",action="store_true")
 a=p.parse_args()
 if a.freeze:prepare_eval()
 elif a.evaluate:evaluate()
 else:
  print("V048_KINETICKICK_RESEARCH_EXECUTOR_READY")
