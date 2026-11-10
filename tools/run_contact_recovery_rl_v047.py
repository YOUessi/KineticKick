#!/usr/bin/env python3
"""v47 one exact safety-aware 29DoF G1 PPO model vs original frozen v32.

Do not refit or select a checkpoint using the 6 predeclared fresh continuous
physical ball/velocity conditions. Current 7s training env uses 18 already
observed existing cases only. New model carries no human-motion expert claim.
"""
from pathlib import Path
import json,hashlib,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/learned_contact_recovery_study_v047.json"
def sha(p):
 h=hashlib.sha256()
 with (ROOT/p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def preflight():
    study=json.loads((ROOT/STUDY).read_text())
    if study["version"]!="0.1.47" or not study["no_hyperparameter_sweep"]:
        raise ValueError("Only original frozen one-shot study")
    for path,digest in study["source_sha256"].items():
        if sha(path)!=digest:raise ValueError("Original training source drift "+path)
    train_dir=study["training"]["output_directory"]
    result=json.loads((ROOT/train_dir/"result.json").read_text())
    if result["updates"]<1 or not result["checkpoint_sha256"]:
        raise ValueError("No bounded genuine PPO training updates")
    newmodel=train_dir+"/last_model.pt"
    if sha(newmodel)!=result["checkpoint_sha256"]:
        raise ValueError("Final frozen PPO checkpoint drift")
    oldmodel=study["training"]["warmstart"]
    if sha(oldmodel)!=study["frozen_recovery_actor_start_SHA256"]:
        raise ValueError("Original source G1 PPO checkpoint drift")
    return study,newmodel,result

def build_eval_cfg(study,newmodel):
    out=ROOT/study["final_evaluation_config"]
    if out.exists():raise FileExistsError("No eval config overwrite allowed")
    template=json.loads((ROOT/study["original_v32_evaluation_config"]).read_text())
    assert template["post_shot_recovery"]["checkpoint"]==study["training"]["warmstart"]
    assert template["post_shot_recovery"]["checkpoint_sha256"]==study["frozen_recovery_actor_start_SHA256"]
    template["version"]="0.1.47"
    template["status"]="TRUE_PHYSX_PPO_RECOVERY_RISK_FINETUNE_RESEARCH_NOT_PRODUCTION"
    template["original_v32_model_saved"]=study["frozen_recovery_actor_start_SHA256"]
    template["post_shot_recovery"]["checkpoint"]=newmodel
    template["post_shot_recovery"]["checkpoint_sha256"]=sha(newmodel)
    template["v047_recovery_evaluation"]=dict(
        source_train_study_sha256=sha(STUDY),
        training_model_checkpoint_sha256=sha(newmodel),
        training_case_count=18,
        original_qdot_and_PD_limits_unchanged=True,
        source_human_professional_instep_data_qualified=False,
        safe_g1_recovery_not_certified=True,
        model_selected_before_new_test=True)
    out.write_text(json.dumps(template,indent=2,ensure_ascii=False)+"\n")
    return study["final_evaluation_config"]

def execute():
    study,newmodel,training=preflight()
    evalconfig=study["final_evaluation_config"]
    if not (ROOT/evalconfig).exists() or sha(evalconfig)!=study["final_evaluation_config_sha256"]:
        raise ValueError("New model evaluation config must be SHA-locked before all new test episodes")
    if "/env_isaaclab/" not in sys.executable:
        raise ValueError("Must use original IsaacLab environment")
    jobs=[]
    for case in study["predeclared_eval_exact_six_ball_conditions"]:
        for name in ("v32","learned"):
            jobs.append(dict(id=case["id"],model=name,
               config=study["original_v32_evaluation_config"] if name=="v32" else evalconfig,
               offset=case["ball_offset_xy_m"],delta=case["velocity_y_delta_mps"],
               output=case["old_control_output"] if name=="v32" else case["trained_control_output"]))
    if len(jobs)!=12:raise ValueError("Six paired full G1 physical episodes required")
    for job in jobs:
        if (ROOT/job["output"]).exists():raise FileExistsError("Do not reuse testing result "+job["output"])
    progress=ROOT/"reports/v047_original_physics_progress.json"
    if progress.exists():raise FileExistsError(progress)
    completed=[]
    for i,job in enumerate(jobs,1):
        study_now,model_now,_=preflight()
        if sha(model_now)!=sha(newmodel):raise ValueError("Frozen model changed during original PhysX")
        dest=ROOT/job["output"]
        args=[sys.executable,str(ROOT/"tools/run_placement_strike.py"),
             "--config",str(ROOT/job["config"]),
             "--ball-offset",str(job["offset"][0]),str(job["offset"][1]),
             "--velocity-delta",str(job["delta"]),"--output",str(dest)]
        print("V047_G1_PHYSX_BEGIN",i,len(jobs),job["id"],job["model"],
              job["offset"],job["delta"],flush=True)
        t0=time.monotonic()
        with dest.with_suffix(".log").open("w") as log:
            proc=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        record=dict(index=i,case=job["id"],controller=job["model"],
            source=job["output"],return_code=proc.returncode,
            seconds=round(time.monotonic()-t0,3),
            bytes=dest.stat().st_size if dest.exists() else 0,
            source_sha256=sha(job["output"]) if dest.exists() else None)
        completed.append(record)
        progress.write_text(json.dumps(dict(version="0.1.47",study_sha256=sha(STUDY),
             full_physics_completed=len(completed),prespecified=12,outputs=completed),
             indent=2)+"\n")
        print("V047_G1_PHYSX_COMPLETE",json.dumps(record),flush=True)
        if proc.returncode!=0 or not dest.exists():
            raise RuntimeError("Original Isaac physics worker failed; preserve all data and stop")
    print("V047_G1_PRESPECIFIED_COMPLETE",len(completed),flush=True)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--execute",action="store_true")
    a=ap.parse_args()
    if a.execute: execute()
    else:
        s,m,r=preflight()
        print("V047_TRAINED_CHECKPOINT_READY",m,r["updates"])
