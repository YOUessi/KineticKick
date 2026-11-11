#!/usr/bin/env python3
"""v90 independent source/physics/CMU/style/contact-head audit.

All source papers/data are already present on Tang: original G1 CUDA
PhsyX v84 C,D; v85 CENTER; CMU human optical v50. Never mutate them.
Independently reload two source-trained checkpoints and recalculate all
230 real 200Hz center historical contact-support prediction windows. No G1
new motor policy or off-policy physical success claim is permitted.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from kineticmimic_physics_v090 import (sha,load_original_physics,
   strict_physics,sample_contact_predictor)
from train_kineticmimic_v090 import (ContactImpactHead,rows_to_training,
   score,performance)

def _close(a,b,name,tol=1e-6):
 if a is None or b is None:
  if a!=b:raise ValueError("Missing metric parity "+name)
 elif not np.isfinite(a) or not np.isfinite(b) or abs(a-b)>tol:
  raise ValueError("Independent model score changed: "+name)

def independent_analysis(config_path,style_path,out):
 if out.exists():raise FileExistsError("Write-once v90 audit cannot overwrite raw evidence")
 c=json.loads(config_path.read_text())
 h=json.loads(style_path.read_text())
 model_dir=ROOT/c["output_dir"]
 savedpath=model_dir/"kineticmimic_contact_training_audit.json"
 result=json.loads(savedpath.read_text())
 if result["protocol_sha256"]!=sha(config_path) or h["version"]!="v0.1.90":
  raise ValueError("Historical source/study protocol SHA changed")
 if result["real_G1_PhysX_new_rollouts"]!=0 or result["model_has_NO_action_output"] is not True:
  raise ValueError("Cannot claim newly executed G1 actions in a CPU contact-head")
 for path,expected in c["source_sha256"].items():
  if sha(ROOT/path)!=expected:raise ValueError("Precommitted source code drift "+path)
 episodes={}
 for key,source in c["source_physx"].items():
  e=load_original_physics(Path(source["path"]),source["sha256"],key)
  episodes[key]=e
  if not any(x["case"]==key and x["raw_SHA256"]==e.trace_source_sha256
             for x in result["source_episode_physics"]):
   raise ValueError("Missing one registered original PhysX source record "+key)
 if len(episodes)!=9:
  raise ValueError("Exactly 9 true original G1 physics sources before score")
 physical={}
 for key,e in episodes.items():
  hard=strict_physics(e)
  physical[key]=dict(source_SHA256=e.trace_source_sha256,
     original_200hz_true_steps=e.true_physics_rows,
     nonphysics_terminal_annotation=e.terminal_annotation_count,
     original_measured_instep_impulse_fraction=e.measured_instep_impulse_fraction,
     independent_all_gates_pass=hard["strict_pass"],
     gate_details=hard["gates"],
     actual_max_29dof_qdot_over_original_limit=float(e.qdot_ratio.max()),
     real_foot_ball_contact_time_s=e.first_impact_s,
     ball_speed_5_to_105ms_mps=e.observed_ball_speed_5_to_105ms_mps,
     foot_ball_unsigned_force_integral_proxy_Ns=e.estimated_foot_ball_scalar_impulse_Ns,
     nonfoot_max_net_force_N=float(e.nonfoot_force_N.max()),
     original_recorded_14s_full=bool(e.true_physics_rows==2800),
     original_G1_contact_manifold_proven=False)
 if not (sum(x["independent_all_gates_pass"] for x in physical.values())==8 and
         not physical["center_cmu05"]["independent_all_gates_pass"] and
         physical["center_cmu05"]["nonphysics_terminal_annotation"]==1):
  raise ValueError("Original physical negative controls modified/incorrect")
 center=[e for key,e in episodes.items() if c["source_physx"][key]["scene"]=="v085_center"]
 assert len(center)==3
 dataset=rows_to_training(center,c)
 holdout={}
 for name,original in result["fitted_contact_heads"].items():
  p=model_dir/(name+"_contact_impact_head.pt")
  if sha(p)!=original["checkpoint_sha256"]:
   raise ValueError("Source-trained contact head checkpoint modified: "+name)
  # Our own SHA-verified local checkpoint stores two NumPy normalization
  # arrays (not compatible with weights_only=True). Never load external files.
  state=torch.load(p,map_location="cpu",weights_only=False)
  if state["not_G1_actor"] is not True or state["human_true_GRf_or_impact_available"] is not False:
   raise ValueError("Original trained classifier pretending to be true G1 actor/human GRF")
  if set(state["train_source_SHA256"])!={key for key in episodes if key.startswith("C_")}:
   raise ValueError("Model trained on center tests or validation D (leakage)")
  for key,expected in state["train_source_SHA256"].items():
   if expected!=episodes[key].trace_source_sha256:raise ValueError("Training SHA not locked")
  mean=np.asarray(state["normalization_mean"],float);std=np.asarray(state["normalization_std"],float)
  if mean.shape!=(71,) or std.shape!=(71,) or (std<=0).any():
   raise ValueError("No training-only original G1 normalizer")
  d=dict(dataset)
  d["x"]=np.clip((dataset["x"]-mean)/std,-12,12).astype(np.float32)
  m=ContactImpactHead(c["hidden"])
  m.load_state_dict(state["model"]);m.eval()
  new=score(m,d,shuffled=name!="ordered")
  old=original["test"]
  _close(new["predicted_stance_accuracy"],old["predicted_stance_accuracy"],"stance_accuracy")
  _close(new["hazard_next_100ms_F1"]["f1"],old["hazard_next_100ms_F1"]["f1"],"impact_F1")
  _close(new["hazard_prob_Brier"],old["hazard_prob_Brier"],"hazard_Brier")
  _close(new["impact_postball_delta_velocity_MAE_mps"],old["impact_postball_delta_velocity_MAE_mps"],"true_ball_velocity")
  _close(new["impact_contact_scalar_impulse_MAE_Ns"],old["impact_contact_scalar_impulse_MAE_Ns"],"true_scalar_impulse")
  holdout[name]=dict(original_checkpoint_SHA256=sha(p),recomputed_scores=new,
     only_original_robot_200hz_measured_labels=True,
     actually_generated_a_29dof_controller=False,
     original_C_scene_only_training=True,D_scene_only_validation=True,
     center_historically_seen_not_blind=True)
 if not len(h["canonical_physical_g1_runs"])==3 or h["human_has_verified_true_instep_football_contact"] is not False:
  raise ValueError("Human optical semantic versus true G1 physical evidence false claim")
 style={}
 for row in h["canonical_physical_g1_runs"]:
  k=row["case"]
  if row["source_original_200hz_SHA256"]!=episodes[k].trace_source_sha256:
   raise ValueError("G1 geometry source not same exact genuine 5ms scene")
  if row["original_strict_physics"]["strict_pass"]!=physical[k]["independent_all_gates_pass"]:
   raise ValueError("Optical G1 style audit incorrectly promoted physical shot")
  style[k]=dict(
   human10_02_relative_geometry_MSE=row["true_original_20step_G1_human_geom_relations"]["10_02"]["geometric_relative_mse_not_naturalness"],
   human10_05_relative_geometry_MSE=row["true_original_20step_G1_human_geom_relations"]["10_05"]["geometric_relative_mse_not_naturalness"],
   source_29dof_real_200hz_qdot_peak_times_relative_to_impact=row["normalized_200hz_G1_limb_group_soft_peak_phases"],
   not_human_professional_GRf_or_true_contact=True)
 conclusion=dict(version="v0.1.90",date="2026-11-12",
    precommit_study_SHA256=sha(config_path),optical_audit_SHA256=sha(style_path),
    original_physical_sources_9_episodes=len(episodes),
    physically_independent_contexts_only_3=True,
    real_new_PhysX_episodes=0,
    original_G1_CUDA_200hz_strict_physics_per_source=physical,
    historical_strict_pass_among_9=sum(v["independent_all_gates_pass"] for v in physical.values()),
    real_G1_original_center_CMU05_nonfoot_failed=True,
    original_training_historical_center_test_has_been_seen=True,
    contact_head_validation_and_center_independent_reloads=holdout,
    no_model_reliably_predicts_contact_on_center=all(
      z["recomputed_scores"]["hazard_next_100ms_F1"]["f1"]==0 for z in holdout.values()),
    stage_prediction_success_does_not_mean_kinetic_mimic_skill=True,
    CMU_human_optical_six_relative_geometry=style,
    G1_true_native_wrist_hip_contact_pair_certified=False,
    human_foot_ball_true_contact_or_GRF_truth_available=False,
    full_body_generated_29dof_real_physics_tested=False,
    original_v32_default_controller_unchanged=True)
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(conclusion,ensure_ascii=False,indent=2)+"\n")
 print("KINETICMIMIC_V090_INDEPENDENT_AUDIT",json.dumps(dict(
  source_real_G1=len(episodes),real_G1_new=0,
  strict_original_physics=conclusion["historical_strict_pass_among_9"],
  contact_F1_by_model={k:v["recomputed_scores"]["hazard_next_100ms_F1"]["f1"]
                       for k,v in holdout.items()},
  human_true_contacts_certified=0,
  G1_hardware_safe_mimic_trained=False)),flush=True)
 return conclusion

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--config",type=Path,default=ROOT/"configs/kineticmimic_v090_study.json")
 p.add_argument("--style",type=Path,default=ROOT/"reports/v090_human_robot_relative_geometry.json")
 p.add_argument("--output",type=Path,default=ROOT/"reports/v090_kineticmimic_independent_audit.json")
 a=p.parse_args()
 torch.set_num_threads(4)
 if torch.cuda.is_available():raise RuntimeError("G1 physical re-audit must not occupy concurrent GPU")
 independent_analysis(a.config,a.style,a.output)
if __name__=="__main__":main()
