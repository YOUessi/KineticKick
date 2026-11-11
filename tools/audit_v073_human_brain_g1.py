#!/usr/bin/env python3
"""Independent v73 CPU human-to-G1 safety/forecast audit (NO IsaacLab/GPU).

Read SHA-frozen 120Hz CMU optical sources, both original CPU-trained GRU
checkpoints, true completely held-out subject-11 test window, and offline
G1 URDF 700x29 retarget proposals. Preserve source and training artifacts.
A G1 IK proposal is NOT a real 200Hz physical rollout nor proven safe kick.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from train_human_temporal_brain_v073 import (sha,load_verified,clip_windows,
 human_metrics,linear_velocity,persistence,infer)
from retarget_v073_human_brain_to_g1 import load_checkpoint
from wholebody_safety_interface_v055 import G1_JOINT_ORDER
from g1_contact_coordination_projection_v040 import load_robot_source,interpolate_source
from build_cmu_g1_candidate_v052 import joint_limits

def match_metric(new,old):
 fields=("normalized_xyz_landmark_euclidean_MAE",
         "observed_8_segment_cosine_MAE",
         "mean_future_xyz_velocity_absolute_error_per_120hz_frame",
         "predicted_cosine_bound_violation_rate")
 for key in fields:
  if abs(new[key]-old[key])>2e-6:raise ValueError("Heldout human score drift: "+key)

def audit(source_root:Path,brain:Path,proposal:Path,out:Path):
 if out.exists():raise FileExistsError("Cannot replace independent v73 audit")
 cfgpath=ROOT/"configs/v073_human_temporal_learning.json"
 cfg=json.loads(cfgpath.read_text())
 brain_file=brain/"human_motion_evaluation.json"
 report=json.loads(brain_file.read_text())
 assert cfg["version"]=="v0.1.73" and report["protocol_sha256"]==sha(cfgpath)
 assert cfg["train_clips"]==["10_01","10_02","10_03","10_05"]
 assert cfg["validation_clips"]==["10_06"] and cfg["test_clips"]==["11_01"]
 assert not cfg["test_subject_seen_in_training"] and not cfg["professional_instestep_contact_groundtruth"]
 assert not cfg["human_GRF_truth"] and cfg["no_CUDA_no_robot_policy_no_PhysX"]
 for f,expected in cfg["source_code_sha256"].items():
  if sha(ROOT/f)!=expected:raise ValueError("Original training source changed: "+f)
 human_dir=source_root/cfg["original_human_rel_dir"]
 sources={}
 for clip,meta in cfg["original_human_source"].items():
  path=human_dir/(clip+".npz")
  f,valid,t,landmarks=load_verified(path,meta["sha256"],meta["subject"])
  if len(f)!=report["original_cmu_capture_frames"][clip]:
   raise ValueError("Original 120Hz human frame-count source drift")
  sources[clip]=dict(n_frames=len(f),valid_frames=int(valid.sum()),
    subject=meta["subject"],sha256=sha(path))
 if set(v["subject"] for c,v in sources.items() if c in cfg["train_clips"])!={"10"}:
  raise ValueError("Other human subject in train")
 if sources["11_01"]["subject"]!="11":raise ValueError("Subject 11 withheld source drift")
 qpath=proposal/"human_brain_to_G1_reference_audit.json"
 proposals=json.loads(qpath.read_text())
 if proposals["version"]!="v0.1.73" or proposals["no_real_robot_PhysX_actuation_or_high_power_claim"] is not True:
  raise ValueError("Original robot physical safety misrepresented")
 g1cfg=json.loads((ROOT/"configs/cmu_phase_model_v064.json").read_text())
 old_robot=source_root/"reports/v063_center_v32.json"
 if sha(old_robot)!=g1cfg["frozen_source_sha256"]["reports/v063_center_v32.json"]:
  raise ValueError("Original G1 robot recorded real physics SHA drift")
 robot=load_robot_source(old_robot)
 assert tuple(robot["joint_names"])==G1_JOINT_ORDER
 joints=joint_limits(robot["joint_names"])
 verify_predictions={}
 verify_proposals={}
 for name,metadata in report["checkpoints"].items():
  p=brain/(name+"_human_motion_gru.pt")
  model,mean,std=load_checkpoint(p,metadata["sha256"],24,12,cfg["gru_hidden"])
  x,mask,_,_=load_verified(human_dir/"11_01.npz",
      cfg["original_human_source"]["11_01"]["sha256"],"11")
  holdx,holdy,_=clip_windows(x,mask,24,12,cfg["eval_stride_frames"],mean,std)
  re_pred=infer(model,holdx)
  newly=human_metrics(re_pred,holdy,mean,std)
  match_metric(newly,report["test_subject_11_per_method"][name])
  baseline_p=human_metrics(persistence(holdx,12),holdy,mean,std)
  baseline_l=human_metrics(linear_velocity(holdx,12),holdy,mean,std)
  match_metric(baseline_p,report["test_subject_11_per_method"]["persistence"])
  match_metric(baseline_l,report["test_subject_11_per_method"]["linear_extrapolation"])
  verify_predictions[name]=dict(test_subject="11",observed_source_windows=len(holdx),
    exact_MAE_from_reloaded_checkpoint=newly["normalized_xyz_landmark_euclidean_MAE"],
    subject11_generalization_supported=(
        newly["normalized_xyz_landmark_euclidean_MAE"]<
        baseline_l["normalized_xyz_landmark_euclidean_MAE"]),
    model_sha256=sha(p))
  pr=proposals["models"][name]
  if pr["checkpoint_sha256"]!=sha(p):raise ValueError("G1 proposal from changed human brain")
  if pr["status"]!="OFFLINE_G1_REFERENCE_ONLY_NOT_PHYSX":
   verify_proposals[name]=dict(status=pr["status"],not_actuated=True)
   continue
  refpath=ROOT/pr["source_generated_G1_29dof_file"]
  if not refpath.exists() or sha(refpath)!=pr["source_generated_G1_29dof_sha256"]:
   raise ValueError("G1 29DoF source proposal altered")
  with np.load(refpath,allow_pickle=False) as z:
   delta=np.asarray(z["joint_delta_rad"],float)
   times=np.asarray(z["times_s"],float)
   names=z["joint_names"].tolist()
   no_physics=bool(z["no_actual_PhysX_evidence"].item())
  if (delta.shape!=(700,29) or tuple(names)!=G1_JOINT_ORDER or
      not no_physics or not np.isfinite(delta).all() or
      not np.allclose(times,np.arange(700)*.02,atol=1e-6)):
   raise ValueError("Nonfinite or unqualified original G1 source reference")
  if (np.max(abs(delta))>.05501 or np.max(abs(np.diff(delta,axis=0)))>.00401 or
      np.any(delta[:,[1,4,7,10,14,18]]!=0) or
      np.max(abs(delta[times>=1.42-1e-6]))>1e-7):
   raise ValueError("Original 29DoF G1 right foot / 20ms action envelope breached")
  active=(np.max(abs(delta),axis=1)>1e-6)
  if np.count_nonzero(active)<5:raise ValueError("Original human prior did not create any G1 motor reference")
  # Static original source q may itself be near original URDF limit;
  # candidate is allowed ONLY when it does not introduce a new violation.
  r,q=interpolate_source(robot,times)
  orig=(q<joints[:,0]+.02)|(q>joints[:,1]-.02)
  newer=(q+delta<joints[:,0]+.02)|(q+delta>joints[:,1]-.02)
  n_new=int(np.count_nonzero(newer & ~orig))
  if n_new>0:raise ValueError("Original G1 reference introduces static joint position-limit violation")
  verify_proposals[name]=dict(
    status="ONLY_OFFLINE_REFERENCE_NOT_REAL_PHYSX",
    applied_new_200hz_PhysX_rollouts=0,
    active_G1_20ms_command_reference_frames=int(active.sum()),
    active_G1_joints=int(np.count_nonzero(np.max(abs(delta),axis=0)>1e-6)),
    actual_kinematic_reference_max_delta_rad=float(np.max(abs(delta))),
    static_URDF_new_q_margin_violations=n_new,
    original_right_instep_leg_indices_untouched=True,
    source_SHA256=sha(refpath))
 audited=dict(version="v0.1.73",source_protocol_SHA256=sha(cfgpath),
   source_human_clip_schema=source_root.as_posix(),
   source_human_counts=sources,
   human_subject_train_one_test_one=True,
   original_training_report_SHA256=sha(brain_file),
   train_subject_10_val_and_test_subject_11_separated=True,
   no_independent_human_elite_skill_or_force_contact_true_labels=True,
   original_G1_robot_reference_SHA256=sha(old_robot),
   independent_subject11_test_checkpoint_reloads=verify_predictions,
   original_g1_29dof_URDF_and_joint_envelope=verify_proposals,
   real_PhysX_new_rollouts=0,
   exact_foot_ball_contact_and_CUDA_contact_manifold_evidence=0,
   learned_G1_controller_evidence=False,
   original_robot_physical_safety_not_certified=True,
   naturalness_metric_is_unavailable=True,
   original_v72_concurrent_GPU_job_not_disturbed=True,
   no_source_or_checkpoint_overwrite=True)
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(audited,ensure_ascii=False,indent=2)+"\n")
 print("V073_INDEPENDENT_MOTION_TO_G1_AUDIT",json.dumps(dict(
  heldout_subject11={k:v["exact_MAE_from_reloaded_checkpoint"] for k,v in verify_predictions.items()},
  offline_only_G1={k:v.get("active_G1_20ms_command_reference_frames") for k,v in verify_proposals.items()},
  new_real_physx_count=0)),flush=True)
 return audited

if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--source-root",type=Path,required=True)
 p.add_argument("--brain",type=Path,required=True)
 p.add_argument("--proposal",type=Path,required=True)
 p.add_argument("--output",type=Path,default=ROOT/"reports/v073_independent_fullhuman_g1_audit.json")
 a=p.parse_args()
 torch.set_num_threads(4)
 if torch.cuda.is_available():raise RuntimeError("CPU-only v73 can't use v72 reserved GPU")
 audit(a.source_root,a.brain,a.proposal,a.output)
