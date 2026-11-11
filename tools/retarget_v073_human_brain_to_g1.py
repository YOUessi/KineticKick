#!/usr/bin/env python3
"""v73: CPU-only, NO ROBOT CONTROL: self-supervised human GRU→G1 29DoF URDF IK.

Accept only SHA-certified local trained checkpoints and original CMU/C3D
source. Forecast a training-person motion window 0.4s before the *inferred*
optical ball-marker launch onset; map six body segment cosines to original
29DoF URDF using the existing conservative v64 IK projection. Do NOT claim
exact foot-ball timing, elite instep biomechanics, actual PhysX, naturalness
or certified contact. This generates proposals, not real robot actions.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from train_human_temporal_brain_v073 import (FEATS,TemporalBrain,load_verified,sha)
from train_cmu_phase_generator_v064 import make_reference
from cmu_g1_semantic_bridge_v051 import HUMAN_FEATURE_INDEX
from g1_contact_coordination_projection_v040 import load_robot_source
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

def load_checkpoint(path,expected,h,forecast,hidden):
 if sha(path)!=expected:raise ValueError("Original trained human checkpoint source changed")
 # The only locally generated checkpoint is SHA-verified before deserialize.
 # The checkpoint is not accepted from untrusted remote sources.
 chk=torch.load(path,map_location="cpu",weights_only=False)
 if chk["history"]!=h or chk["forecast"]!=forecast or chk["features"]!=FEATS:
  raise ValueError("Wrong original observed human semantic feature order")
 if chk["train_clip_subject"]!="10" or chk["test_clip_subject"]!="11" or chk["model_hidden"]!=hidden:
  raise ValueError("Human trained source/test subject qualification corrupted")
 model=TemporalBrain(h,forecast,hidden=hidden)
 model.load_state_dict(chk["state_dict"],strict=True)
 model.eval()
 mean=np.asarray(chk["mean"],dtype=np.float64)
 std=np.asarray(chk["std"],dtype=np.float64)
 if mean.shape!=(65,) or std.shape!=(65,) or (std<=0).any():
  raise ValueError("Invalid human-only training normalization")
 return model,mean,std

def autoregressive_future(model,initial,mean,std,forecast=12,n_blocks=4):
 if initial.shape!=(24,65) or not np.isfinite(initial).all():
  raise ValueError("No clean initial human history for G1 retargeting")
 history=((initial-mean)/std).astype(np.float32)
 generated=[]
 with torch.no_grad():
  for _ in range(n_blocks):
   pred=model(torch.tensor(history[None],dtype=torch.float32))[0].cpu().numpy()
   if pred.shape!=(forecast,65) or not np.isfinite(pred).all():
    raise ValueError("Human phase GRU generated nonfinite original 3D/cosine future")
   generated.append(pred)
   history=np.concatenate([history,pred],axis=0)[-24:]
 whole=np.concatenate(generated)
 original=whole*std+mean
 if original.shape!=(48,65):raise ValueError("Unexpected relative CMU future horizon")
 # Numeric safety check of pose/cosines. No post-hoc clipping for scoring.
 if np.max(abs(original[:,:57]))>5. or np.max(abs(original[:,57:]))>3.:
  raise ValueError("Human-only neural rollout diverged; no G1 motion proposal allowed")
 return original

def select_original_window(pose,valid,time,anchor):
 # 24 raw 120Hz frames finish immediately before 0.40 seconds prior to the
 # observed *ball marker motion onset*. This is NOT a verified impact event.
 first=int(np.searchsorted(time,anchor-.40,side="left"))
 if first<24 or first+48>len(time):
  raise ValueError("Human optical launching window not fully measured")
 if not valid[first-24:first+48].all():
  raise ValueError("Original CMU raw marker dropout crosses event/human prior window")
 return first,pose[first-24:first],pose[first:first+48]

def six_20hz_human(whole):
 if whole.shape!=(48,65):raise ValueError("Only 48-frame actual human future allowed")
 six=whole[:,57+np.asarray(HUMAN_FEATURE_INDEX)]
 if np.any(~np.isfinite(six)):raise ValueError("Invalid original CMU segment values")
 # 20 x 50Hz relative phase proxy. The human 120Hz source covers a true
 # 0.4sec sampling horizon; interpolate only the MODEL predicted 48 frames.
 src=np.arange(48)/120.
 timestamps=np.arange(20)*.02
 fitted=np.stack([np.interp(timestamps,src,six[:,j]) for j in range(6)],axis=1)
 return fitted-fitted[:1]

def run(source_root,brain_output,proposal_output):
 if proposal_output.exists():raise FileExistsError("Never replace published G1 brain proposal")
 protocol=json.loads((ROOT/"configs/v073_human_temporal_learning.json").read_text())
 result_path=brain_output/"human_motion_evaluation.json"
 evaluation=json.loads(result_path.read_text())
 if evaluation["protocol_sha256"]!=sha(ROOT/"configs/v073_human_temporal_learning.json"):
  raise ValueError("Human brain training protocol mismatch")
 if evaluation["split_exact"]["heldout"]!=["11_01"]:
  raise ValueError("Independent human test subject not held out")
 cfg=json.loads((ROOT/"configs/cmu_phase_model_v064.json").read_text())
 if cfg["human_contact_groundtruth"] or cfg["elite_player_power_verified"]:
  raise ValueError("Original v64 safety retargeter falsely qualified impact")
 original_robot=source_root/"reports/v063_center_v32.json"
 if sha(original_robot)!=cfg["frozen_source_sha256"]["reports/v063_center_v32.json"]:
  raise ValueError("Original actual 200Hz G1 robot reference changed")
 robot=load_robot_source(original_robot)
 if tuple(robot["joint_names"])!=G1_JOINT_ORDER:
  raise ValueError("Robot 29DoF G1 original physical joint ordering changed")
 src=source_root/protocol["original_human_rel_dir"]/"10_02.npz"
 feat,mask,t,_=load_verified(src,protocol["original_human_source"]["10_02"]["sha256"],"10")
 qa=source_root/"data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050/quality_audit.json"
 records=json.loads(qa.read_text())
 clip=next(x for x in records["motions"] if x["id"]=="10_02")
 if clip["inferred_event"]["verified_true_foot_ball_contact"] is not False:
  raise ValueError("A human true impact label cannot suddenly be invented")
 anchor=float(clip["inferred_event"]["inferred_ball_launch_onset_s"])
 start,initial,target=select_original_window(feat,mask,t,anchor)
 if len(target)!=48:raise ValueError("Human observed ground pose horizon incomplete")
 summary=dict(version="v0.1.73",models={},original_human_train_clip="CMU 10_02",
  same_person_as_training=True,source_human_marker_sha256=sha(src),
  original_robot_5ms_physical_baseline_SHA256=sha(original_robot),
  event_name="inferred outgoing optical soccer-ball marker onset NOT first foot-ball contact",
  inferred_human_event_s=anchor,
  original_human_frame_start_index=start,
  human_new_subject_test_was_done_separately=True,
  no_real_robot_PhysX_actuation_or_high_power_claim=True,
  G1_reference_does_not_certify_foot_collision_or_followthrough=True)
 proposal_output.mkdir(parents=True)
 for name,check in evaluation["checkpoints"].items():
  model,mean,std=load_checkpoint(brain_output/(name+"_human_motion_gru.pt"),
      check["sha256"],24,12,protocol["gru_hidden"])
  forecast=autoregressive_future(model,initial,mean,std)
  human20=six_20hz_human(forecast)
  # Existing URDF IK applies original 29DoF joint limits, hard foot-shift
  # constraints, slew and total amplitude budget; never weaken them here.
  outcome=dict(checkpoint_sha256=check["sha256"],
    human_48frame_8cosine_error_vs_original_same_training_clip=float(
      np.mean(np.abs(forecast[:,57:]-target[:,57:]))),
    human_48frame_xyz_normalized_landmark_Euclidean_MAE_vs_same_training_clip=float(
      np.mean(np.linalg.norm((forecast[:,:57]-target[:,:57]).reshape(48,19,3),axis=-1))),
    cross_subject_heldout_3D_MAE=evaluation["test_subject_11_per_method"][name]["normalized_xyz_landmark_euclidean_MAE"],
    direct_120hz_CMU_G1_synced_true_impact=False)
  try:
   ref,metrics=make_reference(robot,human20,cfg)
   if ref.shape!=(700,29) or not np.isfinite(ref).all():
    raise ValueError("No original 700x29 true G1 configuration produced")
   if np.max(np.abs(ref[:,[1,4,7,10,14,18]]))!=0:
    raise ValueError("Original right-leg strike geometry not protected")
   if np.max(abs(ref))>cfg["max_joint_delta_rad"]+1e-5 or np.max(abs(np.diff(ref,axis=0)))>cfg["max_slew_per_20ms_rad"]+1e-5:
    raise ValueError("Original physical reference IK exceeded conservative source envelope")
   if np.any(abs(ref[robot["times"]>=1.42-1e-6])>1e-7):
    raise ValueError("Original G1 instep window not protected")
   p=proposal_output/(name+"_G1_29dof_reference.npz")
   np.savez_compressed(p,joint_delta_rad=ref,
     times_s=robot["times"],joint_names=np.asarray(robot["joint_names"]),
     source_model_sha256=np.asarray(check["sha256"]),
     no_actual_PhysX_evidence=np.asarray(True))
   outcome.update(status="OFFLINE_G1_REFERENCE_ONLY_NOT_PHYSX",
       source_generated_G1_29dof_sha256=sha(p),source_generated_G1_29dof_file=str(p.relative_to(ROOT)),
       original_G1_URDF_constraints_passed=True,
       original_physical_foot_collision_unverified=True,retarget=metrics)
  except ValueError as e:
   outcome.update(status="REJECTED_BY_G1_CONSTRAINED_IK",error=str(e),
     original_G1_URDF_constraints_passed=False)
  summary["models"][name]=outcome
 path=proposal_output/"human_brain_to_G1_reference_audit.json"
 path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
 print("V073_CMU_GRU_TO_ORIGINAL_G1_OFFLINE_PROPOSALS",json.dumps({k:dict(
     status=v["status"],human_error=v["human_48frame_xyz_normalized_landmark_Euclidean_MAE_vs_same_training_clip"],
     G1_ref_peak=v.get("retarget",{}).get("original_step_peak")) for k,v in summary["models"].items()}),flush=True)
 return summary

if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--source-root",type=Path,required=True)
 p.add_argument("--brain-output",type=Path,required=True)
 p.add_argument("--proposal-output",type=Path,required=True)
 a=p.parse_args()
 torch.set_num_threads(4)
 if torch.cuda.is_available():raise RuntimeError("GPU in-use by independent v72 IsaacLab triage; CPU retarget only")
 run(a.source_root,a.brain_output,a.proposal_output)
