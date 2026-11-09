"""Physical whole-body coordination shift with a *video-lifted* human proxy.

A physical change in URDF segment cosines is not demonstrated human
naturalness. No physical model is trained or changed by this read-only audit.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
from g1_contact_coordination_projection_v040 import (
    TorchG1Relations,load_robot_source,interpolate_source,build_human_relative_target)
from build_professional_shot_coordination_v039 import PAIRS

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/research_human_kicks/3dsp_cvpr2024/processed_v039/coordination_3dsp_v039.npz"

def analyze(case):
 study=json.loads((ROOT/"configs/g1_coordination_style_physics_v041.json").read_text())
 row=next(x for x in study["conditional_case_order"] if x["id"]==case)
 baseline=ROOT/row["baseline"]
 candidate=ROOT/row["candidate"]
 old=json.loads(baseline.read_text())
 new=json.loads(candidate.read_text())
 if len(old["kinematics"][0])!=700 or len(new["kinematics"][0])!=700:
  raise ValueError("Only complete PhysX 14s effects valid")
 data=np.load(DATA,allow_pickle=False)
 if bool(data["verified_event_contact_truth"]):raise ValueError("Unqualified human impact truth")
 c=np.flatnonzero(data["clip_id"]=="00001")
 if len(c)!=1 or data["split"][c[0]]!="train":raise ValueError("Previously fixed human TRAIN clip unavailable")
 human=data["relation_cosines"][c[0]].astype(float)
 original=load_robot_source(baseline)
 physical=load_robot_source(candidate)
 if original["joint_names"]!=physical["joint_names"]:
  raise ValueError("Different G1 joint order")
 tc=float(original["contact_time"])
 ts=tc-.5+np.arange(20)*.04
 root0,q0=interpolate_source(original,ts)
 root1,q1=interpolate_source(physical,ts)
 fk=TorchG1Relations(original["joint_names"])
 with torch.no_grad():
  rel0,links0=fk(torch.tensor(root0,dtype=torch.float64),
                  torch.tensor(q0,dtype=torch.float64),with_links=True)
  rel1,links1=fk(torch.tensor(root1,dtype=torch.float64),
                  torch.tensor(q1,dtype=torch.float64),with_links=True)
  a=rel0.numpy()
  b=rel1.numpy()
 target,_=build_human_relative_target(a,human,
      scale=json.loads((ROOT/"configs/g1_coordination_generation_v040.json").read_text())["source_relative_scale"])
 # Evaluate only the frozen pre-contact band; never shift the winner's
 # physical time axis to favor one answer.
 mask=(ts<tc-.04)
 def measured_mse(x):return float(((x-target)[mask]**2).mean())
 names=original["joint_names"]
 qdiff=abs(q1-q0)
 groups={
  "left_stance_leg":[i for i,n in enumerate(names) if n.startswith("left_") and ("hip" in n or "knee" in n or "ankle" in n)],
  "right_swing_leg":[i for i,n in enumerate(names) if n.startswith("right_") and ("hip" in n or "knee" in n or "ankle" in n)],
  "waist":[i for i,n in enumerate(names) if n.startswith("waist_")],
  "upper_body":[i for i,n in enumerate(names) if any(part in n for part in ("shoulder","elbow","wrist"))]}
 assert sum(len(x) for x in groups.values())==29
 group_stats={}
 for name,index in groups.items():
  group_stats[name]=dict(affected_joints=len(index),
     actual_q_max_delta_rad=float(qdiff[:,index].max()),
     actual_q_mean_abs_delta_rad=float(qdiff[:,index].mean()))
 link_deviation={}
 for foot in ("left_ankle_roll_link","right_ankle_roll_link"):
  o=links0[foot][:,:3,3].numpy()
  n=links1[foot][:,:3,3].numpy()
  link_deviation[foot]=dict(max_translation_m=float(np.linalg.norm(n-o,axis=-1).max()),
                            mean_translation_m=float(np.linalg.norm(n-o,axis=-1).mean()))
 result={
  "version":"0.1.41",
  "case":case,
  "human_proxy_train_clip_id":"00001",
  "human_impact_event_verified":False,
  "robot_impact_event_verified":True,
  "human_vs_robot_semantic_equivalence_verified":False,
  "prespecified_preimpact_samples":int(mask.sum()),
  "real_physical_reference_goal_mse":measured_mse(a),
  "real_physical_generated_goal_mse":measured_mse(b),
  "physical_proxy_mse_change":measured_mse(b)-measured_mse(a),
  "full_joint_actual_difference_peak_rad":float(qdiff.max()),
  "full_joint_actual_difference_mean_rad":float(qdiff.mean()),
  "mean_absolute_robot_relation_change":float(abs(b-a).mean()),
  "maximum_absolute_robot_relation_change":float(abs(b-a).max()),
  "relative_proxy_error_reduction":float((measured_mse(a)-measured_mse(b))/max(1e-9,measured_mse(a))),
  "body_group_actual_motion_delta":group_stats,
  "foot_link_actual_displacement":link_deviation,
  "contact_time_original_s":original["contact_time"],
  "contact_time_modified_s":physical["contact_time"],
  "ball_speed_metric":"See independent original PhysX gate: this proxy audit does not estimate ball impulse",
  "original_goal":bool(old["records"][0]["goal"]),
  "candidate_goal":bool(new["records"][0]["goal"]),
  "limitations":["Human reference source is video-estimated skeleton joint directions, not calibrated biomechanics",
    "Do not equate a lower feature MSE to human naturalness or high impact power",
    "Only one known original G1 physical case, no unseen-state human transfer",
    "Robot original body link-frames are not human anatomical joint centers",
    "No contact/force model from 3DSP, no joint torque or ball spin obtained from human source"]}
 return result,dict(time=ts,original=a,generated=b,human_proxy=target)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--case",required=True)
 p.add_argument("--output",required=True)
 a=p.parse_args()
 output=ROOT/a.output
 if output.exists():raise FileExistsError(output)
 r,arrays=analyze(a.case)
 output.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n")
 print("V041_MEASURED_PHYSICAL_WHOLE_BODY",json.dumps({k:r[k] for k in (
    "case","real_physical_reference_goal_mse","real_physical_generated_goal_mse",
    "physical_proxy_mse_change","full_joint_actual_difference_peak_rad",
    "mean_absolute_robot_relation_change","original_goal","candidate_goal")}),flush=True)

if __name__=="__main__":main()
