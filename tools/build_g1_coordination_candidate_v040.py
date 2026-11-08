#!/usr/bin/env python3
"""v40: freeze one human-proxy-to-G1 whole-body reference candidate (no rollout).

Prespecified: 3DSP TRAIN clip 00001, 11 relative coordination relations,
source v32 center PhysX, q-root fixed to recorded source for OFFLINE IK,
strong support/foot-trajectory regularization, and a global low-amplitude
reference envelope to avoid collisions. It is *not* humanoid biomechanics.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
from g1_contact_coordination_projection_v040 import (
    TorchG1Relations,load_robot_source,interpolate_source,
    build_human_relative_target,soft_gate,project_trajectory,ROOT,URDF)
from g1_coordination_bridge_v039 import robot_relations

HUMAN=ROOT/"data/research_human_kicks/3dsp_cvpr2024/processed_v039/coordination_3dsp_v039.npz"
SOURCE=ROOT/"reports/v032_learned_center.json"

def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def build(path,config):
 if path.exists():raise FileExistsError(path)
 expected={
  "human_clip_id":"00001","human_original_split":"train",
  "source_v32_case":"center","raw_data_sha256":"304b0f56c8de089fd50a56fab8fc1e43a873a2430028b4e2d1488a69d90bd1a6"}
 if any(config.get(k)!=v for k,v in expected.items()):raise ValueError("Changing prespecified model source")
 if digest(HUMAN)!=config["raw_data_sha256"]:raise ValueError("Human dataset checksum drift")
 if digest(SOURCE)!=config["original_source_sha256"]:raise ValueError("Robot physical source changed")
 if digest(URDF)!=config["urdf_sha256"]:raise ValueError("G1 model checksum drift")
 human=np.load(HUMAN,allow_pickle=False)
 idx=np.flatnonzero(human["clip_id"]==config["human_clip_id"])
 if len(idx)!=1 or human["split"][idx[0]]!="train":
  raise ValueError("Human proxy selected from non-train or ambiguous source")
 assert not bool(human["verified_event_contact_truth"])
 human_signal=human["relation_cosines"][idx[0]].astype(np.float64)
 model_source=load_robot_source(SOURCE)
 fk=TorchG1Relations(model_source["joint_names"])
 tc=model_source["contact_time"]
 assert abs(tc-1.57)<1e-6,("Frozen physics event offset drift",tc)
 near=tc-.5+.04*np.arange(20)
 root,q=interpolate_source(model_source,near)
 import torch
 with torch.no_grad():
  base=fk(torch.tensor(root,dtype=torch.float64),
          torch.tensor(q,dtype=torch.float64)).numpy()
 target,hdelta=build_human_relative_target(base,human_signal,scale=config["source_relative_scale"])
 gate=soft_gate(near,tc,start_delay=.5,precontact_gap=.04)
 delta25,diagnostic=project_trajectory(fk,root,q,target,near,gate,
    iterations=config["optimizer_iterations"],
    lr=config["optimizer_lr"],max_rad=config["optimizer_max_rad"],
    foot_plant_weight=config["support_penalty"],
    smooth_weight=config["temporal_penalty"],joint_weight=config["joint_penalty"])
 # Sampled at the original 50Hz control schedule; zero outside original
 # bracket. Enforced global amplitude and slew bound avoids tuning per joint.
 delta50=np.stack([
    np.interp(model_source["times"],near,delta25[:,j],left=0.,right=0.)
    for j in range(29)],-1)
 delta50[(model_source["times"]<near[0]) | (model_source["times"]>tc-.04)] = 0.
 peak=float(np.max(abs(delta50)))
 step=float(np.max(abs(np.diff(delta50,axis=0))))
 scale=min(1.,
           config["max_applied_joint_rad"]/max(peak,1e-12),
           config["max_applied_joint_slew_rad"]/max(step,1e-12))
 delta50*=scale
 amp=float(np.max(abs(delta50)))
 slew=float(np.max(abs(np.diff(delta50,axis=0))))
 if amp>config["max_applied_joint_rad"]+1e-9 or slew>config["max_applied_joint_slew_rad"]+1e-9:
  raise ValueError("Output trajectory broke registered safety envelope")
 if np.count_nonzero(np.abs(delta50)>1e-6)<3:raise ValueError("No actual candidate generation")
 # Joint position constraints against original robot URDF
 g1_limits=np.asarray([next(j for j in fk.edges if j["name"]==name)["axis"] for name in model_source["joint_names"]])
 if g1_limits.shape!=(29,3):raise ValueError("29DoF source")
 import xml.etree.ElementTree as ET
 original_urdf=ET.parse(URDF).getroot()
 lim={}
 for j in original_urdf.findall("joint"):
  if j.attrib["type"]=="revolute":
   v=j.find("limit")
   lim[j.attrib["name"]]=(float(v.attrib["lower"]),float(v.attrib["upper"]))
 low=np.asarray([lim[n][0] for n in model_source["joint_names"]])
 high=np.asarray([lim[n][1] for n in model_source["joint_names"]])
 q_real=model_source["q"]+delta50
 out_count=int(np.sum((q_real<low-1e-5)|(q_real>high+1e-5)))
 source_count=int(np.sum((model_source["q"]<low-1e-5)|(model_source["q"]>high+1e-5)))
 if out_count>source_count:raise ValueError("Candidate adds q-limit breaches")
 # Original measured joint velocity limits are dynamic and not proven from
 # static q-reference delta; do not claim they pass before PhysX.
 output=path
 output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(output,delta_joint_position_rad=delta50.astype(np.float32),
       controller_time_s=model_source["times"].astype(np.float64),
       joint_names=np.asarray(model_source["joint_names"]),
       source_clip_id=np.asarray(config["human_clip_id"]),
       source_is_verified_instep=np.asarray(False),
       direct_human_to_joint_retarg=False,
       prior_is_robot_physics_policy=np.asarray(False),
       reference_contact_time_s=np.asarray(tc,dtype=np.float64),
       source_reference_case=np.asarray("center"),
       human_source_sha256=np.asarray(config["raw_data_sha256"]),
       source_robot_report_sha256=np.asarray(config["original_source_sha256"]))
 metadata={
   "version":"0.1.40",
   "human_clip":"00001","human_source_qualified":False,
   "original_robot_source":str(SOURCE.relative_to(ROOT)),
   "source_robot_sha256":config["original_source_sha256"],
   "event_alignment_warning":"3DSP source video shot annotation is not true contact; a human-to-robot time correspondence is only a study assumption",
   "target_relations_absolute_transfer":False,
   "trained_neural_generating_policy":False,
   "optimized_human_proxy_coordinate_changes_to_G1_q":True,
   "source_contact_time_s":tc,
   "robot_29dof_original":True,
   "original_urdf_sha256":config["urdf_sha256"],
   "source_human_relations_sha256":config["raw_data_sha256"],
   "target_delta_max":float(abs(hdelta).max()),
   "optimization":diagnostic,
   "preimpact_gate_min_max":list(map(float,(gate.min(),gate.max()))),
   "output_envelope":dict(global_scale=scale,max_joint_target_change_rad=amp,
       max_step_target_change_rad=slew,
       nonzero_control_frames=int((np.abs(delta50).max(axis=1)>1e-7).sum()),
       total_control_frames=len(delta50),
       optimized_25hz_samples=len(delta25),
       original_joint_limit_violations=source_count,
       generated_joint_limit_violations=out_count),
   "human_file_sha256":digest(HUMAN),
   "result_sha256":digest(output),
   "caveats":["Offline q references were optimized with frozen actual G1 root poses from successful prior control",
       "Runtime step actions and actual contacts require physical evaluation to establish feasibility",
       "No time-accurate human impact, energy, force or real human instep biomechanics available",
       "Identical 11 semantic names are not anatomical equivalence of 3DSP and G1",
       "Only a single human training video-clip prior, no unseen football task or new gait learned",
       "Full 14s PhysX acceptance needed to detect support/collision/actuator damage"]}
 m=output.parent/"generation_audit.json"
 if m.exists():raise FileExistsError(m)
 m.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
 print("V040_COORDINATION_REF_GENERATED",json.dumps({
  "urdf_fk_source":"validated","original_mse":diagnostic["coord_mse_baseline"],
  "projected_mse":diagnostic["coord_mse_generated"],
  "human_coordination_proxy_improvement":diagnostic["relative_improvement"],
  "max_ankle_support_move_m":diagnostic["foot_support_max_displacement_m"],
  "max_ankle_swing_move_m":diagnostic["foot_swing_max_displacement_m"],
  "global_scale":scale,"max_rad":amp,"step_rad":slew,
  "nonzero_control_frames":metadata["output_envelope"]["nonzero_control_frames"],
  "reference_sha256":metadata["result_sha256"]},ensure_ascii=False),flush=True)
 return metadata

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--config",default="configs/g1_coordination_generation_v040.json")
 p.add_argument("--output",default="experiments/coordination_projection_v040/train_clip00001_center/g1_joint_delta_50hz.npz")
 args=p.parse_args()
 config=json.loads((ROOT/args.config).read_text())
 for f,h in config["source_sha256"].items():
  if digest(ROOT/f)!=h:raise ValueError("Frozen study source drift: "+f)
 build(ROOT/args.output,config)
