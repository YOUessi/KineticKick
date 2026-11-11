#!/usr/bin/env python3
"""v64 real-CMU fitted phase source -> original G1 29DoF PhysX controller.

This controller is a *limited, within-one-person CMU geometric phase prior*
fit from one 120Hz optically tracked source, not expert professional kick
biomechanics, learned policy, human force transfer or guaranteed collision
clearance. Original v32 balance recovery, G1 PD and ball contact are intact.
"""
import hashlib,math
from pathlib import Path
import numpy as np
import torch
from post_shot_recovery import PostShotRecoveryTask
from running_balance import ground_force
from phase_coordinated_v063 import admissibility
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT=Path(__file__).resolve().parents[1]
MAX_RAD=.055
SLEW=.004
FIRST=1.02
LAST=1.40
RELEASE=1.51
def sha(file):
 h=hashlib.sha256()
 with file.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

class HumanPhaseTask(PostShotRecoveryTask):
 def __init__(self,env,config):
  p=config.get("human_phase_v064")
  if not isinstance(p,dict) or p.get("enabled") is not True:
   raise ValueError("Human phase source must be explicitly frozen and enabled")
  if p.get("source")!="CMU_120Hz_optical_10_02_one_subject_only" or p.get("human_true_contact") is not False:
   raise ValueError("Reject fabricated human instep/impact ground truth")
  if p.get("sequence") not in ("human_ordered","human_reversed"):
   raise ValueError("Human temporal prior mode must be one of the two frozen versions")
  if p.get("max_joint_rad")!=MAX_RAD or p.get("max_step_rad")!=SLEW or p.get("active_window_s")!=[FIRST,LAST]:
   raise ValueError("Frozen original 29DoF human phase safety envelope drift")
  if config.get("phase_coordinated_v063",{}).get("enabled") or config.get("human_coordination_reference",{}).get("enabled") or config.get("coordination_style_v041",{}).get("enabled"):
   raise ValueError("Do not combine synthetic v63 and old weak v41 with real CMU")
  super().__init__(env,config)
  if tuple(self.robot.joint_names)!=G1_JOINT_ORDER:
   raise ValueError("Invalid G1 semantic joint order for live physical control")
  file=ROOT/p["reference"]
  if sha(file)!=p["reference_sha256"]:raise ValueError("Fitted measured human reference has been altered")
  with np.load(file,allow_pickle=False) as z:
   ref=z["delta_joint_position_rad"].astype(np.float64)
   ts=z["controller_time_s"]
   names=z["joint_names"].tolist()
   cmu_clip=str(z["real_human_training_clip"].item())
   human_impact=bool(z["human_optical_motion_event_not_contact"].item())
   source_instep=bool(z["source_is_verified_instep"].item())
  if ref.shape!=(700,29) or len(ts)!=700 or not np.allclose(ts,np.arange(700)*.02,rtol=0,atol=1e-5):
   raise ValueError("Real physical G1 700x29 20ms source only")
  if names!=list(G1_JOINT_ORDER) or cmu_clip!="10_02" or not human_impact or source_instep:
   raise ValueError("Original source joint names or human provenance cannot be verified")
  if not np.isfinite(ref).all() or abs(ref).max()>MAX_RAD+1e-5 or max(abs(np.diff(ref,axis=0)).ravel())>SLEW+1e-5:
   raise ValueError("Trained human reference breaks original registered action limit")
  if np.any(abs(ref[ts>=1.42-1e-9])>1e-8) or np.any(abs(ref[ts<FIRST-1e-9])>1e-8):
   raise ValueError("Human phase action must release before real robot impact")
  # Right leg G1 right hip/knee/ankle must remain ORIGINAL throughout the
  # pre-impact window, because impact geometry is a hard functional constraint.
  if not np.all(ref[:,[1,4,7,10,14,18]]==0):
   raise ValueError("Original right striking leg joint setpoint not protected")
  self.human_cfg=p
  self.human_ref=torch.tensor(ref,device=self.device,dtype=torch.float32)
  self.human_applied=torch.zeros(self.n,29,device=self.device)
  self.human_previous=self.human_applied.clone()
  self.human_action_steps=0
  self.human_q_peak=0.
  self.human_eligible_steps=0
  self.human_gate_true={key:0 for key in ("window","stance","qdot","ball","wrist_hip_origin_proxy","nonfoot_force","pre_contact","joint_position_limits")}
 def reset_fields(self,ids):
  super().reset_fields(ids)
  self.human_applied[ids]=0.
  self.human_previous[ids]=0.
  self.human_action_steps=0
  self.human_q_peak=0.
  self.human_eligible_steps=0
  for key in self.human_gate_true:self.human_gate_true[key]=0
 def commands(self,actions=None):
  if actions is not None:raise ValueError("This is a frozen human-derived time reference, not external policy actions")
  base=super().commands(None)
  self.human_previous=self.human_applied.clone()
  now=self.elapsed*.02
  r=self.robot.data
  support=ground_force(self.env)
  qdot_ratio=(r.joint_vel/r.joint_vel_limits).abs().amax(-1)
  ball_dist=torch.linalg.vector_norm(self.ball.data.root_pos_w[:,:2]-r.root_pos_w[:,:2],dim=-1)
  wrist=r.body_pos_w[:,self.robot.body_names.index("left_wrist_yaw_link")]
  hip=r.body_pos_w[:,self.robot.body_names.index("left_hip_roll_link")]
  separation=torch.linalg.vector_norm(wrist-hip,dim=-1)
  sensor=self.env.scene["contact_forces"]
  indices=[i for i,n in enumerate(sensor.body_names) if n not in ("left_ankle_roll_link","right_ankle_roll_link")]
  nonfoot=sensor.data.net_forces_w[:,indices].norm(dim=-1).amax(-1)
  limits=r.soft_joint_pos_limits
  ok,flags=admissibility(now,support,qdot_ratio,ball_dist,separation,self.contact,nonfoot,r.joint_pos,limits)
  ok &= (now>=FIRST)&(now<=LAST)&(~self.contact)
  for key, flag in flags.items():self.human_gate_true[key]+=int(flag.sum().item())
  self.human_eligible_steps+=int(ok.sum().item())
  frame=self.elapsed.long().clamp(0,699)
  goal=torch.where(ok[:,None],self.human_ref[frame],torch.zeros_like(self.human_applied))
  correction=torch.clamp(goal-self.human_applied,-SLEW,SLEW)
  self.human_applied=torch.clamp(self.human_applied+correction,-MAX_RAD,MAX_RAD)
  # Zero late action with a bounded slew rather than a reference cliff.
  if bool((now>=RELEASE).any()) and bool((self.human_applied.abs()>1e-6).any()):
   raise RuntimeError("Human phase did not decay to zero safely before original foot-ball impact")
  scale=torch.as_tensor(self.env.action_manager.get_term("joint_pos")._scale,device=self.device,dtype=base.dtype)
  if not bool(torch.isfinite(scale).all()) or bool((scale==0).any()):
   raise ValueError("Original 29DoF motor action scaling invalid")
  action=base+self.human_applied/scale
  self.human_applied=(action-base)*scale
  active=bool((self.human_applied.abs()>1e-5).any())
  if active:self.human_action_steps+=1
  self.human_q_peak=max(self.human_q_peak,float(self.human_applied.abs().max()))
  if self.n==1 and self.debug:
   self.debug[-1]["human_phase_v064"]=dict(
    enabled=True,sequence=self.human_cfg["sequence"],source_clip="CMU10_02",
    was_in_original_safety_window=bool(ok[0]),gate_flags={k:bool(v[0]) for k,v in flags.items()},
    measured_qdot_ratio=float(qdot_ratio[0]),
    measured_left_wrist_hip_link_origin_distance_m=float(separation[0]),
    measured_nonfoot_N=float(nonfoot[0]),
    applied_delta_rad=self.human_applied[0].tolist(),
    previous_delta_rad=self.human_previous[0].tolist(),
    maximum_delta_rad=float(self.human_applied[0].abs().max()),
    original_controller_action=base[0].tolist(),
    resulting_controller_action=action[0].tolist(),
    ball_state_write=False,robot_state_write=False,
    human_foot_ball_contact_truth=False,original_right_leg_strike_geometry_protected=True)
  return action

def get_task(env):
 if not hasattr(env,"_human_phase_v064_task"):
  env._human_phase_v064_task=HumanPhaseTask(env,env.cfg.joint_strike)
 return env._human_phase_v064_task
