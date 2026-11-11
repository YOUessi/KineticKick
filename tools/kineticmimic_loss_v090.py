#!/usr/bin/env python3
"""KineticMimic v90: genuinely differentiable contact+coordination+impact losses.

Separates three truth domains:
  HUMAN optical six segment-cosine trajectories: geometric style ONLY.
  ROBOT PhysX measured G1 support, body q/dq, foot-ball impulse, ball velocity.
  EXPERIMENTAL phase-order hypothesis: *not* measured elite kinetics.

No missing human GRF or true contact can be used as a supervised impact label.
An optimization loss can encourage a motion but NEVER certifies safety:
original 29DoF physical velocities, collider/nonfoot, instep>=0.95, goal and
14s standing are scored independently from real measured G1 PhysX episodes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

JOINTS=G1_JOINT_ORDER
GROUPS={
 "stance":("left_hip_pitch_joint","left_knee_joint","left_ankle_roll_joint"),
 "hip":("right_hip_pitch_joint","left_hip_yaw_joint","waist_yaw_joint"),
 "trunk":("waist_yaw_joint","waist_pitch_joint","waist_roll_joint"),
 "swing":("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint"),
 "counter_arm":("left_shoulder_pitch_joint","right_shoulder_pitch_joint",
                "left_shoulder_roll_joint","right_shoulder_roll_joint",
                "left_elbow_joint","right_elbow_joint"),
}
IDS={k:tuple(JOINTS.index(n) for n in v) for k,v in GROUPS.items()}

@dataclass(frozen=True)
class TruthScope:
 human_120hz_optical:bool=True
 human_true_foot_ball_contact:bool=False
 human_synchronized_GRF:bool=False
 human_elite_instep_kinetics:bool=False
 robot_actual_PhysX_contact:bool=True
 robot_installed_native_collider_pair_certified:bool=False
 human_subject_split_independently_qualified:bool=False

def validate_scopes(scope:TruthScope):
 if (not scope.human_120hz_optical or
     scope.human_true_foot_ball_contact or
     scope.human_synchronized_GRF or
     scope.human_elite_instep_kinetics or
     not scope.robot_actual_PhysX_contact):
  raise ValueError("Unqualified human football contact force / professional kinetics")
 # A native collider witness is not present in these sources. Native pair
 # absent doesn't logically imply no collision. No fake geometry certificate.
 if scope.robot_installed_native_collider_pair_certified:
  raise ValueError("Current original G1 GPU native collider pair has NOT been certified")

def _finite(x,name,shape=None):
 if not isinstance(x,torch.Tensor) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):
  raise ValueError(name+" must be a finite torch float tensor")
 if shape is not None and tuple(x.shape)!=tuple(shape):
  raise ValueError("Invalid "+name+" tensor shape "+str(tuple(x.shape)))
 return x

def soft_dtw_path_cost(x,y,gamma=.05):
 """Differentiable monotone time-warp alignment of relative BODY GEOMETRY.

 x: [B,Tx,6] original G1 6D FK from 29DoF; y: [B,Ty,6]
 human optical relative 6D. NO true contact time or ground impulse label.
 """
 x=_finite(x,"robot 6D segments")
 y=_finite(y,"human 6D segments")
 if (x.ndim!=3 or y.ndim!=3 or x.shape[0]!=y.shape[0] or
     x.shape[-1]!=6 or y.shape[-1]!=6 or
     not (4<=x.shape[1]<=60 and 4<=y.shape[1]<=60) or
     not 0<gamma<=1):
  raise ValueError("Only qualified finite BxTx6 body geometries and monotone soft-DTW")
 B,A,D=x.shape;C=y.shape[1]
 squared=torch.sum((x[:,:,None,:]-y[:,None,:,:])**2,dim=-1)/6.
 high=torch.full((B,),float("inf"),dtype=x.dtype,device=x.device)
 zero=torch.zeros_like(high)
 past=[zero]+[high]*C
 for i in range(A):
  cur=[high]
  for j in range(C):
   options=torch.stack((past[j+1],cur[j],past[j]),dim=-1)
   softmin=-gamma*torch.logsumexp(-options/gamma,dim=-1)
   cur.append(squared[:,i,j]+softmin)
  past=cur
 result=past[-1]/(A+C)
 if not bool(torch.isfinite(result).all()):
  raise FloatingPointError("Invalid optical body soft-DTW result")
 return result

def semantic_phase_cost(robot_six,human_six,gamma=.05):
 """Enforce *relative changes*, not absolute anatomical pose identity."""
 robot=_finite(robot_six,"robot semantic 6D")
 human=_finite(human_six,"human optical six-body channels")
 if robot.shape!=human.shape:raise ValueError("Human/G1 relative semantic channel mismatch")
 if torch.any(abs(robot)>1.01) or torch.any(abs(human)>1.01):
  raise ValueError("Human 6D cosine proxy is not normalized")
 # Skip exact anatomy: match relative changes from earliest phase, then allow
 # monotone temporal shift. Not a human true-impact imitation loss.
 a=robot-robot[:,:1]
 b=human-human[:,:1]
 return soft_dtw_path_cost(a,b,gamma=gamma).mean()

def soft_phase_peak(dq,indices,temperature=.04):
 """Expected normalized peak phase, avoiding hard non-differentiable argmax."""
 arr=_finite(dq,"actual/proposed 29DoF rate")
 if arr.ndim!=3 or arr.shape[-1]!=29 or arr.shape[1]<4 or not (0<temperature<=1):
  raise ValueError("Measured 29DoF speed trace shape or phase temperature invalid")
 power=torch.sqrt(torch.sum(arr[:,:,indices]**2,dim=-1)+1e-9)
 # Each group normalization per episode allows us to compare relative timing
 # rather than absolute angle velocity or body mass/muscle power.
 power=power/(power.amax(-1,keepdim=True).clamp_min(1e-6))
 phase=torch.linspace(0.,1.,arr.shape[1],device=arr.device,dtype=arr.dtype)
 prob=torch.softmax(power/temperature,dim=-1)
 return torch.sum(prob*phase[None],dim=-1)

def phase_chain_hypothesis_loss(dq,margin=.04):
 """FALSIFIABLE hypothesis only; never claimed a measured human firing order.

 Requires stance load->hip->trunk->leg; contralateral arm counter-motion
 temporally close to trunk. Does NOT estimate mechanical power or true GRF.
 """
 if not (0<=margin<=.15):raise ValueError("Invalid proposed timing margin")
 peak={k:soft_phase_peak(dq,v) for k,v in IDS.items()}
 e=(F.relu(peak["stance"]+margin-peak["hip"])**2+
    F.relu(peak["hip"]+margin-peak["trunk"])**2+
    F.relu(peak["trunk"]+margin-peak["swing"])**2+
    torch.abs(peak["counter_arm"]-peak["trunk"])*.25)
 return e.mean(),{k:v.detach().cpu().tolist() for k,v in peak.items()}

def contact_prediction_loss(stance_logits,impact_hazard_logits,
                            true_future_stance,true_future_impact):
 """Supervised ONLY by actual robot PhysX 200Hz ground/contact event labels.

 stance logits: [B,4] for airborne/left/right/double support 100ms ahead.
 impact logits: [B] for measured first foot-ball contact within next 100ms.
 """
 z=_finite(stance_logits,"stance logits")
 h=_finite(impact_hazard_logits,"impact hazard logits")
 y=torch.as_tensor(true_future_stance,device=z.device,dtype=torch.long)
 impact=_finite(torch.as_tensor(true_future_impact,device=z.device,dtype=z.dtype),"actual robot impact supervision")
 if z.ndim!=2 or z.shape[1]!=4 or h.shape!=(len(z),) or y.shape!=(len(z),) or impact.shape!=(len(z),):
  raise ValueError("Original PhysX stance/impact supervision shape mismatch")
 if torch.any((y<0)|(y>3)) or torch.any((impact<0)|(impact>1)):
  raise ValueError("Invalid real 4-state support or foot-ball contact label")
 # Class imbalance is handled by TRAIN-SPLIT-only sample weighting in trainer.
 return F.cross_entropy(z,y)+F.binary_cross_entropy_with_logits(h,impact)

def impact_prediction_loss(predicted_ball_delta_v_xyz,measured_ball_delta_v_xyz,
                           predicted_foot_ball_scalar_impulse_Ns,
                           measured_foot_ball_scalar_impulse_Ns,physx_contact_mask):
 """Robot world-impact labels ONLY, not human contact force from optical video."""
 v=_finite(predicted_ball_delta_v_xyz,"predicted robot ball delta v")
 actual=_finite(measured_ball_delta_v_xyz,"actual robot ball delta v")
 p=_finite(predicted_foot_ball_scalar_impulse_Ns,"predicted absolute scalar impulse")
 t=_finite(measured_foot_ball_scalar_impulse_Ns,"observed absolute scalar impulse")
 m=torch.as_tensor(physx_contact_mask,device=v.device,dtype=torch.bool)
 B=len(v)
 if v.shape!=(B,3) or actual.shape!=(B,3) or p.shape!=(B,) or t.shape!=(B,) or m.shape!=(B,):
  raise ValueError("Measured original robot impact tensor/mask shape mismatch")
 if not bool(m.any()):
  raise ValueError("Cannot train impact from absent G1 PhysX force/contact window")
 if (t[m]<0).any():raise ValueError("Original robot contact impulse magnitude not physical")
 return F.smooth_l1_loss(v[m],actual[m])+F.smooth_l1_loss(p[m],t[m])

class ContactModeHead(nn.Module):
 """Feasibility component: predict FUTURE support and first impact from 60ms history.

 An explicit NON-ACTION model. It learns neither human professional power nor
 robot actor policy. Feed to a separate contact-conditioned Mimic *later* only
 after independent data, live collision witness and new G1 physics evaluation.
 """
 def __init__(self,hidden=48,features=71,history=12):
  super().__init__()
  if features!=71 or history!=12 or hidden<8:raise ValueError("Original G1 71D x12 contact mode schema")
  self.history=history
  self.recurrent=nn.GRU(features,hidden,num_layers=1,batch_first=True)
  self.stance_head=nn.Linear(hidden,4)
  self.impact_head=nn.Linear(hidden,1)
 def forward(self,x):
  v=_finite(x,"G1 contact mode prehistory")
  if v.ndim!=3 or v.shape[1:]!=(self.history,71):
   raise ValueError("Only Bx12x71 original 200Hz stance context accepted")
  _,h=self.recurrent(v)
  hidden=h[-1]
  return self.stance_head(hidden),self.impact_head(hidden).flatten()

@dataclass(frozen=True)
class CombinedLoss:
 geometry:float=1.0
 phase_chain:float=.4
 contact_mode:float=1.0
 impact:float=.8

def kineticmimic_multiobjective(geometry_robot_six,geometry_human_six,
     real_or_predicted_29dof_rates,contact_logits,impact_logits,
     actual_future_stance,actual_future_impact,impact_predictions,
     loss_weights=CombinedLoss(),human_truth_scope=TruthScope()):
 validate_scopes(human_truth_scope)
 if not isinstance(impact_predictions,dict):raise ValueError("Missing separately measured robot impact evidence")
 geometry=semantic_phase_cost(geometry_robot_six,geometry_human_six)
 chain,phases=phase_chain_hypothesis_loss(real_or_predicted_29dof_rates)
 contact=contact_prediction_loss(contact_logits,impact_logits,
                                actual_future_stance,actual_future_impact)
 impact=impact_prediction_loss(
   impact_predictions["predicted_ball_delta_v_xyz"],
   impact_predictions["actual_ball_delta_v_xyz"],
   impact_predictions["predicted_scalar_foot_ball_impulse_Ns"],
   impact_predictions["actual_scalar_foot_ball_impulse_Ns"],
   impact_predictions["has_true_robot_PhysX_contact"])
 out=loss_weights.geometry*geometry+loss_weights.phase_chain*chain+loss_weights.contact_mode*contact+loss_weights.impact*impact
 if not torch.isfinite(out):raise FloatingPointError("Unqualified KineticMimic loss")
 return out,dict(geometry=geometry,phase_chain=chain,contact=contact,robot_impact=impact,phase_proxy=phases,
   human_contact_or_GRF_training_claim=False,physical_hardware_safety_certificate=False)
