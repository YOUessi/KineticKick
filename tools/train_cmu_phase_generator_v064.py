#!/usr/bin/env python3
"""v64: original 120Hz CMU soccer phase fit + local G1 29DoF constrained IK.

This is a deterministic TWO-clip single-subject exploratory *motion prior*
train on ONE clip 10_02. A temporally reversed control is negative ablation.
The source does NOT give verified impact/contact, professional skill, muscle
power, or GRF; a fitted 6-cosine signal is NOT human biomechanics. Output is
an OFFLINE reference only, never a robot policy result without PhysX.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from cmu_g1_semantic_bridge_v051 import ROOT,CMU,human_six_timed
from build_cmu_g1_candidate_v052 import g1_six_differentiable,joint_limits
from g1_contact_coordination_projection_v040 import TorchG1Relations,load_robot_source,interpolate_source
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

OUTPUT=ROOT/"experiments/cmu_phase_fit_v064"
ROBOT=ROOT/"reports/v063_center_v32.json"
HUMAN_SOURCE=ROOT/CMU/"quality_audit.json"
CONFIG=ROOT/"configs/cmu_phase_model_v064.json"

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for s in iter(lambda:f.read(1<<20),b""):h.update(s)
 return h.hexdigest()

def phase_features(t):
 """Smooth local regression basis, conditioning numbers and source fixed."""
 u=np.asarray(t,dtype=float)
 return np.column_stack((np.ones(len(u)),u,*[
  np.exp(-.5*((u-c)/.20)**2) for c in np.linspace(0.,1.,7)]))

def fit_human(relative,human,regularizer=.025):
 if human.shape!=(20,6) or not np.isfinite(human).all():
  raise ValueError("Need exactly 20 measured 6D CMU human phase samples")
 u=np.linspace(0,1,20)
 basis=phase_features(u)
 target=human-human[:1]
 penalty=np.diag([0.]+[regularizer]*(basis.shape[1]-1))
 coeff=np.linalg.solve(basis.T@basis+penalty,basis.T@target)
 fit=basis@coeff
 fit=fit-fit[:1]
 if not np.isfinite(fit).all():raise ValueError("Nonfinite trained human phase")
 return fit,coeff,dict(source_relative_pose_change_rms=float(np.sqrt(np.mean(target**2))),
  in_sample_6d_mean_squared_fit_error=float(np.mean((fit-target)**2)),
  max_observed_pose_change_cos=float(np.max(abs(target))),
  max_fitted_pose_change_cos=float(np.max(abs(fit))),
  source_human_count=1,coeff_shape=list(coeff.shape),
  no_independent_human_generalization=True)

def make_reference(robot,human_fit,config):
 relative=np.arange(-.40,0,.02).round(8)
 times=np.arange(1.02,1.401,.02).round(8)
 if len(times)!=20:raise ValueError("Time alignment drift")
 r,q=interpolate_source(robot,times)
 fk=TorchG1Relations(robot["joint_names"])
 X=torch.tensor(r,dtype=torch.float64)
 Q=torch.tensor(q,dtype=torch.float64,requires_grad=True)
 curve=g1_six_differentiable(fk,X,Q)
 base=curve.detach().numpy()
 jac=[]
 for j in range(6):
  g=torch.autograd.grad(curve[:,j].sum(),Q,retain_graph=(j!=5))[0]
  jac.append(g.detach().numpy())
 jac=np.stack(jac,axis=1)  # [20,6,29]
 selected=tuple(config["movable_G1_indices"])
 all29=set(range(29))
 if not set(config["frozen_right_swing_leg_G1_indices"]).issubset(all29-set(selected)):
  raise ValueError("Protect original right striking leg throughout entire prior")
 if tuple(robot["joint_names"])!=G1_JOINT_ORDER:raise ValueError("Wrong G1 semantically locked 29DoF")
 w=np.asarray(config["six_feature_weights"])
 if w.shape!=(6,) or np.any(w<0) or np.sum(w)<=0:raise ValueError("Missing six named human geometry weights")
 envelope=np.sin(np.linspace(0,np.pi,20))**2
 target=np.clip(config["relative_transfer_gain"]*human_fit*envelope[:,None],-.25,.25)
 delta=np.zeros_like(q)
 from scipy.linalg import solve
 for i in range(20):
  M=jac[i,:,selected] if False else jac[i][:,list(selected)]
  # Damped morphology-aware local least squares, underdetermined 6D->20DoF.
  W=np.sqrt(w)[:,None]
  A=M*W
  b=target[i]*W[:,0]
  ATA=A.T@A+config["ik_ridge"]*np.eye(len(selected))
  sol=solve(ATA,A.T@b,assume_a="pos")
  delta[i,list(selected)]=sol
 # The CMU signal changes shape; finite impulses and stable pose are enforced
 # in source 50Hz robot joint reference, not inferred human feet contacts.
 for j in selected:
  delta[:,j]=np.convolve(np.pad(delta[:,j],(1,1),mode="edge"),
                       [0.25,0.5,0.25],mode="valid")
 delta*=envelope[:,None]
 first=np.max(abs(delta))
 if first<1e-8:raise ValueError("No meaningful human-sourced 29DoF reference")
 delta*=min(1.,config["max_joint_delta_rad"]/first)
 # Enforce absolute per-20ms time slew globally, rather than per-phase retune.
 maximum_step=float(np.max(abs(np.diff(delta,axis=0))))
 delta*=min(1.,config["max_slew_per_20ms_rad"]/max(maximum_step,1e-12))
 lowerupper=joint_limits(robot["joint_names"])
 margin_lo=lowerupper[:,0]+.02
 margin_hi=lowerupper[:,1]-.02
 # Original true G1 trace may itself start inside the 0.02rad margin (or
 # slightly exceed a URDF soft bound). Never demand the immutable baseline
 # to be strictly interior; permit motion toward safety, not farther outside.
 lower_delta=np.where(q<margin_lo,0.,margin_lo-q)
 upper_delta=np.where(q>margin_hi,0.,margin_hi-q)
 delta=np.minimum(np.maximum(delta,lower_delta),upper_delta)
 qnew=q+delta
 source_margin=(q<margin_lo)|(q>margin_hi)
 generated_margin=(qnew<margin_lo-1e-10)|(qnew>margin_hi+1e-10)
 if np.any(generated_margin & ~source_margin):
  raise ValueError("Candidate adds original G1 q-margin breach")
 with torch.no_grad():
  _,orig_links=fk(X,Q.detach(),with_links=True)
  def foot_devs(inp):
   _,links=fk(X,torch.tensor(q+inp,dtype=torch.float64),with_links=True)
   return [float(torch.linalg.vector_norm(
       links[k][:,:3,3]-orig_links[k][:,:3,3],dim=-1).max())
       for k in ("left_ankle_roll_link","right_ankle_roll_link")]
  prev=foot_devs(delta)
  norm=min(1.,config["max_support_ankle_shift_m"]/max(prev[0],1e-12),
              config["max_swing_ankle_shift_m"]/max(prev[1],1e-12))
  delta*=norm
  final=foot_devs(delta)
  attained=g1_six_differentiable(fk,X,torch.tensor(q+delta,dtype=torch.float64)).numpy()
 oldmse=float(np.mean((base-(base+target))**2))
 newmse=float(np.mean((attained-(base+target))**2))
 if final[0]>config["max_support_ankle_shift_m"]+1e-8 or final[1]>config["max_swing_ankle_shift_m"]+1e-8:
  raise ValueError("Foot trajectory geometry safety failed")
 out=np.zeros((700,29),np.float64)
 grid=robot["times"]
 for j in selected:
  out[:,j]=np.interp(grid,times,delta[:,j],left=0.,right=0.)
 out[(grid<times[0])|(grid>=1.42-1e-6)]=0.
 # Ensure interpolation endpoint is driven to 0 and the new reference releases
 # safely BEFORE any anticipated first real G1 foot-ball force contact.
 if max(abs(np.diff(out,axis=0)).ravel())>config["max_slew_per_20ms_rad"]+1e-6:
  raise ValueError("Interpolated 50Hz reference violates original bound")
 if max(abs(out).ravel())>config["max_joint_delta_rad"]+1e-6:
  raise ValueError("Reference joint amplitude violated")
 if np.count_nonzero(np.max(abs(out),axis=1)>1e-6)<5:
  raise ValueError("Human phase prior would be a null motion")
 return out.astype(np.float32),dict(
   offline_6d_target_mse_before=oldmse,offline_6d_target_mse_after=newmse,
   original_step_peak=float(max(abs(out).ravel())),
   max_reference_step_rad=float(max(abs(np.diff(out,axis=0)).ravel())),
   active_20ms_control_frames=int(np.count_nonzero(np.max(abs(out),axis=1)>1e-6)),
   nonzero_joint_count=int(np.count_nonzero(np.max(abs(out),axis=0)>1e-6)),
   predicted_ankle_origin_shift_m=final,
   original_G1_URDF_differentiable_fk=True,
   real_human_impact_groundtruth=False)

def main():
 if CONFIG.exists():raise RuntimeError("v64 model config should only be created after source code")
 # The outer command writes the versioned configuration before invocation.
def train():
 cfg=json.loads(CONFIG.read_text())
 if cfg["version"]!="0.1.64" or cfg["human_fit_train_clips"]!=["10_02"] or cfg["human_contact_groundtruth"]:
  raise ValueError("Wrong frozen one-subject human source and qualification")
 for p,expected in cfg["frozen_source_sha256"].items():
  if sha(ROOT/p)!=expected:raise ValueError("Human / source G1 original sha drift: "+p)
 if OUTPUT.exists():raise FileExistsError("Generated v64 human prior immutable; require another version for retrain")
 human=json.loads(HUMAN_SOURCE.read_text())
 relative=np.arange(-.40,0,.02).round(8)
 measured,anchor=human_six_timed("10_02",relative,human)
 fitted,coeff,diagnostics=fit_human(relative,measured,cfg["human_ridge"])
 robot=load_robot_source(ROBOT)
 candidate,quality=make_reference(robot,fitted,cfg)
 # Negative-control reverse the frozen human temporal changes without changing
 # the source amplitude budget, original robot or trained coefficients.
 backwards=candidate.copy()
 ix=np.where(np.max(abs(candidate),axis=1)>1e-10)[0]
 if len(ix)<10:raise ValueError("Human reference too short to reverse")
 backwards[ix]=candidate[ix[::-1]]
 # A temporal reversal may violate per-20ms step envelope; reject instead
 # of secretly weakening it. Smooth clock profile through a fixed 3-tap kernel.
 if max(abs(np.diff(backwards,axis=0)).ravel())>cfg["max_slew_per_20ms_rad"]+1e-6:
  raise ValueError("Pre-registered negative-control shuffled reference exceeds slew")
 OUTPUT.mkdir(parents=True)
 refs={}
 for name,signal in [("human_ordered",candidate),("human_reversed",backwards)]:
  dest=OUTPUT/(name+".npz")
  np.savez_compressed(dest,delta_joint_position_rad=signal,
      controller_time_s=robot["times"].astype(np.float64),
      joint_names=np.asarray(robot["joint_names"]),
      real_human_training_clip=np.asarray("10_02"),
      human_optical_motion_event_not_contact=True,
      source_is_verified_instep=False,
      independent_human_test=False,
      source_human_sha256=np.asarray(cfg["frozen_source_sha256"]["data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050/10_02.npz"]))
  refs[name]=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),
    actual_precontact_reference_frames=int(np.count_nonzero(np.max(abs(signal),axis=1)>1e-6)),
    max_rad=float(max(abs(signal).ravel())))
 report=dict(version="0.1.64",source_subjects=1,train_human_clip="10_02",
   real_human_120hz_optical_source=True,human_impact_time_certified=False,
   professional_high_power_instestep_truth=False,new_policy_trained=False,
   learned_phase_spline_coefficients_shape=list(coeff.shape),
   human_training_diagnostics=diagnostics,
   G1_constrained_reference_diagnostics=quality,
   original_robot_source_sha256=cfg["frozen_source_sha256"]["reports/v063_center_v32.json"],
   generated_references=refs,no_heldout_player_or_groundtruth_contact=True)
 (OUTPUT/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
 print("V064_REAL_CMU_PHASE_TRAINED",json.dumps(dict(train_clip="10_02",references=refs,
     training_fit_MSE=diagnostics["in_sample_6d_mean_squared_fit_error"],
     IK_original=quality["offline_6d_target_mse_before"],IK_new=quality["offline_6d_target_mse_after"],
     active_frames=quality["active_20ms_control_frames"]),ensure_ascii=False),flush=True)
if __name__=="__main__":train()
