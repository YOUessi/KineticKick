#!/usr/bin/env python3
"""KineticMimic v90 exploratory PhysX contact-conditioning head training.

Predict the 100ms-ahead support class, first future foot-ball contact and
robot-only PhysX ball delta-v/unsigned impulse from PAST 60ms 200Hz G1 state.
Train original G1 v84 scene-C, choose on v84 D, inspect v85 center only AFTER
both equally budgeted candidate models selected. Repeated policy variants in
the same physical context are NOT independent humanoid trials.

This small contact head is NOT a policy driving joints. It demonstrates the
actual learned Contact/Impact part of a future physics-in-the-loop Mimic method;
full-body G1 generation and professional human contact are still unsolved.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from kineticmimic_physics_v090 import (
 load_original_physics,sample_contact_predictor,strict_physics,sha)
from kineticmimic_loss_v090 import contact_prediction_loss
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT=Path(__file__).resolve().parents[1]
class ContactImpactHead(nn.Module):
 def __init__(self,hidden=48):
  super().__init__()
  if hidden<16 or hidden>256:raise ValueError("No out-of-contract actor size")
  self.gru=nn.GRU(71,hidden,num_layers=2,batch_first=True,dropout=.10)
  self.stance=nn.Linear(hidden,4)
  self.hazard=nn.Linear(hidden,1)
  self.delta_v=nn.Linear(hidden,3)
  self.scalar_impulse=nn.Linear(hidden,1)
 def forward(self,x):
  if x.ndim!=3 or x.shape[1:]!=(12,71) or not bool(torch.isfinite(x).all()):
   raise ValueError("Real original-G1 Bx12x71 sensor history only")
  _,h=self.gru(x)
  f=h[-1]
  return self.stance(f),self.hazard(f)[:,0],self.delta_v(f),F.softplus(self.scalar_impulse(f)[:,0])

def physical_outcome(episode):
 if episode.first_impact_s is None:
  return np.zeros(3,np.float32),0.
 t=episode.times
 i=np.flatnonzero(t>=episode.first_impact_s-.0001)[0]
 j=min(len(t)-1,i+20)
 delta=(episode.ball_velocity_xyz[j]-episode.ball_velocity_xyz[max(0,i-1)])
 if np.linalg.norm(delta)>20:
  raise ValueError("Unreasonable ball velocity impulse surrogate")
 return delta.astype(np.float32),float(episode.estimated_foot_ball_scalar_impulse_Ns)

def rows_to_training(episodes,cfg):
 x=[];stance=[];impact=[];v=[];J=[];source=[];current=[]
 for e in episodes:
  batch=sample_contact_predictor(e,history=12,stride=4,horizon=20,precontact_only=True)
  n=len(batch["x"]);V,I=physical_outcome(e)
  x.append(batch["x"])
  stance.append(batch["stance_after_100ms"])
  impact.append(batch["impact_next_100ms"])
  v.append(np.repeat(V[None],n,axis=0))
  J.append(np.full(n,I,np.float32))
  source.extend([e.case]*n)
  current.append(batch["current_stance"])
 if not x:raise ValueError("No measured G1 PhysX data available to train")
 return dict(x=np.concatenate(x).astype(np.float32),
   stance=np.concatenate(stance),impact=np.concatenate(impact).astype(np.float32),
   delta_v=np.concatenate(v).astype(np.float32),impulse=np.concatenate(J).astype(np.float32),
   source=source,current_stance=np.concatenate(current))

def normalize_training(train,val,test):
 mu=train["x"].reshape(-1,71).mean(0).astype(np.float32)
 sigma=np.maximum(train["x"].reshape(-1,71).std(0),.05).astype(np.float32)
 for item in (train,val,test):
  item["x"]=np.clip((item["x"]-mu)/sigma,-12.,12.).astype(np.float32)
 return mu,sigma

def loss_for_network(model,batch,idx,pos_weight,shuffle_mode=False,rng=None):
 x=torch.from_numpy(batch["x"][idx]).clone()
 if shuffle_mode:
  # Order removed BEFORE each model forward, on train AND evaluation.
  # Same 12 x 71 values remain, target is original 100ms future contact.
  if rng is None:
   rnd=np.random.default_rng(9001)
  else:rnd=rng
  p=np.stack([rnd.permutation(x.shape[1]) for _ in range(len(x))])
  order=torch.tensor(p,dtype=torch.long)
  x=torch.gather(x,1,order[:,:,None].expand_as(x))
 s,h,v,imp=model(x)
 y=torch.from_numpy(batch["stance"][idx])
 hit=torch.from_numpy(batch["impact"][idx])
 dv=torch.from_numpy(batch["delta_v"][idx])
 J=torch.from_numpy(batch["impulse"][idx])
 mask=(hit>.5)
 c=F.cross_entropy(s,y)+F.binary_cross_entropy_with_logits(h,hit,pos_weight=pos_weight)
 if mask.any():
  outcome=F.smooth_l1_loss(v[mask],dv[mask])*.1+F.smooth_l1_loss(imp[mask],J[mask])*.08
 else:
  outcome=torch.zeros_like(c)
 return c+outcome

def fit(train,val,cfg,shuffled):
 seed=cfg["seed"]
 torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 model=ContactImpactHead(cfg["hidden"])
 opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
 count=float(train["impact"].sum())
 if not 1<=count<=len(train["x"])-1:raise ValueError("No measured TRAIN rare physical impacts available")
 weight=torch.tensor([(len(train)-count)/count],dtype=torch.float32)
 rng=np.random.default_rng(seed)
 history=[];best=float("inf");best_epoch=0;best_weights=None
 for epoch in range(cfg["epochs"]):
  model.train()
  order=rng.permutation(len(train["x"]))
  epoch_loss=[]
  for start in range(0,len(order),cfg["batch_size"]):
   idx=order[start:start+cfg["batch_size"]]
   opt.zero_grad(set_to_none=True)
   loss=loss_for_network(model,train,idx,weight,shuffled,rng)
   if not torch.isfinite(loss):raise FloatingPointError("Unqualified contact learning objective")
   loss.backward()
   nn.utils.clip_grad_norm_(model.parameters(),1.0)
   opt.step()
   epoch_loss.append(float(loss.detach()))
  model.eval()
  with torch.no_grad():
   # The shuffled control must be evaluated shuffled as well, but with a
   # fixed predeclared seed independent of epoch to avoid model-selection noise.
   vloss=float(loss_for_network(model,val,np.arange(len(val["x"])),
                       weight,shuffled,np.random.default_rng(9001)))
  if not np.isfinite(vloss):raise ValueError("Contact D holdout validation became nonfinite")
  history.append(dict(epoch=epoch+1,train_loss=float(np.mean(epoch_loss)),valid_loss=vloss))
  if vloss<best-1e-8:
   best=vloss;best_epoch=epoch+1
   best_weights=copy.deepcopy(model.state_dict())
  if epoch+1-best_epoch>=cfg["early_stop_patience"]:break
 model.load_state_dict(best_weights);model.eval()
 return model,dict(best_validation_epoch=best_epoch,best_validation_loss=best,
    early_stop_epochs_executed=len(history),history=history,trained_on_scene_C_only=True,
    selected_only_on_scene_D=True,center_test_inaccessible_during_fit=True,
    shuffled_history_negative_control=shuffled,train_only_impact_pos_weight=float(weight[0]))

def performance(y,p):
 """No test-learned threshold; binary 0.5 was fixed before all training."""
 y=np.asarray(y).astype(bool);p=np.asarray(p)>0.5
 tp=int(np.sum(y&p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p))
 precision=tp/(tp+fp) if tp+fp else 0.
 recall=tp/(tp+fn) if tp+fn else 0.
 f1=2*precision*recall/(precision+recall) if precision+recall else 0.
 return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=f1)

def inference(model,batch,shuffled):
 x=batch["x"].copy()
 if shuffled:
  r=np.random.default_rng(9001)
  p=np.stack([r.permutation(x.shape[1]) for _ in x])
  x=np.take_along_axis(x,p[:,:,None],axis=1)
 with torch.no_grad():
  a,b,c,d=model(torch.tensor(x,dtype=torch.float32))
 return a.softmax(-1).numpy(),b.sigmoid().numpy(),c.numpy(),d.numpy()

def score(model,batch,shuffled):
 s,h,v,imp=inference(model,batch,shuffled)
 y=batch["impact"].astype(bool)
 return dict(
  n_windows=len(y),
  predicted_stance_accuracy=float(np.mean(s.argmax(-1)==batch["stance"])),
  hazard_next_100ms_F1=performance(y,h),
  hazard_prob_Brier=float(np.mean((h-batch["impact"])**2)),
  contact_positive_windows=int(y.sum()),
  impact_postball_delta_velocity_MAE_mps=float(np.mean(abs(v[y]-batch["delta_v"][y]))) if y.any() else None,
  impact_contact_scalar_impulse_MAE_Ns=float(np.mean(abs(imp[y]-batch["impulse"][y]))) if y.any() else None,
  not_29dof_motor_trajectory_generated=True,
  subject_human_contact_truth_unavailable=True)

def run(config_path:Path):
 c=json.loads(config_path.read_text())
 if (c["version"]!="v0.1.90" or c["train_scene"]!="v084_C" or
     c["validation_scene"]!="v084_D" or c["test_scene"]!="v085_center" or
     c["train_on_center_scene"] or c["human_true_impact_available"] or
     c["elite_foot_kick_claim"] or c["model_is_29dof_actor"]):
  raise ValueError("KineticMimic measured PhysX evidence split/groundtruth not frozen")
 for file,expected in c["source_sha256"].items():
  if sha(ROOT/file)!=expected:raise ValueError("Original local source method file changed")
 if c["epochs"]>150 or c["history_frames"]!=12 or c["prediction_horizon_physics_frames"]!=20:
  raise ValueError("Contact phase model sample rate or training budget drift")
 source=c["source_physx"]
 # Do not load/open center test source until BOTH models already trained on C
 # and selected on D. The source SHA is a commit-time manifest.
 train=[load_original_physics(Path(z["path"]),z["sha256"],key)
        for key,z in source.items() if z["scene"]=="v084_C"]
 val=[load_original_physics(Path(z["path"]),z["sha256"],key)
        for key,z in source.items() if z["scene"]=="v084_D"]
 if len(train)!=3 or len(val)!=3:raise ValueError("Original C/D policy controls missing")
 for e in train+val:
  if not strict_physics(e)["strict_pass"]:
   raise ValueError("Training physical episode is a different task, not a qualified pass")
 tr=rows_to_training(train,c);v=rows_to_training(val,c)
 # Train-only normalizer, physically measured 71D per-context features.
 mu=tr["x"].reshape(-1,71).mean(0).astype(np.float32)
 sd=np.maximum(tr["x"].reshape(-1,71).std(0),.05).astype(np.float32)
 tr["x"]=np.clip((tr["x"]-mu)/sd,-12.,12.)
 v["x"]=np.clip((v["x"]-mu)/sd,-12.,12.)
 models={};diagnostics={}
 for mode in ("ordered","history_shuffle_negative"):
  model,meta=fit(tr,v,c,shuffled=mode!="ordered")
  models[mode]=model;diagnostics[mode]=meta
 # Now and only now open the third PHYSICAL center-ball context. This is a
 # historical project context already known, NOT a blind paper test set.
 test=[load_original_physics(Path(z["path"]),z["sha256"],key)
       for key,z in source.items() if z["scene"]=="v085_center"]
 if len(test)!=3:raise ValueError("Actual historical center source absent")
 te=rows_to_training(test,c)
 te["x"]=np.clip((te["x"]-mu)/sd,-12.,12.)
 out=dict(version="v0.1.90",protocol_sha256=sha(config_path),
   original_training_scenes=["v084_C"],original_validation_scenes=["v084_D"],
   original_historical_test_scenes=["v085_center"],
   training_physics_unique_scenes=1,validation_physics_unique_scenes=1,
   test_physics_unique_scenes=1,
   model_has_NO_action_output=True,real_G1_PhysX_new_rollouts=0,
   training_precontact_rows=len(tr["x"]),validation_precontact_rows=len(v["x"]),
   testing_precontact_rows=len(te["x"]),source_episode_physics=[
    dict(case=e.case,raw_SHA256=e.trace_source_sha256,real_5ms_rows=e.true_physics_rows,
         terminal_marker=e.terminal_annotation_count,physical_impulse_proxy_Ns=e.estimated_foot_ball_scalar_impulse_Ns,
         original_strict=bool(strict_physics(e)["strict_pass"]),
         actual_ball_speed_5_to_105ms_mps=e.observed_ball_speed_5_to_105ms_mps,
         original_impact_s=e.first_impact_s)
    for e in train+val+test],
   fitted_contact_heads={},evidence_limitation={
    "optical_professional_human_impact_groundtruth":False,
    "original_G1_native_gpu_wrist_hip_collider_pair_certified":False,
    "physical_ball_impulse_is_unsigned_scalar_proxy":True,
    "original_29dof_motor_actuation_trained":False,
    "actual_G1_20ms_future_support_uses_future_label_only":True,
    "test_scene_already_historically_seen_by_project":True,
    "current_G1_controller_safe_to_upgrade":False})
 for name,m in models.items():
  out["fitted_contact_heads"][name]=dict(
    training_metadata=diagnostics[name],
    validation=score(m,v,name!="ordered"),
    test=score(m,te,name!="ordered"))
 out["nonlearned_test_baselines"]=dict(
  stance_persistence_accuracy=float(np.mean(te["current_stance"]==te["stance"])),
  contact_never_predicted_F1=performance(te["impact"].astype(bool),
        np.zeros(len(te["impact"]))),
  constant_training_scene_impact_delta_v_mae_mps=float(
       np.mean(abs(te["delta_v"][te["impact"]>0]-tr["delta_v"][tr["impact"]>0].mean(axis=0))))
 )
 target=ROOT/c["output_dir"]
 if target.exists():raise FileExistsError("KineticMimic measured evidence overwrite prohibited")
 target.mkdir(parents=True)
 for name,m in models.items():
  path=target/(name+"_contact_impact_head.pt")
  torch.save(dict(model=m.state_dict(),features=71,history=12,hidden=c["hidden"],
    normalization_mean=mu,normalization_std=sd,
    train_source_SHA256={e.case:e.trace_source_sha256 for e in train},
    human_true_GRf_or_impact_available=False,not_G1_actor=True),path)
  out["fitted_contact_heads"][name]["checkpoint_sha256"]=sha(path)
 report=target/"kineticmimic_contact_training_audit.json"
 report.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
 print("KINETICMIMIC_V090_CONTACT_REAL_PHYSX_HEAD_TRAINED",json.dumps(dict(
  training_windows=out["training_precontact_rows"],validation_windows=out["validation_precontact_rows"],
  test_windows=out["testing_precontact_rows"],
  test={k:dict(stance_acc=round(v["test"]["predicted_stance_accuracy"],3),
             contact_f1=round(v["test"]["hazard_next_100ms_F1"]["f1"],3),
             ball_delta_MAE=round(v["test"]["impact_postball_delta_velocity_MAE_mps"],3))
        for k,v in out["fitted_contact_heads"].items()},real_new_physx=0)),flush=True)
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--config",type=Path,default=ROOT/"configs/kineticmimic_v090_study.json")
 a=p.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 if torch.cuda.is_available():raise RuntimeError("CPU exploratory real G1 200Hz training only, no concurrent Isaac GPU")
 run(a.config)
if __name__=="__main__":main()
