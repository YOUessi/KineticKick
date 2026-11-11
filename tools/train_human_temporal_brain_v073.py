#!/usr/bin/env python3
"""v73: CPU-trained *human* 120Hz full-body future-motion "brain" pilot.

Uses real CMU C3D-derived 19 body landmarks + 8 segment cosines from six
soccer-labeled optical motion clips; 4 subject-10 train clips, 1 subject-10
validation clip, and 1 entirely held-out subject-11 test clip. This is NOT
professional-quality shooting, exact contact, ball force, GRF, or robot RL.

Hypothesis: motion order in human history improves 0.1s multichannel future
motion prediction versus static/linear and shuffled-history controls.

Rigid rules: split by whole clips/subjects; select checkpoints on validation
only; inspect subject-11 once after both fits; no zero-filled source NaNs, no
test-gradient flow, no GPU, all raw artifacts and weights stay on Tang.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn

FEATS=65
LANDMARKS=19
COS_COUNT=8
ROOT=Path(__file__).resolve().parents[1]

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for part in iter(lambda:f.read(1<<20),b""):h.update(part)
 return h.hexdigest()

def load_verified(path:Path, expected:str, subject:str):
 if not path.exists() or sha(path)!=expected:
  raise ValueError("Original CMU human optical 120Hz source SHA drift: "+str(path))
 with np.load(path,allow_pickle=False) as z:
  time=np.asarray(z["time_s"],dtype=np.float64)
  pose=np.asarray(z["pelvis_relative_scale_normalized_xyz"],dtype=np.float64)
  cos=np.asarray(z["marker_derived_human_8_cosine_relations"],dtype=np.float64)
  marker=np.asarray(z["primary_marker_valid"],dtype=np.bool_)
  actual=str(z["original_human_subject"].item())
  fps=float(z["original_frame_rate_hz"])
  no_true_impact=not bool(z["true_impact_event_available"].item())
  no_grf=not bool(z["force_plate_data_available"].item())
  names=z["primary_landmark_names"].tolist()
 if (actual!=str(subject) or not no_true_impact or not no_grf or fps!=120. or
     pose.ndim!=3 or pose.shape[1:]!=(LANDMARKS,3) or
     cos.shape!=(len(pose),COS_COUNT) or marker.shape!=(len(pose),LANDMARKS) or
     len(time)!=len(pose) or len(names)!=LANDMARKS or
     not np.all(np.diff(time)>0) or
     not np.allclose(np.diff(time),1/120,atol=1e-5,rtol=0)):
  raise ValueError("Human 120Hz original pose/impact/source schema invalid")
 feat=np.concatenate([pose.reshape(len(pose),57),cos],axis=1)
 valid=marker.all(axis=1)&np.isfinite(feat).all(axis=1)
 if (valid.mean()<.95 or np.count_nonzero(valid)<100 or
     np.nanmax(np.abs(pose))>5. or np.nanmax(np.abs(cos))>1.00001):
  raise ValueError("Original CMU capture missing too many reliable body frames")
 return feat,valid,time,names

def clip_windows(feature,mask,history,forecast,stride,mean,std):
 if len(feature)!=len(mask) or feature.shape[1]!=FEATS or stride<1:
  raise ValueError("Invalid human source clip windows")
 features=(feature-mean)/std
 windows_x=[];windows_y=[];starts=[]
 for start in range(0,len(feature)-history-forecast+1,stride):
  stop=start+history+forecast
  if mask[start:stop].all() and np.isfinite(features[start:stop]).all():
   windows_x.append(features[start:start+history])
   windows_y.append(features[start+history:stop])
   starts.append(start)
 if not windows_x:
  raise ValueError("No contiguous reliable 120Hz future-motion windows")
 return (np.stack(windows_x).astype(np.float32),
         np.stack(windows_y).astype(np.float32),
         np.asarray(starts,dtype=np.int64))

class TemporalBrain(nn.Module):
 """Human-only ordered historical pose→12-step future motion; no G1 physics."""
 def __init__(self,history,forecast,features=FEATS,hidden=96):
  super().__init__()
  if history<2 or forecast<1 or features!=FEATS:
   raise ValueError("Invalid full-human 65D temporal horizon")
  self.history=history;self.forecast=forecast
  self.gru=nn.GRU(input_size=FEATS,hidden_size=hidden,
                  num_layers=2,batch_first=True,dropout=.10)
  self.head=nn.Sequential(nn.Linear(hidden,128),nn.GELU(),
                          nn.Linear(128,forecast*FEATS))
  # Starts from no-motion baseline; learns a residual forecast rather than
  # copying an unbounded unrelated pose from a tiny human dataset.
  nn.init.zeros_(self.head[-1].weight)
  nn.init.zeros_(self.head[-1].bias)
 def forward(self,x):
  if x.ndim!=3 or x.shape[1:]!=(self.history,FEATS):
   raise ValueError("Only original B x 24 x (19x3+8) input accepted")
  _,h=self.gru(x)
  out=self.head(h[-1]).view(len(x),self.forecast,FEATS)
  return x[:,-1:, :]+out

def persistence(x,horizon):
 return np.repeat(x[:,-1:,:],horizon,axis=1)

def linear_velocity(x,horizon):
 # Fixed 4-frame finite difference, never learned/tuned on holdout.
 vel=(x[:,-1,:]-x[:,-5,:])/4
 return x[:,-1:, :]+np.arange(1,horizon+1,dtype=np.float32)[None,:,None]*vel[:,None,:]

def human_metrics(pred,target,mean,std):
 """Measured error in scale-normalized original C3D landmark/angle units."""
 if pred.shape!=target.shape or pred.shape[-1]!=FEATS:
  raise ValueError("Only matching whole-body future motion can be scored")
 physical_p=pred*std+mean
 physical_t=target*std+mean
 if not (np.isfinite(physical_p).all() and np.isfinite(physical_t).all()):
  raise ValueError("Invalid 120Hz human predicted 3D path")
 delta=physical_p-physical_t
 # Whole-body 3D RMS over 19 named optical-derived landmarks; not meters,
 # because source is pelvis-relative and scale-normalized.
 l2=np.linalg.norm(delta[:,:,:57].reshape(-1,LANDMARKS,3),axis=2)
 angle=np.abs(delta[:,:,57:])
 cosine_p=physical_p[:,:,57:].clip(-1,1)
 cosine_t=physical_t[:,:,57:].clip(-1,1)
 motion_p=np.diff(physical_p[:,:,:57],axis=1)
 motion_t=np.diff(physical_t[:,:,:57],axis=1)
 ref_spread=np.mean(np.abs(motion_t))
 return dict(
  n_future_12_frame_windows=int(len(pred)),
  normalized_xyz_landmark_euclidean_MAE=float(l2.mean()),
  normalized_xyz_landmark_euclidean_p95=float(np.percentile(l2,95)),
  observed_8_segment_cosine_MAE=float(angle.mean()),
  predicted_cosine_bound_violation_rate=float(np.mean(np.abs(physical_p[:,:,57:])>1+1e-6)),
  mean_future_xyz_velocity_absolute_error_per_120hz_frame=float(np.abs(motion_p-motion_t).mean()),
  target_future_xyz_velocity_scale_per_120hz_frame=float(ref_spread),
  note="Optical pelvis-relative scale-normalized marker coordinates, NOT metric G1 joints/actual force/naturalness"
 )

def weighted_loss(p,y):
 # Train-only weights, locked before observing subject-11 test.
 err=p-y
 position=err[:,:,:57].square().mean()
 relation=err[:,:,57:].square().mean()
 derivatives=(p[:,1:]-p[:,:-1]-(y[:,1:]-y[:,:-1])).square().mean()
 return position+relation*.45+derivatives*.15

def train_net(x,y,vx,vy,cfg,shuffled):
 seed=int(cfg["seed"])
 torch.manual_seed(seed)
 np.random.seed(seed)
 random.seed(seed)
 model=TemporalBrain(cfg["history_frames"],cfg["forecast_frames"],
                     hidden=cfg["gru_hidden"])
 opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],
                       weight_decay=cfg["weight_decay"])
 xt=torch.from_numpy(x);yt=torch.from_numpy(y)
 valx=torch.from_numpy(vx);valy=torch.from_numpy(vy)
 n=len(x);batch=int(cfg["batch_size"])
 rng=np.random.default_rng(seed)
 best=float("inf");best_epoch=0;best_weights=None
 history=[]
 for epoch in range(cfg["epochs"]):
  model.train()
  perm=rng.permutation(n)
  train_losses=[]
  for chunk in range(0,n,batch):
   ii=perm[chunk:chunk+batch]
   inp=xt[ii].clone()
   if shuffled:
    # Exact sample ordering is removed only from the negative-control input.
    # Target future remains untouched. Training budget/seed remains equal.
    p=np.stack([rng.permutation(inp.shape[1]) for _ in range(len(ii))])
    inp=torch.gather(inp,1,torch.tensor(p,dtype=torch.long)[:,:,None].expand_as(inp))
   predicted=model(inp)
   loss=weighted_loss(predicted,yt[ii])
   if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
   opt.zero_grad(set_to_none=True);loss.backward()
   nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
   opt.step()
   train_losses.append(float(loss.detach()))
  model.eval()
  with torch.no_grad():
   val=float(weighted_loss(model(valx),valy))
  if not np.isfinite(val):raise FloatingPointError("Nonfinite human validation loss")
  if val < best-1e-9:
   best=val;best_epoch=epoch+1
   best_weights=copy.deepcopy(model.state_dict())
  history.append(dict(epoch=epoch+1,train_weighted=float(np.mean(train_losses)),
                      validation_weighted=val))
  if epoch+1-best_epoch>=cfg["early_stop_patience"]:
   break
 assert best_weights is not None
 model.load_state_dict(best_weights);model.eval()
 return model,dict(train_history=history,best_epoch=best_epoch,
                    best_validation_weighted=best,epochs_executed=len(history),
                    negative_control_input_shuffled=bool(shuffled),
                    training_samples=int(n),
                    validation_samples=int(len(vx)),
                    selection_uses_test_subject=False)

def infer(model,inputs,batch=128):
 out=[]
 model.eval()
 with torch.no_grad():
  for start in range(0,len(inputs),batch):
   x=torch.from_numpy(inputs[start:start+batch])
   out.append(model(x).cpu().numpy())
 return np.concatenate(out,axis=0)

def run(protocol_path,source_root,output):
 protocol=json.loads(protocol_path.read_text())
 if (protocol["version"]!="v0.1.73" or protocol["seed"]!=7301 or
     protocol["train_clips"]!=["10_01","10_02","10_03","10_05"] or
     protocol["validation_clips"]!=["10_06"] or
     protocol["test_clips"]!=["11_01"] or
     protocol["test_subject_seen_in_training"] or
     protocol["professional_instestep_contact_groundtruth"]):
  raise ValueError("No changed human subject isolation / training protocol")
 if protocol["forecast_frames"]!=12 or protocol["history_frames"]!=24:
  raise ValueError("Whole-body 120Hz history/forecast scientific horizon changed")
 for rel, expected in protocol["source_code_sha256"].items():
  if sha(ROOT/rel)!=expected:
   raise ValueError("Original pretraining source code/protocol SHA drift: "+rel)
 if output.exists():
  raise FileExistsError("No replacement of original training logs/checkpoints")
 source_dir=source_root/protocol["original_human_rel_dir"]
 trainvalclips=protocol["train_clips"]+protocol["validation_clips"]
 loaded={}
 for clip in trainvalclips:
  source=source_dir/(clip+".npz")
  spec=protocol["original_human_source"][clip]
  loaded[clip]=load_verified(source,spec["sha256"],spec["subject"])
 # Freeze normalizer purely on original observed and valid SUBJECT-10 TRAIN.
 train_records=np.concatenate([loaded[c][0][loaded[c][1]] for c in protocol["train_clips"]],axis=0)
 mean=train_records.mean(axis=0)
 std=np.maximum(train_records.std(axis=0),.025)
 if len(mean)!=FEATS or not np.isfinite(std).all():
  raise ValueError("Failed to learn isolated human subject train-only normalization")
 allwindows={}
 for c in trainvalclips:
  data,valid,time,names=loaded[c]
  stride=protocol["train_stride_frames"] if c in protocol["train_clips"] else protocol["eval_stride_frames"]
  allwindows[c]=clip_windows(data,valid,protocol["history_frames"],protocol["forecast_frames"],
                             stride,mean,std)
 xt=np.concatenate([allwindows[c][0] for c in protocol["train_clips"]])
 yt=np.concatenate([allwindows[c][1] for c in protocol["train_clips"]])
 vx=allwindows["10_06"][0];vy=allwindows["10_06"][1]
 # Actual independent subject-11 record is NOT even loaded before selecting
 # BOTH equal-budget models on subject-10 validation.
 models={};models_diag={}
 for name,shuffled in [("ordered",False),("history_shuffle_negative",True)]:
  model,metadata=train_net(xt,yt,vx,vy,protocol,shuffled)
  models[name]=model
  models_diag[name]=metadata
 # Now and ONLY NOW open the held-out subject-11 source once, for reporting.
 clip=protocol["test_clips"][0]
 spec=protocol["original_human_source"][clip]
 loaded[clip]=load_verified(source_dir/(clip+".npz"),spec["sha256"],spec["subject"])
 data,valid,time,names=loaded[clip]
 allwindows[clip]=clip_windows(data,valid,protocol["history_frames"],
    protocol["forecast_frames"],protocol["eval_stride_frames"],mean,std)
 allclips=trainvalclips+[clip]
 summary=dict(protocol_sha256=sha(protocol_path),
   source_dir=str(source_dir),
   distinct_human_train_subjects=1,distinct_human_test_subjects=1,
   split_exact=dict(train=protocol["train_clips"],val=protocol["validation_clips"],
                    heldout=protocol["test_clips"]),
   original_cmu_capture_frames={c:int(len(loaded[c][0])) for c in allclips},
   training_windows_by_original_clip={c:int(len(allwindows[c][0])) for c in allclips},
   observed_source_missing_frames_by_clip={c:int(np.count_nonzero(~loaded[c][1])) for c in allclips},
   raw_motion_unit="pelvis-relative scale-normalized 3D, 8 cosine channels; not meters, no contact/ball impulse",
   no_motor_actuation_or_real_PhysX_in_this_CPU_batch=True,
   source_person_is_skilled_elite_verified=False,
   independent_foot_ball_contact_time_verified=False,
   original_ground_reaction_force_verified=False,
   target_clips_test_untouched_until_both_validation_selections=True)
 testx,testy=allwindows["11_01"][:2]
 baselines={"persistence":persistence(testx,protocol["forecast_frames"]),
            "linear_extrapolation":linear_velocity(testx,protocol["forecast_frames"])}
 for name,m in models.items():baselines[name]=infer(m,testx)
 summary["test_subject_11_per_method"]={
  name:human_metrics(arr,testy,mean,std) for name,arr in baselines.items()}
 summary["validation_subject_10_clip_06_per_method"]={
   "persistence":human_metrics(persistence(vx,12),vy,mean,std),
   "linear_extrapolation":human_metrics(linear_velocity(vx,12),vy,mean,std),
   **{name:human_metrics(infer(m,vx),vy,mean,std) for name,m in models.items()}}
 summary["training_metadata"]=models_diag
 summary["test_subject_11_is_single_sequence_no_population_generalization"]=True
 # A 57D marker predictor is NOT a human "brain" for original robot until
 # downstream G1 morphology-aware IK and 200Hz real physical safety pass.
 summary["predictive_naturalness_claim"]=False
 summary["model_technology"]="GRU-2layer residual multi-frame human optical motion prediction"
 output.mkdir(parents=True)
 for name,m in models.items():
  path=output/(name+"_human_motion_gru.pt")
  torch.save(dict(state_dict=m.state_dict(),mean=mean.astype(np.float32),
      std=std.astype(np.float32),history=24,forecast=12,features=65,
      model_hidden=protocol["gru_hidden"],train_clip_subject="10",test_clip_subject="11"),
      path)
  summary.setdefault("checkpoints",{})[name]=dict(
    path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)
 artifact=output/"human_motion_evaluation.json"
 artifact.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
 print("V073_REAL_CMU_120HZ_CPU_TRAINED",json.dumps({
  "train_windows":len(xt),"validation_windows":len(vx),"subject11_heldout_windows":len(testx),
  "test_landmark_3D_normalized_MAE":{k:round(v["normalized_xyz_landmark_euclidean_MAE"],6) for k,v in summary["test_subject_11_per_method"].items()},
  "ordered_best_epoch":models_diag["ordered"]["best_epoch"],
  "shuffled_best_epoch":models_diag["history_shuffle_negative"]["best_epoch"],
  "all_scores_are_human_optical_only":True},ensure_ascii=False),flush=True)
 return summary

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--protocol",type=Path,default=ROOT/"configs/v073_human_temporal_learning.json")
 p.add_argument("--source-root",type=Path,required=True)
 p.add_argument("--output",type=Path,default=ROOT/"experiments/v073_cmu_human_temporal")
 a=p.parse_args()
 torch.set_num_threads(4)
 torch.set_num_interop_threads(1)
 if torch.cuda.is_available():
  raise RuntimeError("Never consume CUDA during independent v72 IsaacLab boot isolation")
 run(a.protocol,a.source_root,a.output)

if __name__=="__main__":
 main()
