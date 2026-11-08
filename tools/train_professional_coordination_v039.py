#!/usr/bin/env python3
"""Fixed-budget supervised P3-interface pilot on broadcast-estimated 3DSP.

Strict game-grouped train/val/test: no test-driven tuning, no images upload,
no claim of actual verified human instep coordination or G1 control.
"""
from __future__ import annotations
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from conditional_coordination_v039 import CoordinationCVAE

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path("data/research_human_kicks/3dsp_cvpr2024/processed_v039/coordination_3dsp_v039.npz")
CONFIG=Path("configs/professional_coordination_pilot_v039.json")

def digest(path):
 h=hashlib.sha256()
 with (ROOT/path).open("rb") as f:
  for part in iter(lambda:f.read(1<<20),b""):h.update(part)
 return h.hexdigest()

def input_data(cfg):
 src=ROOT/SOURCE
 if digest(SOURCE)!=cfg["data_sha256"]:raise ValueError("Training data changed since protocol freeze")
 z=np.load(src,allow_pickle=False)
 assert str(z["source_sha256"])==cfg["original_archive_sha256"]
 assert bool(z["verified_event_contact_truth"]) is False
 x=z["relation_cosines"].astype(np.float32)
 mask=z["valid"].astype(bool)
 splits=z["split"]
 games=z["source_game"]
 ids=z["clip_id"]
 labels=z["source_label"]
 if x.shape!=(200,20,11) or mask.shape!=x.shape:
  raise ValueError("Wrong original source schema")
 if not np.isfinite(x[mask]).all() or (np.abs(x[mask])>1+1e-6).any():
  raise ValueError("Raw human relation cosines must be finite [-1,1]")
 observed={s:set(games[splits==s].tolist()) for s in ("train","val","test")}
 if any(observed[a]&observed[b] for a,b in (("train","val"),("val","test"),("train","test"))):
  raise ValueError("Football-game split leaked across train/val/test")
 required={"train":108,"val":47,"test":45}
 if {s:int((splits==s).sum()) for s in required} != required:
  raise ValueError("Source/split counts differ from frozen research protocol")
 if np.any(~mask):raise ValueError("Study requires actual complete feature observations; no silent imputation")
 return x,mask,splits,games,ids,labels

def metrics(name,pred,y,mask,groups):
 if pred.shape!=y.shape or mask.shape!=y.shape:raise ValueError(name+" invalid predicted shape")
 if not np.isfinite(pred).all():raise ValueError(name+" predicted nonfinite pose relations")
 diff=np.where(mask,pred-y,0.)
 n=int(mask.sum())
 mae=float(abs(diff).sum()/n)
 mse=float((diff**2).sum()/n)
 sample_squared=np.sum(diff**2,axis=(1,2))/np.sum(mask,axis=(1,2))
 game_mse={g:float(np.mean(sample_squared[groups==g])) for g in sorted(set(groups.tolist()))}
 if not np.all(np.isfinite(sample_squared)):raise ValueError("Bad per clip errors")
 return dict(name=name,mse=mse,rmse=float(np.sqrt(mse)),mae=mae,
             scalar_values_evaluated=n,clips=int(len(y)),
             mean_per_game_mse=float(np.mean(list(game_mse.values()))),
             game_count=len(game_mse),
             game_mse=game_mse,
             relation_mse=(np.sum(diff**2,axis=(0,1))/np.sum(mask,axis=(0,1))).round(8).tolist())

def ridge_fit(train_x,train_y,alpha):
 X=np.concatenate([train_x.reshape(len(train_x),-1),np.ones((len(train_x),1))],axis=-1).astype(np.float64)
 Y=train_y.reshape(len(train_y),-1).astype(np.float64)
 p=np.eye(X.shape[1],dtype=np.float64)*alpha
 p[-1,-1]=0.
 reg=np.linalg.solve(X.T@X+p,X.T@Y)
 if not np.isfinite(reg).all():raise ValueError("Bad ridge fit")
 return reg

def ridge_pred(x,reg):
 X=np.concatenate([x.reshape(len(x),-1),np.ones((len(x),1))],axis=-1).astype(np.float64)
 return np.clip((X@reg).reshape((len(x),12,11)),-1,1).astype(np.float32)

def eval_pilot(model, x, mask, splits, games, config, reg, mean_future, device):
 results={}
 t=8
 for s in ("val","test"):
  idx=np.flatnonzero(splits==s)
  past=x[idx,:t];future=x[idx,t:]
  with torch.no_grad():
   pred=model.predict(torch.from_numpy(past).to(device)).cpu().numpy()
  pred=pred.astype(np.float32)
  samples=dict(
    persistence=np.broadcast_to(past[:,-1:,:],future.shape).copy(),
    train_mean_curve=np.broadcast_to(mean_future[None],future.shape).copy(),
    ridge=ridge_pred(past,reg),
    conditional_vae_prior_mean=pred)
  results[s]={k:metrics(k,v,future,mask[idx,t:],games[idx]) for k,v in samples.items()}
 return results

def main():
 a=argparse.ArgumentParser()
 a.add_argument("--config",default=str(CONFIG))
 args=a.parse_args()
 cfg_path=Path(args.config)
 if cfg_path!=CONFIG:raise ValueError("Only registered v39 fixed study is permitted")
 cfg=json.loads((ROOT/CONFIG).read_text())
 for file,expected in cfg["source_sha256"].items():
  if digest(file)!=expected:raise ValueError("Source file drift: "+file)
 output=ROOT/cfg["output_dir"]
 if output.exists():raise FileExistsError("Preserve prior training outputs: "+str(output))
 output.mkdir(parents=True)
 (output/"frozen_protocol.json").write_bytes((ROOT/CONFIG).read_bytes())
 start=time.monotonic()
 seed=cfg["training"]["seed"]
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.set_num_threads(2)
 x,mask,split,games,ids,labels=input_data(cfg)
 t=cfg["history_frames"]
 assert t==8 and cfg["future_frames"]==12
 tr=x[split=="train"];val=x[split=="val"]
 mu=tr.reshape(-1,11).mean(0)
 sigma=tr.reshape(-1,11).std(0).clip(min=0.05)
 if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):raise ValueError("Bad train-only normalization")
 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.benchmark=False
 model=CoordinationCVAE(channels=11,history=t,future=12,hidden=128,latent=12,
                        mean=mu,std=sigma).to(device)
 optim=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["lr"],
                         weight_decay=cfg["training"]["weight_decay"])
 tx=torch.from_numpy(tr).to(device)
 vx=torch.from_numpy(val).to(device)
 batch=cfg["training"]["batch_size"]
 max_steps=cfg["training"]["steps"]
 step_period=cfg["training"]["eval_period"]
 best_val=float("inf")
 selected_step=None
 history=[]
 selection=output/"best_validation_model.pt"
 final=output/"last_model.pt"
 mean_future=tr[:,t:].mean(axis=0).astype(np.float32)
 reg=ridge_fit(tr[:,:t],tr[:,t:],cfg["ridge_alpha"])
 (output/"ridge_weights_train_only.npz").parent.mkdir(exist_ok=True)
 np.savez_compressed(output/"ridge_weights_train_only.npz",ridge_coef=reg,
                     train_mean_curve=mean_future,normalization_mean=mu,
                     normalization_std=sigma)
 for step in range(1,max_steps+1):
  model.train()
  ii=torch.randint(0,len(tx),(batch,),device=device)
  loss,components=model.training_objective(tx[ii,:t],tx[ii,t:],
                         kl_weight=cfg["training"]["kl_weight"],
                         posterior_weight=cfg["training"]["posterior_weight"])
  if not torch.isfinite(loss):raise ValueError(f"Nonfinite training loss at step {step}")
  optim.zero_grad(set_to_none=True)
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=cfg["training"]["max_grad_norm"])
  optim.step()
  if step%step_period==0 or step==max_steps:
   model.eval()
   with torch.no_grad():
    pred=model.predict(vx[:,:t])
    true=vx[:,t:]
    val_mse=float(((pred-true)**2).mean())
   if not np.isfinite(val_mse):raise ValueError("Nonfinite validation loss")
   entry=dict(step=step,train_total_loss=float(loss),validation_cosine_mse=val_mse,
              train_prior_loss=float(components["prior_mse"]),
              train_post_loss=float(components["posterior_mse"]),
              train_latent_kl=float(components["kl"]))
   history.append(entry)
   if val_mse<best_val:
    selected_step=step
    best_val=val_mse
    torch.save(dict(model=model.state_dict(),step=step,config=cfg,
                    original_source_sha256=cfg["original_archive_sha256"],
                    derived_source_sha256=cfg["data_sha256"],
                    source_sha256=cfg["source_sha256"],torch_version=torch.__version__),selection)
   if step%200==0:
    print("V039_TRAIN_STEP",step,"validation_mse",round(val_mse,6),
          "best",round(best_val,6),"best_at",selected_step,flush=True)
 torch.save(dict(model=model.state_dict(),step=max_steps,config=cfg,
                 source_sha256=cfg["source_sha256"]),final)
 # Do not open the independent game-held-out test until the final selection
 # is fixed by train/validation alone.
 state=torch.load(selection,map_location=device,weights_only=False)
 model.load_state_dict(state["model"],strict=True)
 model.eval()
 result=eval_pilot(model,x,mask,split,games,cfg,reg,mean_future,device)
 selected_test=result["test"]["conditional_vae_prior_mean"]["mse"]
 print("V039_HELD_OUT_FIRST_OPEN",json.dumps({k:round(v["mse"],6) for k,v in result["test"].items()}),flush=True)
 success=selected_test<result["test"]["persistence"]["mse"]
 manifest=dict(status="COMPLETED_SCIENTIFIC_PILOT_NOT_G1_POLICY",
     model="Conditional variational temporal coordination prior with supervised deployment prior mean",
     source="broadcast 3DSP posture estimates, NOT verified instep/impact/force data",
     professional_context_not_professional_mocap=True,
     source_data_sha256=cfg["data_sha256"],source_zip_sha256=cfg["original_archive_sha256"],
     game_disjoint_train_val_test=True,
     splits={s:dict(n_clips=int((split==s).sum()),game_count=len(set(games[split==s])))
             for s in ("train","val","test")},
     model_selection="lowest validation raw cosine MSE only, once fixed evaluation steps; no test model selection",
     model_selected_step=selected_step,
     model_val_mse=best_val,
     model_final_step=max_steps,
     trained_parameters=sum(p.numel() for p in model.parameters()),
     optimizer="AdamW",
     device=str(device),torch_version=torch.__version__,
     elapsed_seconds=round(time.monotonic()-start,3),
     training_updates=max_steps,
     training_rows=len(tr),
     first_test_access_after_selection=True,
     train_only_normalization={k:v.tolist() for k,v in [("mean",mu),("std",sigma)]},
     metrics=result,
     beats_persistence_on_test=success,
     best_model="experiments/human_coordination_v039/cvae/best_validation_model.pt",
     best_model_sha256=digest(str(selection.relative_to(ROOT))),
     final_model_sha256=digest(str(final.relative_to(ROOT))),
     limitation=[
      "No robot action predicted; model predicts pose-estimated relation cosine time sequences only",
      "Only 20 frames per clip and 8 observed/12 future; time relative to SoccerNet shot annotation is NOT exact foot strike",
      "Underlying game videos are professional soccer context, NOT verified instep shot biomechanical ground truth",
      "Test covers only 4 unseen games, all from historical SoccerNet 2015/16 3DSP source; not calibrated real G1 or expert rating",
      "No comparison to BeyondMimic/SuperMimic, no PhysX execution or naturalness preference data",
      "Training/validation/test group leakage is checked but source player IDs may recur across games",
      "Even if prediction MSE wins baseline, P2 qualified biomechanics/P3 humanoid skill generation are NOT completed"
     ])
 dest=output/"result.json"
 dest.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str)+"\n")
 (output/"training_history.json").write_text(json.dumps(history,indent=2)+"\n")
 print("V039_TRAIN_DONE",json.dumps({k:manifest[k] for k in (
      "model_selected_step","model_val_mse","training_updates","elapsed_seconds",
      "beats_persistence_on_test","best_model_sha256")},default=str),flush=True)
 print("V039_DATA_METRICS",json.dumps({s:{k:round(v["mse"],6) for k,v in z.items()}
                                  for s,z in result.items()}),flush=True)

if __name__=="__main__":main()
