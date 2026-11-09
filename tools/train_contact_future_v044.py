#!/usr/bin/env python3
"""v44 one-shot 29DoF future risk pilot: grouped train/val/test and baselines.

Only fixed source data. Hyperparameters and test groups are frozen before
training; model selection uses validation MAE. The test split is not truly
unseen to the research program because v42 case results were already audited.
The trained model is not silently deployed as a safety shield.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,math,random,time
import numpy as np
import torch
from contact_risk_model_v044 import ActionConditionedRiskMLP,risk_metrics

ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/contact_future_forecast_v044.json"

def digest(f):
    h=hashlib.sha256()
    with (ROOT/f).open("rb") as p:
        for b in iter(lambda:p.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def ridge_train(x,y,reg):
    X=np.concatenate([x.astype(float),np.ones((len(x),1),dtype=float)],axis=-1)
    M=np.eye(X.shape[1],dtype=float)*reg
    M[-1,-1]=0.
    coeff=np.linalg.solve(X.T@X+M,X.T@y)
    if coeff.shape!=(x.shape[1]+1,y.shape[1]):
        raise ValueError("Wrong ridge transfer")
    return coeff

def ridge_pred(x,weight):
    X=np.concatenate([x.astype(float),np.ones((len(x),1),dtype=float)],axis=-1)
    return np.clip(X@weight,0,None).astype(np.float32)

@torch.no_grad()
def predict(model,x,mu,std,ym,ys,device):
    model.eval()
    return model(torch.as_tensor(x,dtype=torch.float32,device=device),
                 mu,std,ym,ys).cpu().numpy()

def run():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default=STUDY)
    args=p.parse_args()
    if args.config!=STUDY:raise ValueError("Only one preregistered study")
    cfg=json.loads((ROOT/STUDY).read_text())
    if (cfg["version"]!="0.1.44" or cfg["training"]["steps"]!=1000
        or cfg["model_uses_future_physics_at_inference"] is not False):
        raise ValueError("Wrong original trainer study")
    for name,dig in cfg["source_sha256"].items():
        if digest(name)!=dig:raise ValueError("Source SHA has drifted "+name)
    data_path=cfg["output_directory"]+"/planned_action_future_qdot_v044.npz"
    if digest(data_path)!=cfg["original_dataset_sha256"]:
        raise ValueError("Training data changed after preregistration")
    with np.load(ROOT/data_path,allow_pickle=False) as z:
        x=z["features"].astype(np.float32)
        y=z["targets_max_qdot_ratio_per_joint"].astype(np.float32)
        s=z["split"]
        groups=z["episode_id"]
        controllers=z["controller"]
        times=z["relative_observation_time_s"]
    if x.shape!=(144,67) or y.shape!=(144,29):
        raise ValueError("Training database mismatch")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("G1 observations/future labels must be finite")
    for role,required_groups in cfg["group_split"].items():
        present=set(groups[s==role].tolist())
        if present!=set(required_groups):raise ValueError("Future/test leakage "+role)
    tr=s=="train"
    vl=s=="val"
    ts=s=="test"
    if (tr.sum(),vl.sum(),ts.sum())!=(96,24,24):
        raise ValueError("Expected frozen grouped sample counts")
    if np.unique(x[:,66]).tolist()!=[0.,1.]:raise ValueError("Plan action style flag missing from source")
    spec=cfg["training"]
    seed=spec["seed"]
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    out=ROOT/cfg["training_dir"]
    if out.exists():raise FileExistsError("Cannot rerun the training over existing trial")
    out.mkdir(parents=True)
    (out/"study.json").write_bytes((ROOT/STUDY).read_bytes())
    mean=x[tr].mean(axis=0).astype(np.float32)
    std=x[tr].std(axis=0).clip(.05,None).astype(np.float32)
    # The action commitment is binary; train-only mean/std still defines it.
    target_mean=y[tr].mean(axis=0).astype(np.float32)
    target_std=y[tr].std(axis=0).clip(.15,None).astype(np.float32)
    model=ActionConditionedRiskMLP().to(device)
    xm=torch.tensor(mean,device=device);xs=torch.tensor(std,device=device)
    ym=torch.tensor(target_mean,device=device);ys=torch.tensor(target_std,device=device)
    optim=torch.optim.AdamW(model.parameters(),lr=spec["lr"],weight_decay=spec["weight_decay"])
    train_x=torch.as_tensor(x[tr],device=device)
    train_y=torch.as_tensor(y[tr],device=device)
    valid_x=x[vl];valid_y=y[vl]
    ridge_weight=ridge_train((x[tr]-mean)/std,y[tr],cfg["ridge_alpha"])
    np.savez_compressed(out/"train_only_ridge_and_scaling.npz",ridge_weights=ridge_weight,
        input_mean=mean,input_std=std,target_mean=target_mean,target_std=target_std)
    best=float("inf")
    chosen=None
    history=[]
    start=time.monotonic()
    for step in range(1,spec["steps"]+1):
        model.train()
        idx=torch.randint(0,len(train_x),(spec["batch_size"],),device=device)
        current=train_x[idx]
        true=train_y[idx]
        pred=model(current,xm,xs,ym,ys)
        loss=torch.nn.functional.smooth_l1_loss(pred,true,beta=spec["huber_beta"])
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite physical qdot prediction loss")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),spec["max_grad_norm"])
        optim.step()
        if step%spec["val_interval"]==0:
            out_val=predict(model,valid_x,xm,xs,ym,ys,device)
            val_mae=float(np.mean(abs(out_val-valid_y)))
            history.append(dict(step=step,train_huber=float(loss.detach()),
                                validation_29joint_maxqdot_mae=val_mae))
            if val_mae<best:
                chosen=step
                best=val_mae
                torch.save(dict(model_state_dict=model.state_dict(),source_study_sha256=digest(STUDY),
                   original_data_sha256=cfg["original_dataset_sha256"],trained_steps=step,
                   input_mean=xm.cpu(),input_std=xs.cpu(),target_mean=ym.cpu(),target_std=ys.cpu(),
                   architecture=dict(input=67,hidden=128,output=29),
                   torch_version=str(torch.__version__)),out/"best_validation_model.pt")
            if step%200==0:
                print("V044_TRAIN",step,"best_val_mae",round(best,6),"selected_step",chosen,flush=True)
    if chosen is None:raise ValueError("Validation-only checkpoint selection missing")
    checkpoint=torch.load(out/"best_validation_model.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    model.eval()
    # Independently read the already known trial06 group only ONCE AFTER
    # validation selection; this is a grouped diagnostic, not newly blinded.
    output={}
    detailed={}
    ridge_cache={}
    for name,mask in (("train",tr),("val",vl),("test",ts)):
        xi=x[mask];yi=y[mask]
        base=dict(
           current_qdot_persistence=np.clip(abs(xi[:,29:58]),0,None),
           training_mean_future=np.tile(target_mean,(len(xi),1)),
           training_only_ridge=ridge_pred((xi-mean)/std,ridge_weight),
           planned_action_mlp=predict(model,xi,xm,xs,ym,ys,device))
        output[name]={k:risk_metrics(p,yi) for k,p in base.items()}
        detailed[name]=dict(episode_ids=groups[mask].tolist(),controller=controllers[mask].tolist(),
            time_relative_contact_s=times[mask].tolist(),
            actual_future_any_overspeed=(yi.max(axis=1)>1.).tolist(),
            mlp_predicts_unsafe=(base["planned_action_mlp"].max(axis=1)>1.).tolist(),
            ridge_predicts_unsafe=(base["training_only_ridge"].max(axis=1)>1.).tolist())
        ridge_cache[name]={k:v for k,v in base.items()}
    out_train=output["train"]
    val=output["val"]
    test=output["test"]
    result=dict(version="0.1.44",
      trained_on_retrospective_v42_data=True,not_blind_to_prior_v42_research=True,
      data_is_independent_real_world=False,
      no_new_physx_or_robot_control=True,
      n_physics_episode_groups=dict(train=4,val=1,test=1),
      n_controller_episodes=dict(train=8,val=2,test=2),
      n_correlated_windows=dict(train=96,val=24,test=24),
      future_horizon_s=1.25,
      current_time_physical_only_plus_committed_future_style_action=True,
      model_selection_only_by_val_mae=True,
      selected_training_step=chosen,
      best_validation_mae=best,
      scheduled_training_steps=spec["steps"],
      completed_updates=spec["steps"],
      measured_compute_seconds=round(time.monotonic()-start,3),
      trained_device=str(device),
      n_trainable_parameters=sum(p.numel() for p in model.parameters()),
      checkpoint_sha256=digest(cfg["training_dir"]+"/best_validation_model.pt"),
      training_data_sha256=cfg["original_dataset_sha256"],
      metrics=output,
      heldout_triple_test_MLP_beats_ridge=test["planned_action_mlp"]["mae"]<test["training_only_ridge"]["mae"],
      heldout_triple_test_MLP_beats_train_mean=test["planned_action_mlp"]["mae"]<test["training_mean_future"]["mae"],
      safety_certificate_obtained=False,
      model_allowed_to_replace_existing_robot_policy=False,
      model_allowed_to_claim_professional_human_biomechanics=False,
      limitations=[
       "Only six original known development ball-parameter groups (12 controller episodes)",
       "Train/val/test are group-disjoint by trial but all six were previously analyzed in v42; do not claim prospective blind validation",
       "Training examples within same PhysX episode are strongly correlated; N=96 windows not 96 independent robot trials",
       "Feature includes the precommitted future controller flag; no future joint outcome values in input",
       "1.25s qdot forecasts are not certified control barrier functions and cannot guarantee real/hardware safety",
       "True joint speed limits remain the original G1 model; no human explosive instep data or robot hardware testing"
      ])
    (out/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    (out/"training_history.json").write_text(json.dumps(history,indent=2)+"\n")
    (out/"scoring_samples.json").write_text(json.dumps(detailed,indent=2,ensure_ascii=False)+"\n")
    print("V044_MODEL_FROZEN_RESULT",json.dumps(dict(
       chosen=chosen,steps=spec["steps"],time=result["measured_compute_seconds"],
       val={k:round(v["mae"],5) for k,v in val.items()},
       test={k:round(v["mae"],5) for k,v in test.items()},
       mlp_better_than_ridge=result["heldout_triple_test_MLP_beats_ridge"],
       test_unsafe_windows=test["planned_action_mlp"]["all_future_unsafe_window_count"],
       test_model_false_safe=test["planned_action_mlp"]["underpredicted_unsafe_window_count"],
       model_sha=result["checkpoint_sha256"])),flush=True)

if __name__=="__main__":run()
