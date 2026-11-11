#!/usr/bin/env python3
"""Actual learned 29DoF 0.4-second joint-trajectory proposals from SHA-certified
original G1 PhysX. NOT a human expert policy, command interface, or physics trial.

Frozen train scenes v84 C/D; never-used G/H 200Hz PhysX original holdout only, no episode mixing.
Only observation at start of a 0.4s segment determines 20 predicted future
50Hz velocity vectors and integrated position deltas. Causal current support
is optional, the future stance/impulse is never an input. Deterministic ridge
maps to six temporal basis knots. Ablations genuinely emit different full29
candidate sequences, not just descriptive metrics.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v097_coordination_observer as obs
obs.V91=ROOT
obs.V84=Path("/home/you/projects/kinetickick-wt-v084")
TRAIN=obs.TRAIN
TEST=obs.TEST
H=20
ANCHORS=np.array([0,4,8,12,16,19],dtype=int)
STRIDE=5
ALPHA=10.
# A G1 right-kick movement encompasses bilateral legs, waist and both arms.
GROUPS={
 "left_support_leg":("left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint"),
 "right_strike_leg":("right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"),
 "trunk":("waist_yaw_joint","waist_roll_joint","waist_pitch_joint"),
 "left_counterarm":("left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_yaw_joint","left_wrist_roll_joint","left_wrist_pitch_joint"),
 "right_arm":("right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_yaw_joint","right_wrist_roll_joint","right_wrist_pitch_joint")
}
# Joint group membership is verified against the actual 29 DoF robot before fitting.
def group_indices(names):
    ix={g:np.array([names.index(j) for j in joints],dtype=int) for g,joints in GROUPS.items()}
    if len(names)!=29 or any(len(z)!=len(set(z)) for z in ix.values()):
        raise ValueError("INVALID_REAL_29_JOINT_ORDER")
    flat=[int(v) for a in ix.values() for v in a]
    if len(flat)!=29 or set(flat)!=set(range(29)):raise ValueError("GROUPS_MUST_PARTITION_ACTUAL_29_DOF")
    return ix
def make_windows(z,feature:str):
    if feature not in ("state","full"):raise ValueError("INVALID_CAUSAL_FEATURE_SET")
    group_indices(z["joint_names"])
    X=z["base"] if feature=="state" else np.column_stack([z["base"],z["contact"]])
    if X.shape[1]!=(73 if feature=="state" else 80):raise ValueError("CAUSAL_INPUT_BAD")
    # z.base columns 35:64 are 29 actual joint speeds normalized by actual joint limits.
    speeds=z["base"][:,35:64]
    if speeds.shape[1]!=29:raise ValueError("FUTURE_NOT_ACTUAL_29DOF")
    starts=np.arange(0,len(X)-H,STRIDE)
    if not len(starts):raise ValueError("NO_COMPLETE_400MS_ROLLOUT")
    truth=np.stack([speeds[i+1:i+H+1] for i in starts])
    return X[starts].copy(),truth,starts
def learn(train:list[dict],feature:str,permute_contact=False):
    names=train[0]["joint_names"]
    if any(z["joint_names"]!=names for z in train):raise ValueError("ROBOT_JOINT_MISMATCH")
    Xs,Ys=[],[]
    rng=np.random.default_rng(950103)
    for z in train:
        x,y,_=make_windows(z,feature)
        if permute_contact:
            if feature!="full":raise ValueError("BAD_NEGATIVE")
            x[:,73:]=x[rng.permutation(len(x)),73:]
        Xs.append(x);Ys.append(y[:,ANCHORS].reshape(len(x),-1))
    X=np.concatenate(Xs);Y=np.concatenate(Ys)
    mu=X.mean(axis=0);sd=X.std(axis=0);sd=np.where(sd<.001,1.,sd)
    X=np.column_stack([np.ones(len(X)),(X-mu)/sd])
    reg=np.eye(X.shape[1])*ALPHA;reg[0,0]=0.
    W=np.linalg.solve(X.T@X+reg,X.T@Y)
    return {"W":W,"mu":mu,"std":sd,"feature":feature,
      "fit_rows":len(X),"train_sha256":{z["case"]:z["source_sha256"] for z in train},
      "robot_joint_order":names,"horizon_steps_50Hz":H,"basis_knots":ANCHORS.tolist()}
def infer(m,z):
    x,truth,starts=make_windows(z,m["feature"])
    xm=np.column_stack([np.ones(len(x)),(x-m["mu"])/m["std"]])
    predicted=(xm@m["W"]).reshape(len(x),len(ANCHORS),29)
    # Interpolate *predicted* 29 joint velocities at intermediate 50Hz steps;
    # the ground-truth future is never passed to the generator.
    output=np.empty_like(truth)
    for j in range(29):
        for i in range(len(x)):
            output[i,:,j]=np.interp(np.arange(H),ANCHORS,predicted[i,:,j])
    return output,truth,starts
def integrate_measured_q0_29dof(normalized_velocity, measured_q0, measured_velocity_limits, dt_s=.02):
    """Convert an actual 29-joint predicted 50Hz qdot sequence into full-body
    q(t), using ONLY measured current q0 and source joint speed limits.
    This does NOT certify joint angle limits, torque, support or collider safety,
    and cannot be dispatched to robot motors.
    """
    vel=np.asarray(normalized_velocity,dtype=float)
    q0=np.asarray(measured_q0,dtype=float)
    lim=np.asarray(measured_velocity_limits,dtype=float)
    if vel.ndim!=3 or vel.shape[1:]!=(H,29) or q0.shape!=(len(vel),29) or lim.shape!=(29,):
        raise ValueError("REQUIRES_FULL29_50HZ_TIMELINE_AND_29_LIMITS")
    if not np.isfinite(vel).all() or not np.isfinite(q0).all() or not np.isfinite(lim).all() or np.any(lim<=0):
        raise ValueError("UNQUALIFIED_MEASURED_POSITION_OR_SPEED")
    if abs(dt_s-.02)>1e-12:
        raise ValueError("FROZEN_50HZ_OUTPUT")
    q=q0[:,None,:]+np.cumsum(vel*lim[None,None,:]*dt_s,axis=1)
    if q.shape!=(len(vel),H,29) or not np.isfinite(q).all():
        raise ValueError("NO_29_DOF_CONFIGURATION")
    return q

def apply_ablation(v,ablation,indices):
    out=v.copy()
    if ablation=="full":pass
    elif ablation=="time_reversed":out=out[:,::-1,:].copy()
    elif ablation=="freeze_arms":
        out[:,:,np.concatenate((indices["left_counterarm"],indices["right_arm"]))]=0.
    elif ablation=="strike_leg_only":
        other=np.concatenate([val for key,val in indices.items() if key!="right_strike_leg"])
        out[:,:,other]=0.
    else:raise ValueError("UNKNOWN_PRE_REGISTERED_ABLATION")
    return out
def score(v,truth,indices):
    if v.shape!=truth.shape or v.shape[1:]!=(H,29):raise ValueError("SCORE_29DOF")
    absdev=np.abs(v-truth)
    phase_error={}
    per_group_mae={}
    for k,ind in indices.items():
        # Temporal peak of normalized joint kinetic activity: proxy only.
        ph=np.argmax(np.sum(np.square(truth[:,:,ind]),axis=2),axis=1)
        pr=np.argmax(np.sum(np.square(v[:,:,ind]),axis=2),axis=1)
        phase_error[k]=float(.02*np.mean(np.abs(pr-ph)))
        per_group_mae[k]=float(absdev[:,:,ind].mean())
    return {"windows":len(v),"mean_abs_29_qdot_normalized":float(absdev.mean()),
      "p95_abs_29_qdot_normalized":float(np.percentile(absdev,95)),
      "candidate_qdot_over_original_limit_fraction":float(np.mean(np.abs(v)>1.000001)),
      "maximum_candidate_velocity_limit_ratio":float(np.max(np.abs(v))),
      "all29_delta_q_rad_safety_unverified":True,
      "group_peak_time_mae_s":phase_error,"group_velocity_mae":per_group_mae}
def experiment(train,test,output):
    if output.exists() or output.with_suffix(".npz").exists():raise FileExistsError("WRITE_ONCE_MODEL")
    if set(z["case"] for z in train)&set(z["case"] for z in test):raise ValueError("EPISODE_LEAKAGE")
    if any(not z["certification"] for z in train+test):raise ValueError("UNCERTIFIED_ORIGINAL_G1")
    groups=group_indices(train[0]["joint_names"])
    models={"state_only":learn(train,"state"),"state_and_support":learn(train,"full"),
        "shuffled_support_negative":learn(train,"full",permute_contact=True)}
    results={}
    for z in test:
        cases={}
        for key,model in models.items():
            pred,truth,starts=infer(model,z)
            q0=2*z["base"][starts,6:35]
            pred_q=integrate_measured_q0_29dof(pred,q0,z["qdot_limit"])
            cases[key]=score(pred,truth,groups)
            cases[key]["max_abs_generated_displacement_rad"]=float(np.max(np.abs(pred_q-q0[:,None,:])))
            cases[key]["generated_positions_unchecked_joint_angle_torque_geometry"]=True
            if key=="state_and_support":
                # All four *generated* trajectories independently scored.
                for a in ("time_reversed","freeze_arms","strike_leg_only"):
                    proposed=apply_ablation(pred,a,groups)
                    pred_q_abl=integrate_measured_q0_29dof(proposed,q0,z["qdot_limit"])
                    cases[a]=score(proposed,truth,groups)
                    cases[a]["max_abs_generated_displacement_rad"]=float(np.max(np.abs(pred_q_abl-q0[:,None,:])))
                    cases[a]["generated_positions_unchecked_joint_angle_torque_geometry"]=True
        # This is a per-window metric on original robot *physical* observations,
        # not a physical rollout of generated predictions.
        results[z["case"]]={"source_sha256":z["source_sha256"],"physical_200hz_original_frames":z["real_200hz_frames"],
          "candidate_50hz_400ms_windows":cases["state_only"]["windows"],
          "actual_original_physx_hard_gate_pass":z["original_hard_gate_pass"],
          "original_first_true_foot_ball_contact_s":z["impact_time_s"],
          "terminal_censored":not z["original_physx_complete"],"metrics":cases}
    result={"version":"v0.1.97","training_case_sha256":{z["case"]:z["source_sha256"] for z in train},
      "holdout_case_sha256":{z["case"]:z["source_sha256"] for z in test},
      "train_scenes":"v84 C,D original G1","holdout_scenes":"v97 G,H original G1",
      "trained_on_certified_G1_PhysX_robot_not_human_elite":True,
      "future_support_grf_true_impact_as_input":False,
      "generates_full29_400ms_openloop_candidate_sequences":True,
      "position_deltas_integrate_predicted_velocities_not_true_future":True,
      "human_motion_style_qualified":False,"original_native_manifold_qualified":False,
      "physical_actuation_authorized":False,"model_physx_trials":0,"model_success_rate":None,
      "model":{"regularization":ALPHA,"anchor_offsets_steps":ANCHORS.tolist(),
        "candidate_dt_s":.02,"rollout_s":.4,"actuation_API":None},
      "holdout":results}
    with output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    weights={}
    for k,m in models.items():
        for field in ("W","mu","std"):weights[k+"_"+field]=m[field]
    np.savez(output.with_suffix(".npz"),**weights)
    print("V097_400MS_29DOF_GENERATOR",json.dumps({"training":list(result["training_case_sha256"]),
      "holdout":{k:{x:y["mean_abs_29_qdot_normalized"] for x,y in val["metrics"].items()} for k,val in results.items()},
      "output":str(output)},sort_keys=True),flush=True)
    return result
def run(output):
    train=[obs.load(case) for case in TRAIN]
    test=[obs.load(case,"v97") for case in TEST]
    return experiment(train,test,output)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    run(args.output)
