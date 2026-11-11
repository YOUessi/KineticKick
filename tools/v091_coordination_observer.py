#!/usr/bin/env python3
"""v91 KineticMimic causal 29DoF contact-conditioned coordination observer.

The input is real *robot* 200Hz PhysX, not human professional motion truth.
Only current and historical quantities enter features. Targets are actual future
29 joint velocities, stance state and actual next-100ms foot-ball first contact.
No motor commands can be issued by this observer.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
V84=Path("/home/you/projects/kinetickick-wt-v084")
V91=ROOT
DT=.005;HORIZON=20;STRIDE=4;ALPHA=10.
TRAIN=("C_v32_off","C_cmu02","D_v32","D_cmu02")
TEST=("E_v32","E_cmu02","F_v32","F_cmu02")
GROUPS={
 "support_left":("left_hip_pitch_joint","left_hip_roll_joint","left_knee_joint"),
 "pelvis":("left_hip_yaw_joint","right_hip_yaw_joint","left_hip_roll_joint","right_hip_roll_joint"),
 "trunk":("waist_yaw_joint","waist_roll_joint","waist_pitch_joint"),
 "swing_right":("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"),
 "counter_arm_left":("left_shoulder_pitch_joint","left_shoulder_roll_joint","left_elbow_joint","left_wrist_yaw_joint"),
}
def filehash(path:Path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for buf in iter(lambda:f.read(1024*1024),b""):h.update(buf)
 return h.hexdigest()

def load(case,source="v84"):
 if source=="v84":
  path=V84/"reports"/("v084_"+case+".json")
  audit=V84/"reports"/("v084_"+case+".audit.json")
 else:
  path=V91/"reports"/("v091_"+case+".json")
  audit=V91/"reports"/("v091_"+case+".audit.json")
 cert=json.loads(audit.read_text())
 if not cert.get("true_original_200Hz_G1_CUDA_PhysX"):raise ValueError("Uncertified original G1 source "+case)
 if filehash(path)!=cert["raw_sha256"]:raise ValueError("SHA drift in original real G1 "+case)
 raw=json.loads(path.read_text())
 if raw.get("root_fixed") is not False or raw.get("post_initialization_state_writes") is not False:
  raise ValueError("This is not uncontrolled free-root physical G1")
 names=tuple(raw["joint_names"])
 if len(names)!=29 or len(set(names))!=29:raise ValueError("Not 29 unique joints")
 records=raw["traces"][0]
 nraw=len(records)
 annotations=0
 if nraw>=2 and abs(float(records[-1]["time_s"])-float(records[-2]["time_s"])-.000001)<1e-7:
  a,b=records[-1],records[-2]
  if not a["terminated"] or any(a[k]!=b[k] for k in ("joint_pos","joint_vel","whole_body_com_xyz","root_xyz","ball_xyz")):
   raise ValueError("Nonphysics marker changes real observed physical state")
  records=records[:-1];annotations=1
 t=np.asarray([z["time_s"] for z in records],dtype=float)
 if len(t)<200 or not np.allclose(np.diff(t),DT,rtol=0,atol=1e-6) or abs(t[0]-DT)>1e-6:
  raise ValueError("Not actual uninterrupted 200Hz G1 physical times")
 n=len(t)
 def fetch(k,width):
  arr=np.asarray([z[k] for z in records],dtype=float)
  if arr.shape!=(n,width) or not np.isfinite(arr).all():raise ValueError("Invalid "+k)
  return arr
 q,dq=fetch("joint_pos",29),fetch("joint_vel",29)
 lim=np.asarray(raw["joint_velocity_limits"],dtype=float)
 if lim.shape==(1,29):lim=lim[0]  # original G1 NumPy-export schema is num_envs x all29
 if lim.shape!=(29,) or not np.isfinite(lim).all() or np.any(lim<=0):raise ValueError("Bad original velocity limits")
 feet=np.asarray([[z["support_foot_force_N"]["left"],z["support_foot_force_N"]["right"]] for z in records],dtype=float)
 if feet.shape!=(n,2) or not np.isfinite(feet).all() or np.any(feet<0):raise ValueError("Bad original foot forces")
 stance=(feet[:,0]>=80).astype(int)+2*(feet[:,1]>=80).astype(int)
 last_change=np.zeros(n,dtype=float)
 for i in range(1,n):
  last_change[i]=0 if stance[i]!=stance[i-1] else min(last_change[i-1]+DT,.6)
 root=fetch("root_xyz",3)
 relball=fetch("ball_xyz",3)-root
 relcom=fetch("whole_body_com_xyz",3)-root
 footv=fetch("right_foot_point_velocity",3)
 nonfoot=np.asarray([z["nonfoot_contact_N"] for z in records],dtype=float)
 if nonfoot.shape!=(n,) or not np.isfinite(nonfoot).all():raise ValueError("Invalid nonfoot force")
 impact=raw["records"][0]["contact_time_s"]
 force=np.asarray([sum(z["robot_ball_contact_force_N"].values()) for z in records],dtype=float)
 if not np.isfinite(force).all() or np.any(force<0):raise ValueError("Bad original foot-ball forces")
 actualhits=np.where(force>5)[0]
 if impact is not None:
  if len(actualhits)==0 or abs(t[actualhits[0]]-impact)>.006:raise ValueError("No physically measured first foot-ball event")
  impact=t[actualhits[0]]
 elif len(actualhits):raise ValueError("Contact force exists without reported true event")
 clock=np.column_stack((t/14,(t/14)**2,np.sin(2*np.pi*t),np.cos(2*np.pi*t),
                        np.sin(2*np.pi*t/.48),np.cos(2*np.pi*t/.48)))
 base=np.column_stack((clock,q/2,dq/lim[None,:],relball/4,relcom/2,footv/10))
 contact=np.column_stack((np.eye(4)[stance],np.log1p(feet)/8,last_change/.6))
 assert clock.shape[1]==6 and base.shape[1]==73 and contact.shape[1]==7
 ids=np.arange(0,n-HORIZON,STRIDE)
 nextids=ids+HORIZON
 Y=np.column_stack((dq[nextids]/lim[None,:],np.eye(4)[stance[nextids]],
                    (np.isfinite(impact) if impact is not None else False)*
                    ((t[nextids]>=impact) & (t[ids]<impact)).astype(float) if impact is not None
                    else np.zeros(len(ids))))
 assert Y.shape==(len(ids),34)
 mask=(t[nextids] >= impact-.40)&(t[nextids] <= impact+1.25) if impact is not None else np.zeros(len(ids),bool)
 return dict(case=case,source_sha256=cert["raw_sha256"],certification=bool(cert["true_original_200Hz_G1_CUDA_PhysX"]),
  t=t,ids=ids,when_future=t[nextids],clock=clock[ids],base=base[ids],contact=contact[ids],
  y=Y,stance=stance[ids],full_stance=stance,near_contact=mask,
  joint_names=names,qdot_limit=lim,impact_time_s=impact,annotation_rows=annotations,
  real_200hz_frames=n,nonfoot_peak_N=float(nonfoot.max()),
  original_inst_fraction=float(raw["records"][0]["requested_surface_impulse_fraction"]),
  original_goal=bool(raw["records"][0]["goal"]),original_recovered=bool(raw["records"][0]["recovered"]),
  original_physx_complete=bool(len(t)==2800),
  original_hard_gate_pass=bool(cert["gates"]["hard_gate_pass"]))

def standardized(X,mean=None,std=None):
 X=np.asarray(X,dtype=float)
 if mean is None:mean=X.mean(axis=0)
 if std is None:std=X.std(axis=0)
 std=np.where(std<1e-3,1,std)
 return (X-mean)/std,mean,std

def fit_ridge(train,feature,permute=False):
 X=np.concatenate([z[feature] if feature!="full" else np.column_stack((z["base"],z["contact"])) for z in train])
 Y=np.concatenate([z["y"] for z in train])
 if permute:
  rng=np.random.default_rng(9101)
  parts=[]
  for z in train:
   x=np.column_stack((z["base"],z["contact"]))
   j=rng.permutation(len(x))
   x[:,-7:]=x[j,-7:]
   parts.append(x)
  X=np.concatenate(parts)
 Z,mu,std=standardized(X)
 Z=np.column_stack((np.ones(len(Z)),Z))
 reg=np.eye(Z.shape[1])*ALPHA
 reg[0,0]=0
 B=np.linalg.solve(Z.T@Z+reg,Z.T@Y)
 return dict(weight=B,mean=mu,std=std,feature=feature,permuted=permute,train_rows=len(Z))
def predict(model,z):
 X=z[model["feature"]] if model["feature"]!="full" else np.column_stack((z["base"],z["contact"]))
 return np.column_stack((np.ones(len(X)),(X-model["mean"])/model["std"]))@model["weight"]
def persist(z):
 Y=np.zeros_like(z["y"])
 Y[:,:29]=z["base"][:,35:64] # 6 clock + 29 joint positions = 35
 Y[:,29:33]=np.eye(4)[z["stance"]]
 return Y
def score(y,p,mask):
 if not np.any(mask):return None
 actual=y[mask]; pred=p[mask]
 E=np.abs(actual[:,:29]-pred[:,:29])
 truth=actual[:,29:33].argmax(axis=1)
 guessed=pred[:,29:33].argmax(axis=1)
 cls=[]
 for k in range(4):
  tp=np.sum((truth==k)&(guessed==k))
  fp=np.sum((truth!=k)&(guessed==k))
  fn=np.sum((truth==k)&(guessed!=k))
  # macro F1 over present true classes only to avoid artificially penalizing missing classes
  if not np.any(truth==k):continue
  cls.append(float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else 0.)
 impact=actual[:,33]
 return dict(n=int(len(actual)),joint_qdot_ratio_mae=float(E.mean()),
    joint_qdot_ratio_p95_abs_error=float(np.percentile(E,95)),
    support_accuracy=float(np.mean(truth==guessed)),
    support_macro_f1_present_classes=float(np.mean(cls)) if cls else None,
    impact_positive_count=int(np.sum(impact)),
    impact_squared_error=float(np.mean((impact-pred[:,33])**2)),
    predicted_joint_qdot_ratio_over_limit_fraction=float(np.mean(np.abs(pred[:,:29])>1.)))

def coordination(z,p):
 if z["impact_time_s"] is None:return None
 selection=(z["when_future"]>=z["impact_time_s"]-.4)&(z["when_future"]<=z["impact_time_s"])
 if selection.sum()<10:return None
 t=z["when_future"][selection]
 def peak(sequence,group):
  idx=[z["joint_names"].index(k) for k in group]
  E=np.sum(np.square(sequence[selection][:,idx]),axis=1)
  return float(t[np.argmax(E)]-z["impact_time_s"])
 true={k:peak(z["y"][:,:29],ids) for k,ids in GROUPS.items()}
 pred={k:peak(p[:,:29],ids) for k,ids in GROUPS.items()}
 return dict(true_group_peak_relative_contact_s=true,
    pred_group_peak_relative_contact_s=pred,
    group_peak_timing_mae_s=float(np.mean([abs(true[k]-pred[k]) for k in GROUPS])),
    positive_proximal_distal_order_verified_from_human=False,
    physically_observed_robot_kinematic_proxy_only=True)

def experiment(train,test):
 if set(z["case"] for z in train)&set(z["case"] for z in test):raise ValueError("TRAIN_EVAL_EPISODE_LEAKAGE")
 if not all(z["certification"] for z in [*train,*test]):raise ValueError("UNCERTIFIED_PHYSX")
 models={
  "clock_only_ridge":fit_ridge(train,"clock"),
  "state_only_ridge":fit_ridge(train,"base"),
  "contact_conditioned_ridge":fit_ridge(train,"full"),
  "contact_feature_shuffled_negative":fit_ridge(train,"full",permute=True),
 }
 out=[]
 for z in test:
  summary={"case":z["case"],"sha256":z["source_sha256"],
    "real_200hz_frames":z["real_200hz_frames"],"terminal_annotation_rows":z["annotation_rows"],
    "actual_impact_time_s":z["impact_time_s"],"full_original_physx_14s":z["original_physx_complete"],
    "contact_pre0p4_post1p25_window_complete":bool(z["impact_time_s"] is not None and z["t"][0]<=z["impact_time_s"]-.4+1e-7 and z["t"][-1]>=z["impact_time_s"]+1.25-1e-7),
    "contact_window_metric_is_censored_if_incomplete":True,
    "physical_hard_gate_pass":z["original_hard_gate_pass"],"nonfoot_peak_N":z["nonfoot_peak_N"],
    "instep_fraction":z["original_inst_fraction"],"goal":z["original_goal"],
    "recovered":z["original_recovered"],"metrics":{}}
  for name,model in [("velocity_persistence",None),*models.items()]:
   p=persist(z) if model is None else predict(model,z)
   summary["metrics"][name]={
    "full_available_trace":score(z["y"],p,np.ones(len(p),bool)),
    "near_first_contact_-0p4_to_+1p25s":score(z["y"],p,z["near_contact"]),
    "wholebody_peak_timing":coordination(z,p)}
  out.append(summary)
 meta=dict(training_cases=[z["case"] for z in train],
   heldout_cases=[z["case"] for z in test],
   training_original_sha256={z["case"]:z["source_sha256"] for z in train},
   eval_original_sha256={z["case"]:z["source_sha256"] for z in test},
   model_parameters_local_only=True,
   data="actual_original_G1_200Hz_PhysX_robot_motion_not_human",
   future_true_impacts_as_input=False,weight_alpha=ALPHA,
   actuation_blocked_native_original_G1_collider_geometry_unqualified=True,
   source_sha_verified=True,metrics=out)
 return meta,models

def run(phase,out):
 train=[load(x) for x in TRAIN]
 if phase=="holdout":
  test=[load(x,source="v91") for x in TEST]
 else:
  train=[load(x) for x in TRAIN[:2]]
  test=[load(x) for x in TRAIN[2:]]
 summary,models=experiment(train,test)
 summary["phase"]=phase
 with out.open("x") as f:json.dump(summary,f,indent=2);f.write("\n")
 modelpath=out.with_suffix(".npz")
 weights={}
 for k,m in models.items():
  for field in ("weight","mean","std"):
   weights[k+"_"+field]=m[field]
 np.savez(modelpath,**weights)
 print("V091_OBSERVER_SUMMARY",json.dumps({"phase":phase,"train":summary["training_cases"],"eval":summary["heldout_cases"],
  "metrics":[{"case":z["case"],"pass":z["physical_hard_gate_pass"],
               "near_contact":{k:v["near_first_contact_-0p4_to_+1p25s"]["joint_qdot_ratio_mae"] if v["near_first_contact_-0p4_to_+1p25s"] else None
                 for k,v in z["metrics"].items()}} for z in summary["metrics"]]}),flush=True)
 return summary
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--phase",choices=("development","holdout"),required=True)
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 if a.output.exists() or a.output.with_suffix(".npz").exists():raise FileExistsError("WRITE_ONCE_OBSERVER")
 run(a.phase,a.output)
if __name__=="__main__":main()
