#!/usr/bin/env python3
"""SHA-certified independent 200Hz 29DoF support/contact/coupling forensic audit.

NO force-plate GRF, no human impulse, no mesh pair, no causal power inference.
Censored early termination never receives artificial 14s recovery samples.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from wholebody_safety_interface_v055 import G1_JOINT_ORDER
from v083_phase_chain_candidate import offline_plan
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports/v083_final_real_physx.audit.json"
OUT=ROOT/"reports/v083_independent_coupling.audit.json"
GROUPS={
 "support_leg":["left_hip_pitch_joint","left_hip_roll_joint","left_knee_joint","left_ankle_pitch_joint"],
 "swing_leg":["right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint"],
 "torso":["waist_yaw_joint","waist_pitch_joint","waist_roll_joint"],
 "counter_arm":["left_shoulder_pitch_joint","right_shoulder_pitch_joint","left_elbow_joint","right_elbow_joint"]
}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for part in iter(lambda:f.read(1<<20),b""):h.update(part)
 return h.hexdigest()
def _finite(name,v):
 if not np.isfinite(v).all():raise ValueError("NAN_INF_ACTUAL_PHYSICS_"+name)
def audit_one(meta,raw):
 rows=raw["traces"][0]
 if tuple(raw["joint_names"])!=G1_JOINT_ORDER:
  raise ValueError("ORIGINAL_29DOF_JOINT_ORDER_MISMATCH")
 t=np.asarray([x["time_s"] for x in rows],float)
 qdot=np.asarray([x["joint_vel"] for x in rows],float)
 limits=np.asarray(raw["joint_velocity_limits"],float).reshape(-1)
 com=np.asarray([x["whole_body_com_xyz"] for x in rows],float)
 foot=np.asarray([[x["support_foot_force_N"]["left"],
                   x["support_foot_force_N"]["right"]] for x in rows],float)
 nonfoot=np.asarray([x["nonfoot_contact_N"] for x in rows],float)
 if (qdot.shape!=(len(t),29) or com.shape!=(len(t),3) or
        foot.shape!=(len(t),2) or limits.shape!=(29,)):
  raise ValueError("MISSING_29DOF_200HZ_OR_SUPPORT")
 for k,a in dict(t=t,qdot=qdot,lim=limits,com=com,foot=foot,nonfoot=nonfoot).items():_finite(k,a)
 if len(t)<100 or not np.all(np.diff(t)>0) or np.any(limits<=0):
  raise ValueError("INVALID_REAL_PHYSICS_TIME_OR_JOINT_LIMITS")
 # Pure 5ms frames plus possible microsecond terminal sample are retained
 # but never counted as additional 200Hz integration steps.
 true_steps=int(np.sum(np.isclose(np.diff(t),.005,atol=1e-6)))
 hit=raw["records"][0].get("contact_time_s")
 windows={}
 if hit is not None:
  for name,(a,b) in {"before":(-.4,0.),"after":(0.,1.25)}.items():
   expected=80 if name=="before" else 250
   selector=(t>=hit+a-1e-6)&(t<=hit+b+1e-6)
   windows[name]={"real_samples":int(selector.sum()),
                  "observed_full_window":bool(t[0]<=hit+a+.005 and t[-1]>=hit+b-.005),
                  "max_actual_qdot_ratio":float(np.abs(qdot[selector]/limits).max()) if selector.any() else None,
                  "mean_left_support_sensor_N":float(foot[selector,0].mean()) if selector.any() else None,
                  "mean_right_support_sensor_N":float(foot[selector,1].mean()) if selector.any() else None}
 else:windows={"before":None,"after":None}
 # Kinodynamic association (rate magnitudes and lag), not impulse/force causality.
 freq=1/.005
 groups={}
 for name,names in GROUPS.items():
  ix=[G1_JOINT_ORDER.index(s) for s in names]
  power=np.linalg.norm(qdot[:,ix],axis=1)
  groups[name]={"time_peak_absolute_group_joint_rate_s":float(t[np.argmax(power)]),
                "rate_rms_rad_s":float(np.sqrt(np.mean(qdot[:,ix]**2))),
                "mean_absolute_group_joint_rate_rad_s":float(np.mean(np.abs(qdot[:,ix]))),
                "max_abs_group_joint_rate_rad_s":float(np.max(np.abs(qdot[:,ix])))}
 ratio=np.max(np.abs(qdot)/limits,axis=1)
 signal=np.asarray([np.linalg.norm(qdot[:,[G1_JOINT_ORDER.index(k) for k in GROUPS[n]]],axis=1)
                     for n in ("torso","swing_leg","counter_arm")])
 correlated={}
 contact_window=(t>=hit-.4)&(t<=hit+1.25) if hit is not None else np.zeros(len(t),dtype=bool)
 if contact_window.sum()>80:
  for x,y in ((0,1),(0,2),(1,2)):
   a=signal[x,contact_window];b=signal[y,contact_window]
   a=a-a.mean();b=b-b.mean()
   vals=[]
   # +/-0.2s lag bounded and denominators reject constant signals.
   for lag in range(-40,41):
    aa=a[max(lag,0):len(a)+min(lag,0)]
    bb=b[max(-lag,0):len(b)+min(-lag,0)]
    d=np.linalg.norm(aa)*np.linalg.norm(bb)
    vals.append(float(aa@bb/d) if d>1e-8 else float("nan"))
   valid=[(lag-40,v) for lag,v in enumerate(vals) if np.isfinite(v)]
   correlated[f"{('torso','swing_leg','counter_arm')[x]}_vs_{('torso','swing_leg','counter_arm')[y]}"]=(
    {"lag_s_at_max_correlation":float(max(valid,key=lambda x:x[1])[0]/freq),
     "max_normalized_correlation":float(max(v for _,v in valid))}
    if valid else None)
 return {"case":meta["case"],"scenario":meta["scenario"],
    "certified_original_G1":True,"raw_sha256":meta["raw_sha256"],
    "actual_samples":len(t),"actual_5ms_physics_intervals":true_steps,
    "last_real_sample_s":float(t[-1]),"complete_real_14s":bool(t[-1]>=13.995),
    "actual_29qdot_limit_peak_ratio":float(ratio.max()),
    "actual_29qdot_limit_violation_frames":int((ratio>1.000001).sum()),
    "peak_nonfoot_net_contact_N":float(nonfoot.max()),
    "measured_support_foot_load_mean_N":foot.mean(axis=0).tolist(),
    "both_foot_sensor_gt10_N_fraction":float(np.mean((foot[:,0]>10)&(foot[:,1]>10))),
    "zero_foot_sensor_gt10_N_fraction":float(np.mean((foot[:,0]<=10)&(foot[:,1]<=10))),
    "com_height_min_max_m":[float(com[:,2].min()),float(com[:,2].max())],
    "actual_ball_contact_s":hit,"contact_event_windows":windows,
    "actual_group_qdot":groups,"bounded_joint_rate_cross_correlations_not_causality":correlated,
    "measured_nonfoot_body_events":len(raw.get("nonfoot_contact_diagnostics",[])),
    "strict_gates":meta["strict_gates"],
    "native_manifold_true":False,"humanGRF_true":False,
    "support_force_is_sensor_net_NOT_COP_or_forceplate":True}
def audit():
 if OUT.exists():raise FileExistsError("NO_REWRITE_INDEPENDENT_AUDIT")
 report=json.loads(REPORT.read_text())
 out=[]
 for meta in report["runs"]:
  if not meta["true_physics"]:continue
  path=ROOT/"reports"/(meta["run"]+".json")
  log=ROOT/"reports"/(meta["run"]+".log")
  if sha(path)!=meta["raw_sha256"] or sha(log)!=meta["log_sha256"]:
   raise RuntimeError("SOURCE_200HZ_JSON_OR_STDOUT_DRIFT")
  raw=json.loads(path.read_text())
  out.append(audit_one(meta,raw))
 plan=offline_plan()
 summary={"version":"v0.1.83","protocol_sha256":report["protocol_sha256"],
   "real_200hz_g1_episodes":len(out),
   "genuine_strict_pass":sum(x["strict_gates"]["hard_gate_pass"] for x in out),
   "right_censored_early_end":sum(not x["complete_real_14s"] for x in out),
   "actual_full_event_window_count":sum(bool(x["contact_event_windows"]["before"] and
     x["contact_event_windows"]["before"]["observed_full_window"] and
     x["contact_event_windows"]["after"]["observed_full_window"]) for x in out),
   "native_manifold_certified":0,
   "new_wholebody_candidate":"UNTRAINED_ONLY_NOT_DEPLOYED",
   "candidate_ablation_offline_nonphysical":plan,
   "episodes":out}
 with OUT.open("x") as f:json.dump(summary,f,indent=2);f.write("\n")
 print("V083_INDEPENDENT",len(out),"strict",summary["genuine_strict_pass"],
       "censored",summary["right_censored_early_end"],
       "complete_event_windows",summary["actual_full_event_window_count"],flush=True)
 return summary

if __name__=="__main__":audit()
