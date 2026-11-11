#!/usr/bin/env python3
"""Evidence-graded 120Hz CMU geometry and real 200Hz G1 joint-phase audit.

No human GRF, instep impulse, contact time, professional identity or naturalness
ground truth is claimed from optical marker closest-approach geometry.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
GROUPS={
 "pelvis":("RTHI","LTHI"),
 "torso":("CLAV","PEL0"),
 "swing_right":("RTOE","RTHI"),
 "support_left":("LTOE","LTHI"),
 "counter_arm_left":("LWRA","LSHO"),
 "other_arm_right":("RWRA","RSHO")
}
ROBOT_GROUPS={
 "pelvis":("left_hip_yaw_joint","right_hip_yaw_joint","left_hip_roll_joint","right_hip_roll_joint"),
 "torso":("waist_yaw_joint","waist_roll_joint","waist_pitch_joint"),
 "swing_right":("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"),
 "support_left":("left_hip_pitch_joint","left_hip_roll_joint","left_knee_joint"),
 "counter_arm_left":("left_shoulder_pitch_joint","left_shoulder_roll_joint","left_elbow_joint","left_wrist_yaw_joint")
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def speed_peak(signal,t,center,pre_s=.4):
 good=(t>=center-pre_s)&(t<=center)&np.isfinite(signal)
 if good.sum()<8:return None
 indices=np.where(good)[0]
 i=indices[np.argmax(signal[indices])]
 return dict(relative_to_candidate_s=round(float(t[i]-center),6),
             speed_proxy_mps=round(float(signal[i]),5),
             samples=int(good.sum()))
def human_clip(p,base):
 import ezc3d
 data=ezc3d.c3d(str(p))
 arr=np.asarray(data["data"]["points"],dtype=float)
 labels=list(data["parameters"]["POINT"]["LABELS"]["value"])
 names={name.rsplit(":",1)[-1].upper():i for i,name in enumerate(labels)}
 fps=float(data["parameters"]["POINT"]["RATE"]["value"][0])
 units=str(data["parameters"]["POINT"]["UNITS"]["value"][0]).strip().lower()
 if units not in ("mm","m"):raise ValueError("C3D_UNITS_UNQUALIFIED:"+units)
 scale=.001 if units=="mm" else 1.
 need=set(("RTOE","LTOE","RANK","LANK","CLAV","PEL0","RTHI","LTHI",
           "LSHO","RSHO","LWRA","RWRA","WEP1","WEP2","WEP3"))
 missing=sorted(need-set(names))
 n=arr.shape[2];t=np.arange(n)/fps
 def marker(k):
  q=arr[:3,names[k],:].T*scale
  residual=arr[3,names[k],:]
  q[~np.isfinite(q).all(axis=1)|(residual<0)]=np.nan
  return q
 report=dict(file=str(p.relative_to(base)),sha256=sha(p),frames=n,rate_hz=fps,
   c3d_point_units=units,required_missing=missing,
   synchronized_real_GRF=False,verified_professional_player=False,
   measured_foot_ball_impulse=False,shot_contact_time_ground_truth=False,
   marker_geometry_is_synthetic=False)
 if missing:return report
 M={k:marker(k) for k in need}
 stacked=np.stack([M["WEP1"],M["WEP2"],M["WEP3"]])
 count=np.isfinite(stacked).all(axis=2).sum(axis=0)
 ball=np.full_like(M["WEP1"],np.nan)
 good=count>0
 ball[good]=np.nansum(stacked[:,good,:],axis=0)/count[good,None]
 toe=M["RTOE"]
 dist=np.linalg.norm(toe-ball,axis=1)
 eligible=np.isfinite(dist)&(t>.45)&(t<t[-1]-1.3)
 report["ball_and_right_toe_visible_fraction"]=round(float(np.isfinite(dist).mean()),5)
 if eligible.sum()<30:
  report["candidate_status"]="NO_FULL_GEOMETRIC_EVENT_WINDOW"
  return report
 index=np.where(eligible)[0][np.argmin(dist[eligible])]
 center=float(t[index])
 report["candidate_status"]="GEOMETRIC_CLOSEST_APPROACH_ONLY"
 report["candidate_time_s"]=round(center,6)
 report["candidate_right_toe_ball_marker_centroid_distance_m"]=round(float(dist[index]),5)
 report["candidate_ball_center_definition"]="mean of three surface markers, not calibrated real ball center"
 groups={}
 for name,(first,second) in GROUPS.items():
  if name=="pelvis":
   path=(M["RTHI"]+M["LTHI"])/2
  else:path=M[first]-M[second]
  # Smooth 5-frame position before differentiating; speeds are relative
  # marker geometry, NOT joint angular velocity or GRF.
  smooth=np.full_like(path,np.nan)
  for i in range(2,len(path)-2):
   w=path[i-2:i+3]
   if np.isfinite(w).all():smooth[i]=w.mean(axis=0)
  vel=np.linalg.norm(np.gradient(smooth,1/fps,axis=0),axis=1)
  groups[name]=speed_peak(vel,t,center)
 report["relative_marker_velocity_peak_pre0p4_s"]=groups
 report["selected_side"]="right-toe geometric proxy, not confirmed kicking foot"
 report["phase_learning_eligible"]=bool(all(v is not None for v in groups.values()) and
   dist[index]<.30 and report["ball_and_right_toe_visible_fraction"]>.90)
 report["qualification_rule"]="candidate geometric closeness <0.30m, >90% visibility, valid phase groups; still NOT verified instep contact"
 return report
def robot_case(case,root):
 rawpath=root/"reports"/("v091_"+case+".json")
 auditpath=root/"reports"/("v091_"+case+".audit.json")
 cert=json.loads(auditpath.read_text())
 if not cert["true_original_200Hz_G1_CUDA_PhysX"] or sha(rawpath)!=cert["raw_sha256"]:
  raise ValueError("UNVERIFIED_REAL_G1_SOURCE:"+case)
 raw=json.loads(rawpath.read_text())
 seq=raw["traces"][0];annotations=0
 if len(seq)>1 and abs(seq[-1]["time_s"]-seq[-2]["time_s"]-.000001)<1e-7:
  if not seq[-1]["terminated"]:raise ValueError("BAD_TERMINATION_ROW")
  seq=seq[:-1];annotations=1
 t=np.asarray([x["time_s"] for x in seq],dtype=float)
 qdot=np.asarray([x["joint_vel"] for x in seq],dtype=float)
 if qdot.shape!=(len(t),29) or np.max(np.abs(np.diff(t)-.005))>1e-6:
  raise ValueError("NOT_FULL_200HZ_OR_29DOF")
 names=list(raw["joint_names"])
 limit=np.asarray(raw["joint_velocity_limits"],dtype=float).reshape(-1)
 if len(limit)!=29:raise ValueError("LIMIT_MISMATCH")
 event=raw["records"][0]["contact_time_s"]
 peak=float(np.max(np.abs(qdot)/limit[None,:]))
 groups={}
 if event is not None:
  for key,member in ROBOT_GROUPS.items():
   ind=[names.index(x) for x in member]
   signal=np.sqrt((qdot[:,ind]**2).sum(axis=1))
   groups[key]=speed_peak(signal,t,event)
 feet=np.asarray([[r["support_foot_force_N"][side] for side in ("left","right")] for r in seq])
 force=np.asarray([r["nonfoot_contact_N"] for r in seq])
 window=bool(event is not None and t[0]<=event-.4+1e-6 and t[-1]>=event+1.25-1e-6)
 return dict(case=case,original_raw_sha256=cert["raw_sha256"],
   certified_original_G1_200Hz=True,real_physics_frames=len(t),
   annotation_rows_not_physics=annotations,
   physical_first_foot_ball_contact_s=event,
   full_pre0p4_post1p25_window=window,
   full_14s_recovery_observed=len(t)==2800,
   nonfoot_peak_N=round(float(force.max()),5),
   measured_29dof_qdot_limit_ratio=round(peak,6),
   actual_left_support_force_peak_N=round(float(feet[:,0].max()),3),
   actual_right_support_force_peak_N=round(float(feet[:,1].max()),3),
   human_naturalness_ground_truth=False,
   group_29dof_speed_peak_relative_to_physical_contact_s=groups,
   all_hard_gates_pass=cert["gates"]["hard_gate_pass"],
   safety_failure_flags=cert["gates"]["failed_gates"])
def assemble(humanroot,g1root):
 paths=sorted(humanroot.rglob("*.c3d"))
 if len(paths)!=6:raise ValueError("EXPECTED_SIX_ORIGINAL_C3D_FILES")
 hum=[human_clip(p,humanroot) for p in paths]
 robots=[robot_case(name,g1root) for name in ("E_v32","E_cmu02","F_v32","F_cmu02")]
 return dict(version="v0.1.92",human_source_clip_count=len(hum),
  human_candidate_count=sum(x.get("phase_learning_eligible",False) for x in hum),
  qualified_human_professional_impact_examples=0,
  human_phase_time_is_NOT_real_impact_contact=True,
  source_subject_group_counts={actor:sum(x["file"].split("/")[0]==actor for x in hum)
   for actor in sorted(set(z["file"].split("/")[0] for z in hum))},
  human=hum,original_G1=robots,original_G1_hard_gate_success=sum(z["all_hard_gates_pass"] for z in robots),
  original_G1_total=len(robots),new_G1_this_batch=0,
  original_G1_truncated_full_window=sum(not z["full_pre0p4_post1p25_window"] for z in robots),
  comparative_naturalness_quantitative_valid=False,
  target_learning_status="DATA_QUALIFICATION_ONLY_NO_EXECUTABLE_HUMAN_MIMIC")
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--human-root",type=Path,default=Path("/home/you/projects/kinetickick/data/research_human_kicks/cmu_soccer_120hz_v050/raw"))
 p.add_argument("--g1-root",type=Path,default=Path("/home/you/projects/kinetickick-wt-v091"))
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error("WRITE_ONCE")
 result=assemble(a.human_root,a.g1_root)
 with a.output.open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
 print("V092_HUMAN_ROBOT_QUALIFICATION",json.dumps({k:result[k] for k in
  ("human_source_clip_count","human_candidate_count","qualified_human_professional_impact_examples",
   "original_G1_hard_gate_success","original_G1_total","original_G1_truncated_full_window")}),flush=True)
if __name__=="__main__":main()
