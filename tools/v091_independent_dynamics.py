#!/usr/bin/env python3
"""Independent exact-SHA 200Hz original G1 29DoF failure attribution and figures.

Never silently promote a ball hitting the goal to a legal instep shot or conflate
terminal +1us annotations with real physical steps, nor body net force with
direct collider/path/manifold evidence.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DT=.005
CASES=("E_v32","E_cmu02","F_v32","F_cmu02")
GROUPS={
 "support_L":("left_hip_pitch_joint","left_knee_joint","left_hip_roll_joint"),
 "pelvis":("left_hip_yaw_joint","right_hip_yaw_joint","left_hip_roll_joint","right_hip_roll_joint"),
 "trunk":("waist_yaw_joint","waist_pitch_joint","waist_roll_joint"),
 "swing_R":("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint"),
 "counterarm_L":("left_shoulder_pitch_joint","left_shoulder_roll_joint","left_elbow_joint"),
}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1048576),b""):h.update(block)
 return h.hexdigest()
def separate_true_rows(rows):
 out=list(rows)
 terminal=[]
 if len(out)>1 and abs(out[-1]["time_s"]-out[-2]["time_s"]-1e-6)<1e-7:
  # no new physical step when all core fields identical
  for f in ("joint_pos","joint_vel","whole_body_com_xyz","ball_xyz","root_xyz","nonfoot_contact_N"):
   if out[-1][f]!=out[-2][f]:raise ValueError("Changed physical state in terminal marker")
  if not out[-1]["terminated"]:raise ValueError("Extra timestamp not terminal")
  terminal=[out.pop()]
 return out,terminal
def check_gates(n,terminal,impact,ratio,frac,goal,ballspeed,nonfoot,recovered):
 gates={
  "true_29dof_speed_limits":bool(ratio<=1.000001),
  "physically_observed_foot_ball_contact":bool(impact is not None),
  "instep_ge_95pct":bool(frac>=.95),
  "ball_goes_through_goal":bool(goal),
  "ball_speed_proxy_ge_3p5mps":bool(ballspeed is not None and ballspeed>=3.5),
  "nonfoot_strict_lt20N":bool(nonfoot<20.),
  "full_14s_original_physx":bool(n==2800),
  "14s_recovery":bool(recovered and n==2800 and not terminal)
 }
 return {"strict_pass":all(gates.values()),"gates":gates,
  "failed":[k for k,v in gates.items() if not v]}
def inspect(case):
 stem="v091_"+case
 rawpath=ROOT/"reports"/(stem+".json")
 auditpath=ROOT/"reports"/(stem+".audit.json")
 audit=json.loads(auditpath.read_text())
 if not audit["true_original_200Hz_G1_CUDA_PhysX"]:raise ValueError("Uncertified raw G1 source")
 if sha(rawpath)!=audit["raw_sha256"]:raise ValueError("SOURCE_DRIFT:"+case)
 a=json.loads(rawpath.read_text())
 if a["root_fixed"] or a["post_initialization_state_writes"]:raise ValueError("Frozen original G1 not free-root")
 rows,term=separate_true_rows(a["traces"][0])
 n=len(rows)
 t=np.asarray([r["time_s"] for r in rows],float)
 if abs(t[0]-.005)>1e-6 or not np.allclose(np.diff(t),DT,atol=1e-6,rtol=0):
  raise ValueError("NONCONTIGUOUS_OR_NOT_200HZ")
 names=tuple(a["joint_names"])
 if len(names)!=29 or len(set(names))!=29:raise ValueError("Not 29 unique joints")
 vel=np.asarray([r["joint_vel"] for r in rows],float)
 pos=np.asarray([r["joint_pos"] for r in rows],float)
 limit=np.asarray(a["joint_velocity_limits"],float).reshape(-1)
 if vel.shape!=(n,29) or pos.shape!=(n,29) or limit.shape!=(29,):raise ValueError("Not 29DoF physical trace")
 if np.any(~np.isfinite(vel)) or np.any(limit<=0):raise ValueError("G1 qdot invalid")
 loads=np.asarray([[r["support_foot_force_N"]["left"],r["support_foot_force_N"]["right"]] for r in rows],float)
 com=np.asarray([r["whole_body_com_xyz"] for r in rows],float)
 root=np.asarray([r["root_xyz"] for r in rows],float)
 ball=np.asarray([r["ball_velocity"] for r in rows],float)
 force=np.asarray([r["nonfoot_contact_N"] for r in rows],float)
 ff=np.asarray([sum(r["robot_ball_contact_force_N"].values()) for r in rows],float)
 torque=np.asarray([r["joint_torque_estimate"] for r in rows],float)
 if loads.shape!=(n,2) or torque.shape!=(n,29) or np.any(loads<0):raise ValueError("Missing body/sensor")
 if not np.isfinite(com).all() or not np.isfinite(ball).all() or not np.isfinite(force).all():raise ValueError("Nonfinite sensor")
 rec=a["records"][0]
 impact=rec["contact_time_s"]
 ids=np.where(ff>5)[0]
 if impact is not None:
  if not len(ids) or abs(t[ids[0]]-impact)>.006:raise ValueError("FOOTBALL_EVENT_NOT_IN_PHYSX")
  impact=float(t[ids[0]])
 speed=None
 if impact is not None:
  mask=(t>=impact+.005-1e-7)&(t<=impact+.11+1e-7)
  if np.any(mask):speed=float(np.linalg.norm(ball[mask],axis=1).max())
 frac=float(rec.get("requested_surface_impulse_fraction",0))
 impulses=rec.get("contact_impulse_by_surface_Ns",{})
 total=sum(impulses.values())
 if total>0:
  reconstructed=float(impulses.get("instep",0)/total)
  if abs(frac-reconstructed)>1e-5:raise ValueError("Reported surface ratio disagrees with impulse evidence")
 ratio=np.max(np.abs(vel)/limit)
 peakN=float(np.max(force))
 passed=check_gates(n,term,impact,ratio,frac,rec["goal"],speed,peakN,rec["recovered"])
 if passed["strict_pass"]!=audit["gates"]["hard_gate_pass"]:
  raise ValueError("INDEPENDENT_STRICT_GATE_DISAGREEMENT:"+case)
 if bool(impact is not None)!=bool(a["sequence_records"][0]["contact"]["observed"]):
  raise ValueError("Original impact timing mismatch")
 support=np.where((loads[:,0]>=80)&(loads[:,1]>=80),3,
          np.where(loads[:,0]>=80,1,np.where(loads[:,1]>=80,2,0)))
 supports={z:float(np.mean(support==k)) for k,z in enumerate(("airborne","left_only","right_only","bilateral"))}
 diagnostic=a.get("nonfoot_contact_diagnostics",[])
 observed=[]
 if diagnostic:
  bytime={}
  for event in diagnostic:
   if event.get("body") and float(event["time_s"])>0:
    name=event["body"];vec=np.asarray(event["force_w_N"],float)
    bytime.setdefault(name,[]).append((event["time_s"],float(np.linalg.norm(vec))))
  observed=sorted(({"body":k,"peak_N":max(x[1] for x in v),
                    "first_reported_time_s":float(min(x[0] for x in v))} for k,v in bytime.items()),
                  key=lambda x:-x["peak_N"])[:4]
 grouppeaks={}
 if impact is not None:
  win=(t>=impact-.4)&(t<=impact)
  for group,jointset in GROUPS.items():
   if win.sum()<5:continue
   idx=[names.index(j) for j in jointset]
   energy=np.square(vel[:,idx]/limit[idx]).sum(axis=1)
   k=np.flatnonzero(win)[int(np.argmax(energy[win]))]
   grouppeaks[group]=round(float(t[k]-impact),4)
 result={
  "case":case,"original_raw_sha256":sha(rawpath),
  "original_certified_PhysX":True,"original_g1_29dof":True,
  "real_200hz_frames":n,"unscored_terminal_annotation_rows":len(term),
  "last_real_physics_time_s":float(t[-1]),"first_foot_ball_contact_s":impact,
  "contact_window_pre0p4_complete":bool(impact is not None and t[0]<=impact-.4+1e-7),
  "contact_window_post1p25_complete":bool(impact is not None and t[-1]>=impact+1.25-1e-7),
  "measured_original_ball_speed_5_to_110ms_proxy_mps":speed,
  "computed_true_instep_impulse_fraction_from_raw_surface_bins":frac,
  "original_max_all29_measured_qdot_ratio":float(ratio),
  "original_nonfoot_max_net_N":peakN,"nonfoot_diagnostic_link_names_only_not_direct_collider_pair":observed,
  "COM_height_min_m":float(com[:,2].min()),"COM_height_max_m":float(com[:,2].max()),
  "root_displacement_xy_m":float(np.linalg.norm(root[-1,:2]-root[0,:2])),
  "foot_support_fraction_80N":supports,"actual_joint_group_peak_relative_first_impact_s":grouppeaks,
  "estimated_motor_tau_times_measured_dq_integral_abs_proxy_J":float(np.sum(np.abs(torque*vel))*DT),
  "estimated_power_proxy_is_not_measured_motor_work":True,
  "direct_actor_collider_manifold_paths_proven":False,
  "independent_gates":passed,
  "original_independent_scorer_failed":audit["gates"]["failed_gates"]}
 return result,(t,vel,limit,loads,com,force,ff,impact)
def plot(case,trace):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 t,vel,limit,loads,com,nonfoot,contactforce,impact=trace
 fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True)
 axes[0].plot(t,loads[:,0],label="left foot ground N")
 axes[0].plot(t,loads[:,1],label="right foot ground N")
 axes[0].legend(loc="upper right");axes[0].set_ylabel("Ground N")
 axes[1].plot(t,com[:,2],label="real wholebody COM z (m)")
 axes[1].legend(loc="upper right");axes[1].set_ylabel("COM z (m)")
 axes[2].plot(t,np.max(np.abs(vel)/limit,axis=1),label="measured max 29 joint |dq|/limit")
 axes[2].axhline(1,linestyle="--",label="qdot hard limit")
 axes[2].legend(loc="upper right");axes[2].set_ylabel("joint ratio")
 axes[3].plot(t,contactforce,label="raw foot-ball force magnitude N")
 axes[3].plot(t,nonfoot,label="raw nonfoot net contact N")
 axes[3].axhline(20,linestyle="--",label="nonfoot hard limit 20N")
 axes[3].legend(loc="upper right");axes[3].set_ylabel("Force N")
 if impact is not None:
  for ax in axes:ax.axvline(impact,linestyle=":",alpha=.8)
 axes[-1].set_xlabel("Original G1 200Hz physical time (s)")
 fig.suptitle("v0.1.91 original G1 real PhysX, NOT learned human kick: "+case)
 fig.tight_layout()
 path=ROOT/"reports"/("v091_physx_"+case+".png")
 if path.exists():raise FileExistsError(path)
 fig.savefig(path,dpi=110)
 plt.close(fig)
 return {"local_path":str(path),"sha256":sha(path)}
def main():
 out=ROOT/"reports/v091_independent_dynamics.audit.json"
 if out.exists():raise FileExistsError("WRITE_ONCE independent dynamics audit")
 entries=[];plots={}
 for c in CASES:
  entry,trace=inspect(c);entries.append(entry);plots[c]=plot(c,trace)
 result={"version":"v0.1.91","independent_raw_SHA_and_G1_physics":True,
  "original_g1_new_physical_episodes":len(entries),
  "strict_pass":sum(e["independent_gates"]["strict_pass"] for e in entries),
  "strict_fail":sum(not e["independent_gates"]["strict_pass"] for e in entries),
  "original_collider_pair_manifold_certified":False,
  "whole_body_human_explosive_kick_learned":False,"plots_local_only":plots,"runs":entries}
 with out.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
 print("V091_INDEPENDENT_DONE",json.dumps({"episodes":len(entries),"pass":result["strict_pass"],"fail":result["strict_fail"],
   "cases":[(e["case"],e["real_200hz_frames"],e["independent_gates"]["failed"]) for e in entries]}),flush=True)
if __name__=="__main__":main()
