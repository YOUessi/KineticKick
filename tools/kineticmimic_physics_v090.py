#!/usr/bin/env python3
"""KineticMimic v90: immutable, event-anchored original-G1 PhysX evidence.

Reads the *actual* original 29DoF free-root G1 5ms simulation states (v84/v85),
including actual q/dq/COM/stance, first physically measured foot-ball contact,
original instep impulse, foot speed and nonfoot failures. A +1us terminal marker
is NOT a new 200Hz physics sample. A net contact force is NOT collider identity.

This module NEVER invents human true impact/GRF nor simulates robot control.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,math
from pathlib import Path
import numpy as np
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

FREQUENCY_HZ=200.
DT=.005
N_JOINTS=29
LOAD_THRESHOLD_N=80.
FOOT_MODES=("airborne","left","right","double")
@dataclass(frozen=True)
class G1ContactEpisode:
    case:str
    trace_source_sha256:str
    source_n_rows:int
    true_physics_rows:int
    terminal_annotation_count:int
    times:np.ndarray
    q:np.ndarray
    dq:np.ndarray
    qdot_limits:np.ndarray
    qdot_ratio:np.ndarray
    feet_normal_N:np.ndarray
    stance_state:np.ndarray
    nonfoot_force_N:np.ndarray
    root_xyz:np.ndarray
    com_xyz:np.ndarray
    right_foot_point_velocity_w:np.ndarray
    ball_xyz:np.ndarray
    ball_velocity_xyz:np.ndarray
    foot_ball_force_N:np.ndarray
    first_impact_s:float | None
    measured_instep_impulse_fraction:float
    original_ball_goal:bool
    original_recovered:bool
    original_termination:bool
    estimated_foot_ball_scalar_impulse_Ns:float
    observed_ball_speed_5_to_105ms_mps:float | None
    original_reported_impact_Ns_by_surface:dict
    original_controller_verdict:bool

def sha(path:Path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def stance_class(foot_load_N,threshold=LOAD_THRESHOLD_N):
 """Four mutually exclusive states, not a contact patch/polygon reconstruction."""
 F=np.asarray(foot_load_N,dtype=float)
 if F.ndim!=2 or F.shape[1]!=2 or not np.isfinite(F).all() or np.any(F<0):
  raise ValueError("PhysX original bilateral normal forces must be finite nonnegative Nx2")
 l=F[:,0]>=threshold;r=F[:,1]>=threshold
 return (l.astype(np.int64)+2*r.astype(np.int64))

def strict_physics(episode:G1ContactEpisode,required_ball_speed=3.5):
 """Independent complete-gate score; no pass if shot success alone / terminal."""
 full=bool(len(episode.times)==2800 and abs(float(episode.times[-1])-14.)<=.005001)
 speed=episode.observed_ball_speed_5_to_105ms_mps
 gates=dict(
  original_29dof_joint_order_and_real_velocity=bool(
    episode.q.shape==(len(episode.times),29) and
    episode.dq.shape==episode.q.shape and
    (episode.qdot_ratio<=1.000001).all()),
  true_physics_impact_observed=episode.first_impact_s is not None,
  actual_instep_fraction_ge_0p95=bool(episode.measured_instep_impulse_fraction>=.95),
  original_goal=episode.original_ball_goal,
  observed_ball_speed_ge_3p5=bool(speed is not None and speed>=required_ball_speed),
  no_nonfoot_force_ge_20N=bool(np.max(episode.nonfoot_force_N)<20.),
  original_physics_full_14s=full,
  original_recovery_certified=episode.original_recovered and not episode.original_termination,
 )
 overall=all(gates.values())
 if overall and not episode.original_controller_verdict:
  # Sometimes original wrappers are permissive; never promote beyond wrapper
  # OR independent strict gate. Report disagreement rather than faking pass.
  overall=False
  gates["original_wrapper_verdict"]=False
 return dict(strict_pass=overall,gates=gates,
  failed=[k for k,v in gates.items() if not v],three_dimensional_ball_speed_proxy_not_human_power=True)

def _finite(a,shape,label):
 out=np.asarray(a,dtype=np.float64)
 if out.shape!=shape or not np.isfinite(out).all():
  raise ValueError("Original G1 observed "+label+" missing, invalid or nonfinite")
 return out

def load_original_physics(path:Path,expected_sha:str,case:str):
 if not path.exists() or sha(path)!=expected_sha:
  raise ValueError("Original G1 frozen raw physical JSON SHA drift: "+str(path))
 raw=json.loads(path.read_text())
 if (raw.get("root_fixed") is not False or
     raw.get("post_initialization_state_writes") is not False or
     tuple(raw.get("joint_names",()))!=G1_JOINT_ORDER or
     int(raw.get("seed",-1))<=0):
  raise ValueError("Not original G1 free-root true 29DoF report")
 rows=raw.get("traces",[None])[0]
 if not isinstance(rows,list) or len(rows)<200:raise ValueError("Insufficient real G1 physical source")
 annotation=0
 if len(rows)>=2:
  last=rows[-1];before=rows[-2]
  if abs(float(last["time_s"])-float(before["time_s"])-1e-6)<=1e-7:
   # Precise non-physics end marker only when EVERY shared physical state is
   # identical; never silently trim a real physical time step.
   physical_fields=("joint_pos","joint_vel","ball_xyz","root_xyz","whole_body_com_xyz")
   if (not last.get("terminated") or
       any(last.get(k)!=before.get(k) for k in physical_fields)):
    raise ValueError("Nonphysical termination marker alters real observed state")
   annotation=1
   rows=rows[:-1]
 times=_finite([r["time_s"] for r in rows],(len(rows),),"200Hz time")
 if not np.allclose(np.diff(times),DT,atol=1e-6,rtol=0):
  raise ValueError("No genuinely consecutive original 5ms simulator states")
 if abs(times[0]-DT)>1e-6:raise ValueError("Original physics does not begin at real 5ms state")
 N=len(rows)
 q=_finite([r["joint_pos"] for r in rows],(N,29),"measured joint position")
 dq=_finite([r["joint_vel"] for r in rows],(N,29),"measured joint velocity")
 limits=_finite(raw["joint_velocity_limits"],(1,29),"original velocity limits")[0]
 if np.any(limits<=0):raise ValueError("Original G1 joint limit corrupt")
 feet=_finite([[r["support_foot_force_N"]["left"],r["support_foot_force_N"]["right"]]
              for r in rows],(N,2),"actual foot support normal")
 nonfoot=_finite([r["nonfoot_contact_N"] for r in rows],(N,),"actual nonfoot net force")
 root=_finite([r["root_xyz"] for r in rows],(N,3),"true robot floating root")
 com=_finite([r["whole_body_com_xyz"] for r in rows],(N,3),"whole-body COM")
 vel=_finite([r["right_foot_point_velocity"] for r in rows],(N,3),"actual right foot point velocity")
 ball=_finite([r["ball_xyz"] for r in rows],(N,3),"ball 3D position")
 ballv=_finite([r["ball_velocity"] for r in rows],(N,3),"ball 3D velocity")
 forces=[]
 for r in rows:
  labelled=r["robot_ball_contact_force_N"]
  if not isinstance(labelled,dict):raise ValueError("Original robot-ball contact force not a named dictionary")
  if any(not x.endswith("ankle_roll_link") for x in labelled):
   raise ValueError("Nonfoot body/ball contact cannot be renamed as legal foot")
  forces.append(sum(float(v) for v in labelled.values()))
 football=_finite(forces,(N,),"original foot-ball force magnitude")
 if np.any(football<0) or np.any(nonfoot<0):
  raise ValueError("Measured contact magnitudes cannot be negative")
 record=raw["records"][0]
 c=record.get("contact_time_s")
 measured=np.where(football>5.)[0]
 if c is not None:
  if len(measured)==0 or abs(float(c)-float(times[measured[0]]))>.006:
   raise ValueError("Reported foot-ball contact not found in actual 200Hz force source")
  contact=float(times[measured[0]])
 else:
  if len(measured):raise ValueError("Missing reported contact time despite physical foot-ball force")
  contact=None
 fraction=float(record.get("requested_surface_impulse_fraction",0))
 if not 0<=fraction<=1:raise ValueError("Unphysical original instep proportion")
 speed=None
 if contact is not None:
  sel=(times>=contact+.005-1e-7)&(times<=contact+.105+1e-7)
  if sel.any():speed=float(np.linalg.norm(ballv[sel],axis=1).max())
 # Unsigned summed magnitude dt is scalar impulse proxy, not a vector J/momentum.
 impulse=float(DT*np.sum(football))
 return G1ContactEpisode(
  case=case,trace_source_sha256=sha(path),
  source_n_rows=N+annotation,true_physics_rows=N,terminal_annotation_count=annotation,
  times=times,q=q,dq=dq,qdot_limits=limits,
  qdot_ratio=np.max(abs(dq)/limits[None],axis=1),
  feet_normal_N=feet,stance_state=stance_class(feet),nonfoot_force_N=nonfoot,
  root_xyz=root,com_xyz=com,right_foot_point_velocity_w=vel,
  ball_xyz=ball,ball_velocity_xyz=ballv,foot_ball_force_N=football,
  first_impact_s=contact,measured_instep_impulse_fraction=fraction,
  original_ball_goal=bool(record.get("goal")),original_recovered=bool(record.get("recovered")),
  original_termination=bool(any(bool(x) for x in raw.get("termination_reasons",[]))),
  estimated_foot_ball_scalar_impulse_Ns=impulse,
  observed_ball_speed_5_to_105ms_mps=speed,
  original_reported_impact_Ns_by_surface=dict(record.get("contact_impulse_by_surface_Ns",{})),
  original_controller_verdict=bool(raw.get("stride_shot_success")),
 )

def sample_contact_predictor(episode,history=12,stride=4,horizon=20,precontact_only=True):
 """Prepare 50Hz candidate sensor windows from REAL G1 PhysX, no future input.
 Each history sequence is 12 consecutive 5ms states (60ms). The future label
 is actual impact within the next 20x5ms, never a guessed human timestamp.
 """
 if history<4 or horizon<1 or stride<1:raise ValueError("Bad physics sampling horizon")
 n=episode.true_physics_rows
 if n<history+horizon:raise ValueError("Not enough genuine physics frames")
 source=np.concatenate([
  episode.q/2.,
  episode.dq/episode.qdot_limits[None],
  np.log1p(episode.feet_normal_N)/8.,
  (episode.ball_xyz-episode.root_xyz)/4.,
  episode.right_foot_point_velocity_w/10.,
  (episode.com_xyz-episode.root_xyz)/2.,
  episode.qdot_ratio[:,None],
  np.log1p(episode.nonfoot_force_N[:,None])/8.,
 ],axis=1)
 if source.shape[1]!=71:raise ValueError("G1 physical state feature count changed")
 peak=np.argmax(episode.foot_ball_force_N)
 if episode.first_impact_s is None:peak=-1
 windows=[];future=[];stance=[];current_stance=[];times=[]
 for end in range(history-1,n-horizon,stride):
  if precontact_only and episode.first_impact_s is not None and episode.times[end]>=episode.first_impact_s-1e-7:
   # A future-impact prediction must never inspect any post-impact state.
   break
  start=end-history+1
  val=source[start:end+1]
  if not np.isfinite(val).all():raise ValueError("Original G1 state not finite")
  # Current support regime is already observed, future target predicted.
  future_impact=bool(peak>end and peak<=end+horizon)
  future_stance=int(episode.stance_state[end+horizon])
  windows.append(val)
  future.append(float(future_impact))
  stance.append(future_stance)
  current_stance.append(int(episode.stance_state[end]))
  times.append(float(episode.times[end]))
 if len(windows)<20:raise ValueError("No measured contact/stance prediction windows")
 return dict(
  x=np.asarray(windows,dtype=np.float32),
  impact_next_100ms=np.asarray(future,dtype=np.float32),
  stance_after_100ms=np.asarray(stance,dtype=np.int64),
  current_stance=np.asarray(current_stance,dtype=np.int64),
  query_time_s=np.asarray(times,dtype=np.float64),
  source_SHA256=episode.trace_source_sha256,
  source_case=episode.case,
 )
