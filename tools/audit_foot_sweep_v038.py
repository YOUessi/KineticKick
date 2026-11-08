"""Independent full 14s physical audit for v38 bounded material-point sweep feedback.

Rejects action/schema parity failures even if a goal happens; refuses success
without original real foot-ball contact, full-ball goal, qdot limits and recovery.
Does not rerun a simulator or change a single initial state or acceptance gate.
"""
from pathlib import Path
import argparse,json,hashlib,math,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from strike_baseline_metrics import summarize_strike,validated_episode
from audit_p0_dynamic_strike import limit_observations
from strike_cases import load_cases
from free_shooting_metrics import ShotTracker

def sha(p):
 m=hashlib.sha256()
 with (ROOT/p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):m.update(b)
 return m.hexdigest()

def pose_goal_margin(run):
 rows=validated_episode(run)
 target=float(run["goals"][0][1]-.11)
 for p,q in zip(rows,rows[1:]):
  y1=float(p["ball_xyz"][1])
  y2=float(q["ball_xyz"][1])
  if y1>target>=y2:
   frac=(target-y1)/(y2-y1)
   x=float(p["ball_xyz"][0]+frac*(q["ball_xyz"][0]-p["ball_xyz"][0]))
   lo=float(run["goals"][0][0]-run["goal_width_m"]/2+.11)
   hi=float(run["goals"][0][0]+run["goal_width_m"]/2-.11)
   return {"center_x_m":x,"valid_x_min_m":lo,"valid_x_max_m":hi,
           "signed_nearest_lateral_margin_m":min(x-lo,hi-x),
           "full_sphere_goal":lo<=x<=hi}
 return None

def audit(run,prior,study,config_name,is_zero):
 cfg=json.loads((ROOT/config_name).read_text())
 base=json.loads((ROOT/prior).read_text())
 if cfg["foot_sweep_direction"]["enabled"]!= (not is_zero):
  raise ValueError("Wrong frozen zero/candidate configuration")
 assert run["joint_strike"]["config_sha256"]==sha(config_name)
 assert run["joint_strike"]["mode"]=="foot_sweep_direction"
 assert run["policy_sha256"]==sha(config_name)
 assert run["joint_strike"]["checkpoint_sha256"]==cfg["post_shot_recovery"]["checkpoint_sha256"]
 for file,h in run["joint_strike"]["source_sha256"].items():
  assert sha("tools/"+file)==h,("unexpected runtime code",file)
 assert run["joint_strike"]["config"]["foot_sweep_direction"]==cfg["foot_sweep_direction"]
 rows=validated_episode(run)
 orig=validated_episode(base)
 if len(orig)!=2800:raise ValueError("Frozen original baseline not 14s")
 truncated=len(rows)!=2800
 shot=run["records"][0]
 summary=summarize_strike(run)
 limits=limit_observations(rows,run)
 feedback=run["joint_strike"]["feedback"]
 excess=[]
 if len(feedback)!=700:excess.append("incomplete_14s_policy_decisions")
 peak=0.
 slew=0.
 prev=np.zeros(29)
 used=0
 eligible=0
 earliest=None
 for i,row in enumerate(feedback):
  m=row.get("foot_sweep_direction")
  if not isinstance(m,dict):raise ValueError("Missing full directional evidence")
  t=float(row["time_s"])
  a=np.asarray(m["actual_delta_rad"],float)
  was=np.asarray(m["previous_delta_rad"],float)
  base_action=np.asarray(m["baseline_action"],float)
  final=np.asarray(m["final_action"],float)
  if any(k.shape!=(29,) or not np.isfinite(k).all() for k in (a,was,base_action,final)):
   raise ValueError("Invalid per-joint samples")
  if i==0 and not np.array_equal(was,np.zeros(29)):
   excess.append("first_nonzero")
  if not np.array_equal(was,prev):
   excess.append("previous_delta_not_continuous")
  mdelta=float(np.max(abs(a-prev)))
  amp=float(np.max(abs(a)))
  peak=max(peak,amp)
  slew=max(slew,mdelta)
  if amp>1e-7:
   used+=1
   if earliest is None:earliest=t
  if m["eligible"]:eligible+=1
  if is_zero and amp>1e-7:excess.append("disabled_nonzero")
  if t<.85-1e-7 and amp>1e-7:excess.append("early_nonzero")
  if float(m["context_gate"])==0 and amp>1e-7:excess.append("nominal_nonzero")
  if amp>.022+4e-6:excess.append("amplitude_exceeded")
  if mdelta>.004+4e-6:excess.append("slew_exceeded")
  if m["ball_state_write"] is not False:excess.append("ball_written")
  prev=a
 prefix={}
 zero_stream={}
 for key in ("traces","contacts","kinematics"):
  if is_zero:zero_stream[key]=run[key]==base[key]
 if is_zero:
  for k in ("records","sequence_records","running_evidence","stride_shot_evidence","stride_shot_success"):
   zero_stream[k]=run.get(k)==base.get(k)
  for k in ("root_xyz","ball_xyz","joint_pos","joint_vel","joint_target"):
   a=np.asarray([x[k] for x in rows],float)
   b=np.asarray([x[k] for x in orig],float)
   prefix[k]=float(np.max(abs(a-b)))
  old=base["joint_strike"]["feedback"]
  old_feedback=[{k:v for k,v in x.items()} for x in feedback]
  for x in old_feedback:x.pop("foot_sweep_direction",None)
  zero_stream["original_control_feedback"]=old_feedback==old
  if not all(zero_stream.values()) or any(v!=0 for v in prefix.values()):
   excess.append("original_physics_or_actions_changed")
 speed=summary["contact"]["outgoing_speed_xy_mps"] if summary["contact"] else None
 old_speed=summarize_strike(base)["contact"]["outgoing_speed_xy_mps"]
 ratio=float(speed/old_speed) if speed is not None and old_speed else None
 guard=dict(
   original_full_goal=shot["goal"] is True,
   legal_instep=shot["surface_match"] is True
                and run["sequence_records"][0]["illegal_body_ball_contact"] is False,
   original_task=summary["outcome"]["success"] is True,
   real_qdot=limits["actual_speed_exceedance_sample_joint_count"]==0,
   physical_complete=(not truncated) and abs(rows[-1]["time_s"]-14)<1e-7 and run["termination_reasons"]==[[]],
   original_standing=shot["recovered"] is True and shot["standing_seconds"]>=1.,
   speed_retention=ratio is not None and ratio>=.95,
   no_external_writes=run["root_fixed"] is False and run["post_initialization_state_writes"] is False,
   controller_envelope=not excess,
   zero_equivalence=all(zero_stream.values()) if is_zero else None)
 # The pinned zero condition is an original GOAL MISS. Passing zero-parity
 # means exactly reproducing that miss, not magically making it a goal.
 status=all(v for k,v in guard.items()
            if v is not None and (not is_zero or k not in ("original_full_goal","original_task")))
 return {"status":"pass" if status else "fail",
  "is_zero":is_zero,
  "baseline_file":prior,"baseline_sha256":sha(prior),
  "output_sha256":None,
  "source_digest":sha(config_name),
  "actual_feedback_active_decisions":used,"eligible_decisions":eligible,
  "first_nonzero_s":earliest,"peak_feedback_rad":peak,"peak_slew_rad":slew,
  "violations":sorted(set(excess)),"zero_exact_stream":zero_stream,
  "zero_max_abs_state_diffs":prefix,
  "original_task":summary["outcome"],"guard":guard,
  "ball_speed_xy_mps":speed,"original_ball_speed_xy_mps":old_speed,"speed_ratio":ratio,
  "instep_fraction":shot["requested_surface_impulse_fraction"],
  "standing_seconds":shot["standing_seconds"],
  "physical_steps":len(rows),
  "termination_reasons":run["termination_reasons"],
  "last_physics_sample_s":rows[-1]["time_s"],
  "maximum_measured_joint_speed_ratio":limits["maximum_actual_speed_ratio"],
  "measured_velocity_excess_samples":limits["actual_speed_exceedance_sample_joint_count"],
  "full_ball_goal_margin":pose_goal_margin(run),
  "limitations":["Frozen deterministic development case, never unseen generalization",
                 "Contact-point Jacobian approximates physical response; actual real qdot/ball direction require observed PhysX",
                 "Measured clipped PD estimates are not hardware joint torques"]}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--case",required=True)
 p.add_argument("--input",required=True)
 p.add_argument("--config",required=True)
 p.add_argument("--zero",action="store_true")
 p.add_argument("--output",required=True)
 a=p.parse_args()
 if (ROOT/a.output).exists():raise FileExistsError(ROOT/a.output)
 study=json.loads((ROOT/"configs/foot_sweep_direction_study_v038.json").read_text())
 for file,expected in study["source_sha256"].items():
  if sha(file)!=expected:raise ValueError("Frozen protocol file drift: "+file)
 from strike_cases import load_cases
 cases=load_cases(ROOT)
 if a.case not in cases:raise ValueError("Unknown case")
 row=next(x for x in study["candidate_cases_order"] if x["id"]==a.case)
 assert a.input==(study["zero_trial"]["report"] if a.zero else row["candidate"])
 assert a.config==(study["zero_trial"]["config"] if a.zero else study["candidate_config"])
 run=json.loads((ROOT/a.input).read_text())
 result=audit(run,row["baseline"],study,a.config,a.zero)
 result.update(version="0.1.38",case=a.case,protocol_sha256=sha("configs/foot_sweep_direction_study_v038.json"),
               input=a.input,output_sha256=sha(a.input))
 (ROOT/a.output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
 print("V038_PHYSICAL_AUDIT",json.dumps({k:result[k] for k in
    ("case","status","is_zero","guard","violations","eligible_decisions","actual_feedback_active_decisions",
     "peak_feedback_rad","peak_slew_rad","ball_speed_xy_mps","speed_ratio",
     "maximum_measured_joint_speed_ratio","full_ball_goal_margin")},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
