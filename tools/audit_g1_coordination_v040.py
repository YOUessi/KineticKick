#!/usr/bin/env python3
"""v40 independent original-physics gate for generated 29DoF human-proxy ref.

No retuning after rollout. A zero-parity PASS is not proof of better motion.
Active PASS requires original positive full-task behavior, all actual qdot,
instep, safe stance and nonzero human-generated reference activity. A strong
physics result STILL cannot prove true human biomechanics or naturalness.
"""
from pathlib import Path
import argparse,hashlib,json,sys,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from strike_baseline_metrics import validated_episode,summarize_strike
from audit_p0_dynamic_strike import limit_observations
from free_shooting_metrics import ShotTracker

def digest(p):
 h=hashlib.sha256()
 with (ROOT/p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def collect_feedback(run,zero):
 feedback=run["joint_strike"]["feedback"]
 info={"steps":len(feedback)}
 used=0
 peak=0.
 slew=0.
 previous=np.zeros(29)
 violations=[]
 eligibility=0
 first_nonzero=None
 until=None
 if len(feedback)!=700:violations.append("not_full_700_control_steps")
 for i,row in enumerate(feedback):
  c=row.get("human_coordination_reference")
  if not isinstance(c,dict):raise ValueError(f"Missing original coordination diagnostic at {i}")
  if c["enabled"] is not (not zero):raise ValueError("Wrong controller disabled/enabled input")
  a=np.asarray(c["applied_delta_rad"],float)
  old=np.asarray(c["previous_delta_rad"],float)
  base=np.asarray(c["baseline_action"],float)
  final=np.asarray(c["final_action"],float)
  if any(z.shape!=(29,) or not np.isfinite(z).all() for z in (a,old,base,final)):
   raise ValueError("Untrusted saved 29 joint corrections")
  # Packed debug from original physical process is not an arbitrary proposed action;
  # final policy reference must match when no amplitude was physically applied.
  if np.any(a!=0) and zero:violations.append("disabled_applied_nonzero")
  if float(np.max(abs(a-old)))>.00121:violations.append("excess_joint_slew")
  if float(np.max(abs(a)))>.006002:violations.append("excess_joint_amplitude")
  if i>0 and np.max(abs(old-previous))>2e-6:violations.append("previous_joint_state_changed")
  if row["time_s"]<1.05-1e-5 and abs(a).max()>1e-7:violations.append("protected_early_prefix_changed")
  if row["time_s"]>=1.82 and abs(a).max()>1e-7:violations.append("post_contact_correction_never_released")
  if not bool(c["ball_state_write"]) is False:violations.append("external_ball_write_flag")
  if "eligible" in c and c["eligible"]:eligibility+=1
  if float(np.max(abs(a)))>1e-7:
   used+=1
   if first_nonzero is None:first_nonzero=float(row["time_s"])
   until=float(row["time_s"])
  peak=max(peak,float(np.max(abs(a))))
  slew=max(slew,float(np.max(abs(a-previous))))
  previous=a
 info.update(nonzero_decisions=used,eligible_decisions=eligibility,
    peak_rad=peak,maximum_slew_rad=slew,first_nonzero_s=first_nonzero,
    last_nonzero_s=until,violations=sorted(set(violations)))
 return info

def validate(case,zero,protocol):
 study=protocol
 reg=study["zero"] if zero else next(z for z in study["cases_in_conditional_order"] if z["id"]==case)
 cfg_path=reg["config"] if zero else study["candidate_config"]
 result_path=reg["report"] if zero else reg["candidate"]
 base_path=next(z["baseline"] for z in study["cases_in_conditional_order"] if z["id"]==case)
 cfg=json.loads((ROOT/cfg_path).read_text())
 src=json.loads((ROOT/base_path).read_text())
 run=json.loads((ROOT/result_path).read_text())
 if cfg["human_coordination_reference"]["enabled"] is not (not zero):
  raise ValueError("Physical gate run config not from fixed registered source")
 if cfg["human_coordination_reference"]["trajectory_sha256"]!=study["generated_q_trajectory_sha256"]:
  raise ValueError("Generated artifact replaced")
 if run["joint_strike"]["mode"]!="human_coordination_reference":
  raise ValueError("Wrong controller name")
 if run["joint_strike"]["config_sha256"]!=digest(cfg_path) or run["policy_sha256"]!=digest(cfg_path):
  raise ValueError("Physical controller source/config was modified after protocol freeze")
 if run["joint_strike"]["checkpoint_sha256"]!=study["original_postshot_sha256"]:
  raise ValueError("Old recovery expert weight changed")
 for source,expected in run["joint_strike"]["source_sha256"].items():
  if digest("tools/"+source)!=expected:
   raise ValueError("Runtime source SHA mismatch: "+source)
 if run["root_fixed"] or run["post_initialization_state_writes"]:
  raise ValueError("Invalid physical state write/free-root")
 newrows=validated_episode(run)
 oldrows=validated_episode(src)
 feedback=collect_feedback(run,zero)
 physical_errors=[]
 if not newrows:raise ValueError("No physical episode")
 if len(newrows)!=2800:physical_errors.append("incomplete_14s")
 if abs(newrows[-1]["time_s"]-14)>1e-6:physical_errors.append("wrong_final_time")
 if run["termination_reasons"]!=[[]]:physical_errors.append("physical_termination")
 exact={}
 state_errors={}
 if zero:
  for stream in ("traces","contacts","kinematics","records","sequence_records",
                 "running_evidence","stride_shot_evidence","stride_shot_success"):
   exact[stream]=(run.get(stream)==src.get(stream))
  logged=[]
  for row in run["joint_strike"]["feedback"]:
   c=dict(row);c.pop("human_coordination_reference",None)
   logged.append(c)
  exact["original_joint_feedback"]=logged==src["joint_strike"]["feedback"]
  for term in ("root_xyz","ball_xyz","joint_pos","joint_vel","joint_target"):
   if len(newrows)!=len(oldrows):state_errors[term]=None
   else:state_errors[term]=float(np.max(abs(
       np.asarray([x[term] for x in newrows])-np.asarray([x[term] for x in oldrows]))))
  if not all(exact.values()) or any(x!=0 for x in state_errors.values()):
   physical_errors.append("zero_physics_not_equal")
 original=summarize_strike(src)
 candidate=summarize_strike(run)
 lim=limit_observations(newrows,run)
 speed=candidate["contact"]["outgoing_speed_xy_mps"] if candidate["contact"] is not None else None
 oldspeed=original["contact"]["outgoing_speed_xy_mps"] if original["contact"] is not None else None
 ratio=float(speed/oldspeed) if speed is not None and oldspeed else None
 shot=run["records"][0]
 replay=ShotTracker(initial_root=run["initial_robot_state"]["root_position"][0],
  initial_ball=[*run["points"][0],.113],goal=run["goals"][0],
  forward=run["trials"][0]["goal_forward"],style=run["trials"][0]["style"],
  width=run["goal_width_m"],height=run["goal_height_m"])
 for row in run["traces"][0]:replay.update(row)
 replay_ok=replay.summary()==shot
 bounds={
  "whole_ball_original_goal":shot["goal"] is True,
  "original_task_success":candidate["outcome"]["success"] is True,
  "correct_instep":shot["measured_surface"]=="instep"
       and float(shot["requested_surface_impulse_fraction"])>=.95,
  "no_illegal_ball_contact":run["sequence_records"][0]["illegal_body_ball_contact"] is False,
  "physical_full_14s":len(physical_errors)==0,
  "actual_joint_speed_limit":lim["actual_speed_exceedance_sample_joint_count"]==0,
  "recovered_original":shot["recovered"] is True and shot["standing_seconds"]>=1.,
  "outgoing_speed_retention":ratio is not None and ratio>=study["minimum_outgoing_ball_xy_speed_ratio"],
  "original_physical_goal_replay":replay_ok,
  "base_running":run["running_evidence"]["running_success"] is True,
  "original_strike_transition":run["stride_shot_evidence"]["continuous_running_contact"] is True,
  "generated_ref_observable":feedback["nonzero_decisions"]>0,
  "joint_reference_bounds":not feedback["violations"],
  "exact_zero_source_streams":all(exact.values()) if zero else None
 }
 if zero:
  effective={k:v for k,v in bounds.items()
    if k not in ("generated_ref_observable",)}
 else: effective=bounds
 # Zero source center was already a valid goal; if later testing a known
 # miss on zero, original_task is not required. This study registers center.
 passed=all(x for x in effective.values() if x is not None)
 return dict(
  version="0.1.40",registered_case=case,is_zero=zero,
  tested_input_sha256=digest(result_path),config_sha256=digest(cfg_path),
  original_input_sha256=digest(base_path),
  original_source_original_task=original["outcome"],
  registered_trial_complete=True,passed=passed,
  actual_control_diagnostic=feedback,
  physical_errors=physical_errors,
  full_14s_physics_steps=len(newrows),
  termination_reasons=run["termination_reasons"],
  original_strict_success=candidate["outcome"],
  original_model_checkpoint=study["original_postshot_sha256"],
  actual_joint_velocity_ratio_max=lim["maximum_actual_speed_ratio"],
  real_joint_speed_excess_samples=lim["actual_speed_exceedance_sample_joint_count"],
  physical_contacts_first_s=shot["contact_time_s"],
  right_instep_fraction=shot["requested_surface_impulse_fraction"],
  outgoing_speed_mps=speed,
  original_outgoing_speed_mps=oldspeed,
  original_speed_retention=ratio,
  final_standing_s=shot["standing_seconds"],
  original_goal_replay_exact=replay_ok,
  physical_zero_parity_streams=exact,
  zero_max_absolute_state_diffs=state_errors,
  gates=bounds,
  failures=[k for k,v in effective.items() if v is False],
  claim="Human proxy can only be a kinematic study; passing original PhysX shoot gate is NOT professional biomechanical evidence")

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--case",required=True)
 p.add_argument("--zero",action="store_true")
 p.add_argument("--output",required=True)
 a=p.parse_args()
 dest=ROOT/a.output
 if dest.exists():raise FileExistsError(dest)
 protocol=json.loads((ROOT/"configs/g1_coordination_physics_v040.json").read_text())
 if protocol["no_hyperparameter_search"] is not True:raise ValueError("Frozen gate required")
 for path,expected in protocol["sources"].items():
  if digest(path)!=expected:raise ValueError("Study SHA changed: "+path)
 if not a.zero and not any(c["id"]==a.case for c in protocol["cases_in_conditional_order"]):
  raise ValueError("Not registered")
 if a.zero and a.case!=protocol["zero"]["case"]:
  raise ValueError("Zero only registered on center")
 r=validate(a.case,a.zero,protocol)
 r["protocol_sha256"]=digest("configs/g1_coordination_physics_v040.json")
 dest.parent.mkdir(parents=True,exist_ok=True)
 dest.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n")
 print("V040_PHYSX_AUDIT",json.dumps({
  "case":a.case,"zero":a.zero,"passed":r["passed"],
  "physics_steps":r["full_14s_physics_steps"],
  "goal":r["gates"]["whole_ball_original_goal"],
  "qdot_excess":r["real_joint_speed_excess_samples"],
  "original_contact":r["physical_contacts_first_s"],
  "applied_decisions":r["actual_control_diagnostic"]["nonzero_decisions"],
  "peak_ref_rad":r["actual_control_diagnostic"]["peak_rad"],
  "outgoing_speed_mps":r["outgoing_speed_mps"],
  "standing_s":r["final_standing_s"],
  "failed_gates":r["failures"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
