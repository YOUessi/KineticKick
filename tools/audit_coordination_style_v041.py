#!/usr/bin/env python3
"""v41 strict complete original G1 PhysX acceptance for separated STYLE gate.

The science endpoint is actual nonzero 29DoF activation on nominal ball AND
full original robot-ball physical functionality, not a re-labelled v32 goal.
This audit deliberately replays the original ShotTracker and checks true
qdot, impulse/surface, original expert output and full recovery.
"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from strike_baseline_metrics import validated_episode,summarize_strike
from audit_p0_dynamic_strike import limit_observations
from free_shooting_metrics import ShotTracker

def sha(f):
 h=hashlib.sha256()
 with (ROOT/f).open("rb") as z:
  for b in iter(lambda:z.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def audit(case,zero,study):
 registered=study["zero_trial"] if zero else next(x for x in study["conditional_case_order"] if x["id"]==case)
 cfg=registered["config"] if zero else study["candidate_config"]
 src=registered["report"] if zero else registered["candidate"]
 prior=next(x["baseline"] for x in study["conditional_case_order"] if x["id"]==case)
 config=json.loads((ROOT/cfg).read_text())
 run=json.loads((ROOT/src).read_text())
 baseline=json.loads((ROOT/prior).read_text())
 if run["root_fixed"] or run["post_initialization_state_writes"]:
  raise ValueError("True free-root G1 and no physical state rewrite are mandatory")
 if run["joint_strike"]["mode"]!="coordination_style_v041":
  raise ValueError("Incorrect pipeline selected for v41 study")
 if sha(cfg)!=run["joint_strike"]["config_sha256"] or sha(cfg)!=run["policy_sha256"]:
  raise ValueError("Original physical source did not load declared configs")
 if run["joint_strike"]["checkpoint_sha256"]!=study["source_sha256"]["experiments/post_shot_v032/ppo_s3201/last_model.pt"]:
  raise ValueError("Original physical postshot model changed")
 if run["joint_strike"]["config"]["coordination_style_v041"]["enabled"] is not (not zero):
  raise ValueError("Wrong style config")
 if config["human_coordination_reference"]["trajectory_sha256"]!=study["source_sha256"]["experiments/coordination_projection_v040/train_clip00001_center/g1_joint_delta_50hz.npz"]:
  raise ValueError("Human-generated q reference changed")
 for name,digest in run["joint_strike"]["source_sha256"].items():
  if sha("tools/"+name)!=digest:raise ValueError("Untracked runtime source drift: "+name)
 p=validated_episode(run)
 q=validated_episode(baseline)
 if len(p)==0:raise ValueError("No valid PhysX samples")
 m=run["joint_strike"]["feedback"]
 physics_error=[]
 if len(p)!=2800 or abs(p[-1]["time_s"]-14.)>1e-6 or run["termination_reasons"]!=[[]]:
  physics_error.append("not_full_real_14s_episode")
 if len(m)!=700:physics_error.append("not_full_700_control_decisions")
 prior_out=summarize_strike(baseline)
 actual=summarize_strike(run)
 actual_lim=limit_observations(p,run)
 shot=run["records"][0]
 speed=actual["contact"]["outgoing_speed_xy_mps"] if actual["contact"] else None
 prior_speed=prior_out["contact"]["outgoing_speed_xy_mps"] if prior_out["contact"] else None
 ratio=None if not speed or not prior_speed else speed/prior_speed
 applied=0
 style_gate_eligible=0
 actual_gain_and_zero_error=0
 peak=0.
 max_slew=0.
 prior_delta=np.zeros(29)
 violation=[]
 for index,row in enumerate(m):
  diag=row.get("coordination_style_v041")
  command=row.get("human_coordination_reference")
  if not isinstance(diag,dict) or not isinstance(command,dict):
   raise ValueError("Missing signed style/physical reference diagnostic")
  if diag["enabled"] is not (not zero) or command["enabled"] is not (not zero):
   raise ValueError("Wrong accepted controller enabled")
  if diag["ball_state_write"] is not False or command["ball_state_write"] is not False:
   violation.append("state_write_evidence")
  if not zero and diag["independent_of_nominal_error_gate"] is not True:
   violation.append("legacy_error_coupling")
  if zero and diag["parent_action_identical"] is not True:
   violation.append("zero_parent_action_drift")
  a=np.asarray(command["applied_delta_rad"],float)
  before=np.asarray(command["previous_delta_rad"],float)
  if a.shape!=(29,) or before.shape!=(29,) or not np.isfinite(a).all():
   raise ValueError("Missing or nonfinite real G1 29 channel output")
  if abs(a-before).max()>.00121 or abs(a-prior_delta).max()>.00121:
   violation.append("reference_joint_step")
  if abs(a).max()>.006002:violation.append("reference_joint_amplitude")
  if row["time_s"]<1.05-1e-5 and abs(a).max()>1e-7:violation.append("unprotected_early_prefix")
  if row["time_s"]>=1.82 and abs(a).max()>1e-7:violation.append("nonzero_after_return")
  if zero and abs(a).max()>1e-7:violation.append("disabled_nonzero")
  if abs(a).max()>1e-7:
   applied+=1
   if not zero and abs(float(command["original_context_gate"]))<1e-8:
    actual_gain_and_zero_error+=1
  if diag.get("eligible",False):style_gate_eligible+=1
  peak=max(peak,float(abs(a).max()))
  max_slew=max(max_slew,float(abs(a-prior_delta).max()))
  prior_delta=a
 direct={}
 original_feedback=[]
 if zero:
  for k in ("traces","kinematics","contacts","records","sequence_records",
            "running_evidence","stride_shot_evidence","stride_shot_success"):
   direct[k]=run.get(k)==baseline.get(k)
  for row in m:
   c=dict(row)
   c.pop("human_coordination_reference",None)
   c.pop("coordination_style_v041",None)
   original_feedback.append(c)
  direct["original_feedback"]=original_feedback==baseline["joint_strike"]["feedback"]
  if not all(direct.values()):physics_error.append("zero_physical_fields_changed")
 scorer=ShotTracker(initial_root=run["initial_robot_state"]["root_position"][0],
  initial_ball=[*run["points"][0],.113],goal=run["goals"][0],
  forward=run["trials"][0]["goal_forward"],style=run["trials"][0]["style"],
  width=run["goal_width_m"],height=run["goal_height_m"])
 for row in run["traces"][0]:scorer.update(row)
 matches=scorer.summary()==shot
 criteria={
  "original_full_sphere_goal":shot["goal"] is True,
  "original_whole_task":actual["outcome"]["success"] is True,
  "correct_instep":shot["measured_surface"]=="instep"
      and shot["requested_surface_impulse_fraction"]>=.95
      and run["sequence_records"][0]["illegal_body_ball_contact"] is False,
  "full_14_second_physics":not physics_error,
  "actual_joint_speed_within_original_limits":actual_lim["actual_speed_exceedance_sample_joint_count"]==0,
  "recovery_standing":shot["recovered"] is True and shot["standing_seconds"]>=1.,
  "ball_speed_retained":ratio is not None and ratio>=study["minimum_ball_outgoing_speed_fraction"],
  "original_running":run["running_evidence"]["running_success"] is True,
  "continuous_runup_contact":run["stride_shot_evidence"]["continuous_running_contact"] is True,
  "original_ball_goal_replay":matches,
  "bounded_fullbody_reference":len(violation)==0,
  "real_style_decisions_nonzero":applied>0,
  # The zero-error activation property is ONLY meaningful for the
  # preregistered nominal-center case. A ball-offset run may have strictly
  # positive original correction gain on every valid style activation.
  # Do NOT call an inapplicable diagnosis a physics failure or silently
  # replace it by success: record null for non-nominal cases.
  "new_style_active_despite_zero_original_error":
      (actual_gain_and_zero_error>0) if case=="center" else None,
  "zero_stream_parity":all(direct.values()) if zero else None,
 }
 if zero:
  # Zero evidence can pass even though new-style activity must equal zero.
  gate={k:v for k,v in criteria.items() if k not in (
     "real_style_decisions_nonzero","new_style_active_despite_zero_original_error")}
 else:gate=criteria
 passed=all(v for v in gate.values() if v is not None)
 result={
  "version":"0.1.41", "case":case,"is_zero":zero,
  "protocol_sha256":sha("configs/g1_coordination_style_physics_v041.json"),
  "original_report_sha256":sha(prior),"physical_output_sha256":sha(src),
  "config_sha256":sha(cfg), "passed":passed,
  "criteria":criteria,"failed_gates":[k for k,v in gate.items() if v is False],
  "real_physics_samples":len(p),"real_policy_decisions":len(m),
  "real_14s_completed":len(p)==2800 and not physics_error,
  "real_physics_errors":physics_error,
  "actual_joint_speed_ratio_max":actual_lim["maximum_actual_speed_ratio"],
  "real_speed_exceedance_samples":actual_lim["actual_speed_exceedance_sample_joint_count"],
  "original_contact_time_s":shot["contact_time_s"],
  "original_right_instep_impulse_fraction":shot["requested_surface_impulse_fraction"],
  "outgoing_ball_xy_mps":speed,
  "prior_ball_xy_mps":prior_speed,
  "outgoing_speed_fraction_of_prior":ratio,
  "original_goal":shot["goal"],"original_standing_s":shot["standing_seconds"],
  "additional_reference_active_decisions":applied,
  "style_gate_eligible_steps":style_gate_eligible,
  "active_style_despite_zero_original_error_decisions":actual_gain_and_zero_error,
  "maximum_new_joint_reference_rad":peak,
  "maximum_new_joint_reference_step_rad":max_slew,
  "command_provenance_violations":sorted(set(violation)),
  "zero_equivalence":direct,
  "interpretation":"This is a video-estimated-human coordination proxy/reference integration trial, not evidence of natural/professional explosive instep technique."}
 return result

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--case",required=True)
 p.add_argument("--zero",action="store_true")
 p.add_argument("--output",required=True)
 args=p.parse_args()
 output=ROOT/args.output
 if output.exists():raise FileExistsError(output)
 study=json.loads((ROOT/"configs/g1_coordination_style_physics_v041.json").read_text())
 for path,digest in study["source_sha256"].items():
  if sha(path)!=digest:raise ValueError("Prespecified frozen source changed: "+path)
 if args.zero and args.case!=study["zero_trial"]["case"]:
  raise ValueError("Zero only preregistered for center")
 if not args.zero and args.case not in [x["id"] for x in study["conditional_case_order"]]:
  raise ValueError("Unregistered candidate case")
 result=audit(args.case,args.zero,study)
 output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
 print("V041_G1_PHYSICAL_ACCEPTANCE",json.dumps({k:result[k] for k in (
   "case","is_zero","passed","real_physics_samples","original_goal",
   "additional_reference_active_decisions","active_style_despite_zero_original_error_decisions",
   "outgoing_ball_xy_mps","real_speed_exceedance_samples","original_standing_s","failed_gates")},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
