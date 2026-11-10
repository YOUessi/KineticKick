#!/usr/bin/env python3
"""v48 independent original G1 14s REAL PhysX pre-impact coordination PPO audit.

Original control still free-root, original PD+actual G1 joint speed limits, ball
mass/foot collision, 14s whole-ball goal, correct instep/force, stable recovery.
Use 5ms observed qdot, not bounded action targets, to reject unsafe work.
Research model has no professional explosive human motion/contact truth.
"""
from pathlib import Path
import hashlib,json,argparse,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from strike_baseline_metrics import validated_episode,task_outcome,first_force_episode
from audit_p0_dynamic_strike import limit_observations
STUDY="configs/learned_preimpact_wholebody_study_v048.json"
def sha(p):
 h=hashlib.sha256()
 with (ROOT/p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def original_zero_study(study):
 file=study["zero_result"]
 if not (ROOT/file).exists():raise ValueError("First physically run disabled zero controller")
 z=json.loads((ROOT/file).read_text())
 orig=json.loads((ROOT/"reports/v032_learned_center.json").read_text())
 if z["joint_strike"]["mode"]!="preimpact_wholebody_v048":
  raise ValueError("Expected v48 disabled original comparison")
 if z["joint_strike"]["config_sha256"]!=sha(study["zero_reference_disabled_config"]):
  raise ValueError("Wrong zero controller config")
 if z["joint_strike"]["config"]["preimpact_wholebody_v048"]["enabled"] is not False:
  raise ValueError("Expected truly disabled pre-contact learned actor")
 if z["post_initialization_state_writes"] or orig["post_initialization_state_writes"] or z["root_fixed"]:
  raise ValueError("Zero free-root/physical state integrity violated")
 a=validated_episode(z)
 b=validated_episode(orig)
 if len(a)!=2800 or len(b)!=2800:
  raise ValueError("Whole original PhysX zero episode missing")
 signal=set(a[0])&set(b[0])
 other_a=set(a[0])-set(b[0])
 other_b=set(b[0])-set(a[0])
 # As with v46 original instrumented source, older v32 can include 7
 # additional READ-ONLY telemetry columns. Never claim byte-exact JSON.
 if len(signal)<25:
  raise ValueError("Too few original simulator state signals for parity")
 mismatches=[]
 for k,(x,y) in enumerate(zip(a,b)):
  for name in signal:
   if x[name]!=y[name]:
    mismatches.append(dict(step=k,field=name))
    break
 other={}
 for key in ("contacts","kinematics","records","sequence_records",
   "running_evidence","stride_shot_evidence","stride_shot_success"):
  other[key]=(z.get(key)==orig.get(key))
 old_fb=orig["joint_strike"]["feedback"]
 extra=z["joint_strike"]["feedback"]
 actor_equal=True
 for idx,(x,y) in enumerate(zip(extra,old_fb)):
  diag=x.get("preimpact_wholebody_v048")
  if not isinstance(diag,dict) or diag["enabled"] is not False:
   raise ValueError("Disabled PPO still generated diagnostic activity")
  if max(abs(v) for v in diag["applied_delta_rad"])!=0.:
   raise ValueError("Disabled PPO did change G1 reference")
  cp=dict(x)
  cp.pop("preimpact_wholebody_v048",None)
  if cp!=y:
   actor_equal=False
 if len(extra)!=len(old_fb):actor_equal=False
 result=dict(
  zero_controller_source_sha256=sha(file),
  original_v32_sha256=sha("reports/v032_learned_center.json"),
  zero_parity_pass=not mismatches and actor_equal and all(other.values()),
  physical_5ms_steps=2800,original_matched_fields=len(signal),
  unmatched_readonly_current=sorted(other_a),unmatched_readonly_prior=sorted(other_b),
  fully_byte_identical=False,
  stepwise_original_physics_differences=mismatches[:10],
  original_joint_command_debug_parity=actor_equal,
  original_scoring_contact_kinematics_parity=other)
 return result

def analyze(case,source,kind,study):
 r=json.loads((ROOT/source).read_text())
 cfg=study["original_v32_baseline_config"] if kind=="original_v32" else study["candidate_evaluation_config"]
 if r["policy_sha256"]!=sha(cfg) or r["joint_strike"]["config_sha256"]!=sha(cfg):
  raise ValueError("Source G1 run has wrong freeze config")
 model_cfg=json.loads((ROOT/cfg).read_text())
 expected_mode="post_shot_residual" if kind=="original_v32" else "preimpact_wholebody_v048"
 if r["joint_strike"]["mode"]!=expected_mode or r["root_fixed"] or r["post_initialization_state_writes"]:
  raise ValueError("Original free-root G1 PHYXS cannot be replaced by fixed-root or alternative controller")
 if r["joint_strike"]["checkpoint_sha256"]!=study["original_v32_PPO_recovery_sha256"]:
  raise ValueError("Original postshot trained PPO changed")
 if kind!="original_v32":
  if model_cfg["preimpact_wholebody_v048"]["checkpoint_sha256"]!=study["post_training_final_only_frozen"]["final_checkpoint_sha256"]:
   raise ValueError("Wrong G1 learned pre-impact actor")
 for name,h in r["joint_strike"]["source_sha256"].items():
  if sha("tools/"+name)!=h:raise ValueError("Original PhysX execution source changed "+name)
 if not np.allclose(r["joint_strike"]["ball_offset_xy"],case["ball_offset_xy_m"],atol=1e-8,rtol=0):
  raise ValueError("Wrong predeclared soccer-ball initial position")
 if abs(r["joint_strike"]["velocity_y_delta"]-case["velocity_y_delta_mps"])>1e-8:
  raise ValueError("Wrong root velocity")
 original_states=r["initial_robot_state"]
 data=validated_episode(r)
 if not data:raise ValueError("No physically observed shooting")
 limits=limit_observations(data,r)
 task=task_outcome(r)
 shot=r["records"][0]
 exit=first_force_episode(data)
 ball_mps=None if exit is None else exit["outgoing_speed_xy_mps"]
 legal=shot["measured_surface"]=="instep" and shot["surface_match"] is True and (
  shot["requested_surface_impulse_fraction"]>=.95) and (
  r["sequence_records"][0]["illegal_body_ball_contact"] is False)
 full=len(data)==2800 and abs(data[-1]["time_s"]-14)<1e-6 and r["termination_reasons"]==[[]]
 gates=dict(
  full_14s_original_freeroot_PhysX=full,
  actual_29joint_speed_within_original_limits=limits["actual_speed_exceedance_sample_joint_count"]==0,
  lawful_physical_right_instep_impact=bool(legal),
  original_ball_fully_inside_goal=shot["goal"] is True,
  actual_recovered_original_G1_standing=shot["recovered"] is True and shot["standing_seconds"]>=1.,
  original_runup=bool(r["running_evidence"]["running_success"]),
  original_continuous_runup_to_impact=bool(r["stride_shot_evidence"]["continuous_running_contact"]),
  original_task_success=task["success"] is True,
  real_ball_speed_xy_at_least_3_5_mps=ball_mps is not None and ball_mps>=3.5)
 applied=0
 peak=0.
 max_slew=0.
 any_eligible=0
 first_action=None
 bad=[]
 previous=np.zeros(29)
 if kind=="learned_preimpact_v48":
  for row in r["joint_strike"]["feedback"]:
   diag=row.get("preimpact_wholebody_v048")
   if not isinstance(diag,dict) or not diag.get("enabled"):
    raise ValueError("Missing original PPO action diagnostic")
   delta=np.asarray(diag["applied_delta_rad"],float)
   old=np.asarray(diag["previous_delta_rad"],float)
   if delta.shape!=(29,) or old.shape!=(29,) or not np.isfinite(delta).all():
    raise ValueError("True G1 29DoF action invalid")
   if float(abs(delta).max())>.018002:bad.append("reference_amplitude")
   if float(abs(delta-old).max())>.00301 or float(abs(delta-previous).max())>.00301:
    bad.append("20ms_reference_slew")
   if row["time_s"]<1.02 and float(abs(delta).max())>1e-7:
    bad.append("learned_action_before_registered_window")
   if row["time_s"]>=1.66 and float(abs(delta).max())>1e-7:
    bad.append("learned_action_unreleased_late")
   if diag["ball_state_write"] is not False:
    bad.append("external_soccer_impulse_write")
   if float(abs(delta).max())>1e-7:
    applied+=1
    if first_action is None:first_action=row["time_s"]
   peak=max(peak,float(abs(delta).max()))
   max_slew=max(max_slew,float(abs(delta-old).max()))
   any_eligible+=int(bool(diag["eligible"]))
   previous=delta
  gates["bounded_true_preimpact_joint_reference"]=len(bad)==0
 return dict(
   original_source=source,original_source_sha256=sha(source),control=kind,
   original_model_unchanged=True,
   original_14s_physical_seconds=data[-1]["time_s"],
   actual_physics_steps=len(data),qdot_speed_overlimit_sample_joint_count=limits["actual_speed_exceedance_sample_joint_count"],
   max_original_joint_velocity_limit_ratio=limits["maximum_actual_speed_ratio"],
   max_original_joint_velocity_sample=limits["worst_actual_speed"],
   strict_gates=gates,strict_physical_pass=all(gates.values()),
   first_true_right_foot_ball_contact_s=shot["contact_time_s"],
   measured_right_instep_impulse_fraction=shot["requested_surface_impulse_fraction"],
   actual_original_ball_goal=bool(shot["goal"]),
   true_outgoing_ball_speed_xy_mps=ball_mps,
   true_standing_duration_s=shot["standing_seconds"],
   first_learned_action_s=first_action,
   original_PPO_training_data_not_professional_human_explosive=True,
   new_preimpact_policy_nonzero_control_steps=applied,
   preimpact_eligible_policy_steps=any_eligible,
   preimpact_reference_peak_rad=peak,
   maximum_preimpact_reference_slew_20ms_rad=max_slew,
   preimpact_reference_contract_violations=sorted(set(bad)),
   failure_reasons=task["failure_reasons"],
   initial_robot_state=original_states,
   initial_ball_xy=r["points"][0]
 )

def full_audit(study):
 zero=original_zero_study(study)
 if not zero["zero_parity_pass"]:
  raise ValueError("Original physics zero path not exact")
 samples=[]
 for c in study["planned_6_new_exact_soccer_conditions"]:
  orig=analyze(c,c["baseline_output"],"original_v32",study)
  new=analyze(c,c["candidate_output"],"learned_preimpact_v48",study)
  if orig["initial_robot_state"]!=new["initial_robot_state"] or orig["initial_ball_xy"]!=new["initial_ball_xy"]:
   raise ValueError("Paired v48 initial G1 state or ball position differ")
  for rec in (orig,new):rec.pop("initial_robot_state");rec.pop("initial_ball_xy")
  samples.append(dict(id=c["id"],ball_offset_xy_m=c["ball_offset_xy_m"],velocity_y_delta_mps=c["velocity_y_delta_mps"],
       original_v32=orig,learned_preimpact_v48=new,
       real_ball_speed_gain_mps=None if orig["true_outgoing_ball_speed_xy_mps"] is None or new["true_outgoing_ball_speed_xy_mps"] is None
       else new["true_outgoing_ball_speed_xy_mps"]-orig["true_outgoing_ball_speed_xy_mps"]))
 old=sum(r["original_v32"]["strict_physical_pass"] for r in samples)
 new=sum(r["learned_preimpact_v48"]["strict_physical_pass"] for r in samples)
 engaged=sum(r["learned_preimpact_v48"]["new_preimpact_policy_nonzero_control_steps"]>0 for r in samples)
 return dict(version="0.1.48",original_study_sha256=sha(STUDY),
    zero_physics_equals_original=zero["zero_parity_pass"],
    zero_result=zero,
    number_of_new_exact_ball_conditions=6,
    number_of_paired_physical_episodes=12,
    original_v32_strict_pass=old,
    learned_preimpact_v48_strict_pass=new,
    repaired_old_failures=sum(not r["original_v32"]["strict_physical_pass"] and r["learned_preimpact_v48"]["strict_physical_pass"] for r in samples),
    regressed_old_successes=sum(r["original_v32"]["strict_physical_pass"] and not r["learned_preimpact_v48"]["strict_physical_pass"] for r in samples),
    learned_actor_actually_engaged_conditions=engaged,
    source_training_human_3dsp_broadcast_proxy_only=True,
    original_foot_ball_contact_PhySX_model_unchanged=True,
    qdot_motor_original_limits_unchanged=True,
    new_trained_G1_actor_promotion_approved=False,
    professional_biomechanics_established=False,
    six_trials_only_not_population_generalization=True,
    naturalness_blind_human_rating_established=False,
    no_14s_fresh_test_hyperparameter_tuning=True,
    cases=samples)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--zero-only",action="store_true")
 p.add_argument("--output",required=True)
 a=p.parse_args()
 dest=ROOT/a.output
 if dest.exists():raise FileExistsError("Do not overwrite")
 study=json.loads((ROOT/STUDY).read_text())
 for path,h in study["source_sha256"].items():
  if sha(path)!=h:raise ValueError("Frozen source SHA drift "+path)
 if a.zero_only:
  output=dict(version="0.1.48",zero_parity_pass=False,zero_details=original_zero_study(study))
  output["zero_parity_pass"]=output["zero_details"]["zero_parity_pass"]
 else:
  frozen=study.get("post_training_final_only_frozen")
  if not frozen or sha(study["candidate_evaluation_config"])!=frozen["eval_config_sha256"]:
   raise ValueError("Original learned actor not frozen before test")
  output=full_audit(study)
 dest.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
 print("V048_REAL_PREIMPACT_PHYSX_AUDIT",json.dumps({
  "zero":output["zero_parity_pass"] if a.zero_only else output["zero_physics_equals_original"],
  "n":0 if a.zero_only else output["number_of_new_exact_ball_conditions"],
  "original":None if a.zero_only else output["original_v32_strict_pass"],
  "learned":None if a.zero_only else output["learned_preimpact_v48_strict_pass"],
  "engaged":None if a.zero_only else output["learned_actor_actually_engaged_conditions"],
  "repair":None if a.zero_only else output["repaired_old_failures"],
  "regressed":None if a.zero_only else output["regressed_old_successes"],
  "details":[] if a.zero_only else [
    {"case":r["id"],"old":r["original_v32"]["strict_physical_pass"],
     "learned":r["learned_preimpact_v48"]["strict_physical_pass"],
     "actual_actuator_steps":r["learned_preimpact_v48"]["new_preimpact_policy_nonzero_control_steps"],
     "max_joint_speed_ratio":r["learned_preimpact_v48"]["max_original_joint_velocity_limit_ratio"],
     "new_ball_mps":r["learned_preimpact_v48"]["true_outgoing_ball_speed_xy_mps"]}
    for r in output["cases"]]
  },ensure_ascii=False),flush=True)

if __name__=="__main__":main()
