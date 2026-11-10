#!/usr/bin/env python3
"""v47 strict original 14s G1/PhysX paired recovery policy audit.

Use true 5ms joint velocities / original fixed G1 joint limits and original
right instep contact, full soccer-ball goal, 14s, continuous run-up and standing.
Training proxy rewards cannot imply a true safer or more natural shot.
All ball parameter triples were frozen before the one warm-start PPO train.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from strike_baseline_metrics import validated_episode,task_outcome,first_force_episode
from audit_p0_dynamic_strike import limit_observations
STUDY="configs/learned_contact_recovery_study_v047.json"
def digest(file):
 h=hashlib.sha256()
 with (ROOT/file).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def inspect(case,ctrl,study,config):
    report=case["old_control_output"] if ctrl=="original_v32" else case["trained_control_output"]
    d=json.loads((ROOT/report).read_text())
    if d["root_fixed"] or d["post_initialization_state_writes"]:
        raise ValueError("Original G1 free-base real simulator state required")
    if d["joint_strike"]["config_sha256"]!=digest(config) or d["policy_sha256"]!=digest(config):
        raise ValueError("Wrong frozen G1 recovery checkpoint/config")
    if d["joint_strike"]["mode"]!="post_shot_residual":
        raise ValueError("Expected unchanged original G1 recovery PPO task")
    if not np.allclose(d["joint_strike"]["ball_offset_xy"],case["ball_offset_xy_m"],atol=1e-8,rtol=0):
        raise ValueError("Wrong exact original ball position")
    if abs(d["joint_strike"]["velocity_y_delta"]-case["velocity_y_delta_mps"])>1e-8:
        raise ValueError("Wrong registered G1 original initial velocity")
    for src,sha in d["joint_strike"]["source_sha256"].items():
        if digest("tools/"+src)!=sha:
            raise ValueError("Runtime original Isaac sources changed "+src)
    expected=study["frozen_recovery_actor_start_SHA256"] if ctrl=="original_v32" else study["actual_final_trained_checkpoint_sha256"]
    if d["joint_strike"]["checkpoint_sha256"]!=expected:
        raise ValueError("Incorrect original/tuned PPO checkpoint")
    rows=validated_episode(d)
    speed=limit_observations(rows,d)
    score=task_outcome(d)
    shot=d["records"][0]
    first=first_force_episode(rows)
    exit_speed=None if first is None else first["outgoing_speed_xy_mps"]
    contact_ok=shot["measured_surface"]=="instep" and shot["surface_match"] is True and \
        shot["requested_surface_impulse_fraction"]>=.95 and not d["sequence_records"][0]["illegal_body_ball_contact"]
    complete=len(rows)==2800 and abs(rows[-1]["time_s"]-14)<1e-6 and d["termination_reasons"]==[[]]
    reward_gates=dict(
       original_14s_complete=complete,
       original_full_ball_goal=shot["goal"] is True,
       right_instep_contact=bool(contact_ok),
       measured_qdot_within_original_limits=speed["actual_speed_exceedance_sample_joint_count"]==0,
       original_task_success=score["success"] is True,
       full_runup_shot_transition=d["stride_shot_evidence"]["continuous_running_contact"] is True,
       original_running_success=d["running_evidence"]["running_success"] is True,
       recovery_actual_standing=shot["recovered"] is True and shot["standing_seconds"]>=1.,
       physical_horizontal_ball_speed_at_least_3_5_mps=exit_speed is not None and exit_speed>=3.5)
    feedback=d["joint_strike"]["feedback"]
    real_recovery_actuation=0
    first_active=None
    maxrec=0.
    for item in feedback:
       if "post_shot_recovery" not in item:
          raise ValueError("Actual recovery PPO command debug missing")
       z=item["post_shot_recovery"]
       delta=np.asarray(z["actual_delta_rad"],float)
       if delta.shape!=(29,) or not np.isfinite(delta).all() or z["ball_state_write"] is not False:
          raise ValueError("Invalid actual 29DoF original actuator reference")
       p=float(np.abs(delta).max())
       if p>0.12001:raise ValueError("Original bounded G1 joint reference exceeded")
       if p>1e-7:
          real_recovery_actuation+=1
          if first_active is None:first_active=float(item["time_s"])
       maxrec=max(maxrec,p)
    return dict(case=case["id"],control=ctrl,
      original_source=report,original_source_sha256=digest(report),
      raw_checkpoint_sha256=expected,
      original_shot_goal=bool(shot["goal"]),
      first_actual_ball_contact_s=shot["contact_time_s"],
      right_instep_impulse_fraction=float(shot["requested_surface_impulse_fraction"]),
      original_qdot_speed_exceedance_sample_joints=int(speed["actual_speed_exceedance_sample_joint_count"]),
      actual_qdot_peak_over_original_limit_ratio=float(speed["maximum_actual_speed_ratio"]),
      worst_qdot=speed["worst_actual_speed"],
      original_outgoing_ball_speed_xy_mps=exit_speed,
      physical_steps=len(rows),
      original_standing_seconds=float(shot["standing_seconds"]),
      original_failure_reasons=score["failure_reasons"],
      actual_recovery_PPO_actuated_policy_steps=real_recovery_actuation,
      actual_recovery_delta_peak_rad=maxrec,
      first_recovery_actuation_s=first_active,
      original_physical_gates=reward_gates,
      strict_original_full_task_pass=all(reward_gates.values()),
      initial_robot_state=d["initial_robot_state"],
      initial_ball_xy=d["points"][0],
      preimpact_physics_samples=[
          dict(time_s=x["time_s"],root_xyz=x["root_xyz"],ball_xyz=x["ball_xyz"],
               joint_pos=x["joint_pos"],joint_vel=x["joint_vel"])
          for x in rows if x["time_s"]<=1.20],
      source_warning="G1 original simulated physics only, no professional human kinetic ground-truth or hardware safety"
    )

def run():
 p=argparse.ArgumentParser()
 p.add_argument("--output",required=True)
 a=p.parse_args()
 out=ROOT/a.output
 if out.exists():raise FileExistsError(out)
 cfg=json.loads((ROOT/STUDY).read_text())
 for p,h in cfg["source_sha256"].items():
    if digest(p)!=h:raise ValueError("Source drift: "+p)
 checkpt=cfg["training"]["output_directory"]+"/last_model.pt"
 training=json.loads((ROOT/cfg["training"]["output_directory"]/"result.json").read_text())
 if digest(checkpt)!=training["checkpoint_sha256"] or digest(checkpt)!=cfg["actual_final_trained_checkpoint_sha256"]:
    raise ValueError("Training checkpoint or study audit mismatched")
 c_old=cfg["original_v32_evaluation_config"]
 c_new=cfg["final_evaluation_config"]
 if digest(c_new)!=cfg["final_evaluation_config_sha256"]:
    raise ValueError("Physical evaluation model config changed")
 rows=[]
 for case in cfg["predeclared_eval_exact_six_ball_conditions"]:
    old=inspect(case,"original_v32",cfg,c_old)
    new=inspect(case,"learned",cfg,c_new)
    if old["initial_robot_state"]!=new["initial_robot_state"] or old["initial_ball_xy"]!=new["initial_ball_xy"]:
       raise ValueError("Paired G1 and ball initial conditions differ")
    first=old["first_actual_ball_contact_s"]
    second=new["first_actual_ball_contact_s"]
    if first is not None and second is not None:
       prefix=np.asarray([x["joint_pos"] for x in old["preimpact_physics_samples"]],float)
       actual=np.asarray([x["joint_pos"] for x in new["preimpact_physics_samples"]],float)
       physically_exact_original_prefix=prefix.shape==actual.shape and np.array_equal(prefix,actual)
    else:
       physically_exact_original_prefix=None
    for record in (old,new):
       record.pop("initial_robot_state")
       record.pop("initial_ball_xy")
       record.pop("preimpact_physics_samples")
    rows.append(dict(id=case["id"],exact_parameter_tuple=[*case["ball_offset_xy_m"],case["velocity_y_delta_mps"]],
       old=old,learned=new,exact_first1p2s_robot_q_prefix=physically_exact_original_prefix))
 success_old=sum(r["old"]["strict_original_full_task_pass"] for r in rows)
 success_new=sum(r["learned"]["strict_original_full_task_pass"] for r in rows)
 report=dict(version="0.1.47",
   frozen_training_study_sha256=digest(STUDY),
   original_training_update_count=training["updates"],
   PPO_warmstart_original_sha256=cfg["frozen_recovery_actor_start_SHA256"],
   actual_trained_PPO_checkpoint_sha256=cfg["actual_final_trained_checkpoint_sha256"],
   actual_full14s_new_test_conditions=len(rows),
   original_v32_strict_success=success_old,
   new_finetuned_recovery_strict_success=success_new,
   old_fail_new_pass=sum(not r["old"]["strict_original_full_task_pass"] and r["learned"]["strict_original_full_task_pass"] for r in rows),
   old_pass_new_fail=sum(r["old"]["strict_original_full_task_pass"] and not r["learned"]["strict_original_full_task_pass"] for r in rows),
   original_joint_vel_and_torque_limits_unchanged=True,
   original_foot_geometry_and_soccer_ball_unchanged=True,
   no_test_checkpoint_selection=True,
   training_not_human_kick_policy=True,
   only_six_once_each_not_robust_population_statistics=True,
   model_not_deployed=True,
   scientific_scope="Learned post-shot G1 PPO recovery safety-margin reward on 18 known development cases; six exact new robot-ball conditions, one genuine original PhysX 14s run each.",
   records=rows)
 out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
 print("V047_TRUE_G1_PPO_EVALUATION",json.dumps(dict(
    train_updates=report["original_training_update_count"],
    n=report["actual_full14s_new_test_conditions"],
    old_pass=success_old,new_pass=success_new,
    improved=report["old_fail_new_pass"],regressed=report["old_pass_new_fail"],
    cases=[dict(id=x["id"],old=x["old"]["strict_original_full_task_pass"],
      new=x["learned"]["strict_original_full_task_pass"],
      old_qdot=x["old"]["original_qdot_speed_exceedance_sample_joints"],
      new_qdot=x["learned"]["original_qdot_speed_exceedance_sample_joints"],
      original_precontact_motion_same=x["exact_first1p2s_robot_q_prefix"],
      new_ball_mps=x["learned"]["original_outgoing_ball_speed_xy_mps"],
      old_ball_mps=x["old"]["original_outgoing_ball_speed_xy_mps"]) for x in rows]),ensure_ascii=False),flush=True)
if __name__=="__main__":run()
