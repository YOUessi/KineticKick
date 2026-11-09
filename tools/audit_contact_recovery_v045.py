#!/usr/bin/env python3
"""v45 independent original full PhysX audit, whole-body qdot and task gates.

Checks actual 2800x5ms original G1 joint velocity, full soccer ball goal,
original right-instep impact and 14-second recovery. A post-contact POSITION
reference may oppose qdot but it is not a certified joint-speed shield.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,math,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from strike_baseline_metrics import validated_episode,task_outcome,first_force_episode
from audit_p0_dynamic_strike import limit_observations

STUDY="configs/contact_recovery_governor_v045.json"

def sha(name):
 h=hashlib.sha256()
 with (ROOT/name).open("rb") as f:
  for part in iter(lambda:f.read(1<<20),b""):h.update(part)
 return h.hexdigest()

def validate_case(runpath,cfgpath,controller,context,study):
    if not (ROOT/runpath).is_file():raise FileNotFoundError(runpath)
    r=json.loads((ROOT/runpath).read_text())
    cfg=json.loads((ROOT/cfgpath).read_text())
    if r["root_fixed"] is not False or r["post_initialization_state_writes"] is not False:
        raise ValueError("Free-base original G1 must retain physical contact/state")
    if r["joint_strike"]["config_sha256"]!=sha(cfgpath) or r["policy_sha256"]!=sha(cfgpath):
        raise ValueError("Wrong bound original runner/config hash")
    if r["joint_strike"]["checkpoint_sha256"]!=study["original_v32_checkpoint_sha256"]:
        raise ValueError("Changed original G1 frozen PPO checkpoint")
    for name,expected in r["joint_strike"]["source_sha256"].items():
        if sha("tools/"+name)!=expected:
            raise ValueError("Runtime source SHA different: "+name)
    mode="contact_recovery_governor_v045" if controller=="governor_v45" else "coordination_style_v041"
    if r["joint_strike"]["mode"]!=mode:raise ValueError("Incorrect actual physical controller type")
    if not np.allclose(r["joint_strike"]["ball_offset_xy"],context["ball_offset_xy_m"],rtol=0,atol=1e-8):
        raise ValueError("Wrong registered ball location")
    if abs(r["joint_strike"]["velocity_y_delta"]-context["velocity_y_delta_mps"])>1e-8:
        raise ValueError("Wrong registered root velocity")
    original=validated_episode(r)
    if not original:raise ValueError("Missing actual G1 physical samples")
    speeds=limit_observations(original,r)
    outcome=task_outcome(r)
    shot=r["records"][0]
    contact=first_force_episode(original)
    out_speed=None if contact is None else contact["outgoing_speed_xy_mps"]
    controller_real=outcome["success"]
    legal=(shot["measured_surface"]=="instep" and shot["surface_match"] is True and
           shot["requested_surface_impulse_fraction"]>=.95 and
           r["sequence_records"][0]["illegal_body_ball_contact"] is False)
    full=(len(original)==2800 and abs(original[-1]["time_s"]-14.)<1e-6
          and r["termination_reasons"]==[[]])
    safe=speeds["actual_speed_exceedance_sample_joint_count"]==0
    recovered=(shot["recovered"] is True and shot["standing_seconds"]>=1.)
    gates=dict(full_14s=full,
      original_real_joint_speed_limit=safe,
      right_instep_contact=bool(legal),
      full_soccer_ball_goal=shot["goal"] is True,
      postcontact_recovered=recovered,
      original_task_success=controller_real is True,
      running=bool(r["running_evidence"]["running_success"]),
      continuous_runup_and_shot=bool(r["stride_shot_evidence"]["continuous_running_contact"]),
      ball_speed_at_least_3_5_mps=(out_speed is not None and out_speed>=3.5))
    actual_guard=[]
    braked=0
    eligible=0
    peak=0.
    slew=0.
    first_nonzero=None
    dangerous_count=0
    if controller=="governor_v45":
        if cfg["contact_recovery_governor_v045"]["enabled"] is not True:raise ValueError("Governed run disabled")
        parent=None
        for i,frame in enumerate(r["joint_strike"]["feedback"]):
            x=frame.get("contact_recovery_governor_v045")
            if not isinstance(x,dict) or x.get("enabled") is not True:
                raise ValueError("Missing actual governor debug/qualification")
            before=np.asarray(x["previous_brake_rad"],float)
            after=np.asarray(x["actual_brake_rad"],float)
            orig_action=np.asarray(x["original_action"],float)
            actual_action=np.asarray(x["final_action"],float)
            if any(z.shape!=(29,) or not np.isfinite(z).all() for z in (
                  before,after,orig_action,actual_action)):
                raise ValueError("Invalid control 29DoF trajectory")
            if not x.get("no_state_writes",False):
                actual_guard.append("external_state_write_flag")
            if parent is not None and np.max(abs(parent-before))>2e-6:
                actual_guard.append("safety_delta_history_not_continuous")
            if np.max(abs(after-before))>.00403:
                actual_guard.append("governor_joint_target_slew")
            if np.max(abs(after))>.02501:
                actual_guard.append("governor_joint_target_amplitude")
            t=float(frame["time_s"])
            if shot["contact_time_s"] is None:
                if np.max(abs(after))>1e-7:
                    actual_guard.append("braking_without_real_foot_ball_impact")
            elif t<shot["contact_time_s"]-1e-6 and np.max(abs(after))>1e-7:
                actual_guard.append("braking_before_actual_right_foot_impact")
            if x.get("braking_actuator_count",0)>0:
                dangerous_count+=1
            if x.get("postcontact_eligible",False):eligible+=1
            if float(np.max(abs(after)))>1e-7:
                braked+=1
                if first_nonzero is None:first_nonzero=t
            parent=after
            peak=max(peak,float(np.max(abs(after))))
            slew=max(slew,float(np.max(abs(after-before))))
        if len(r["joint_strike"]["feedback"])!=700 and len(original)==2800:
            actual_guard.append("missing_policy_steps_in_full_physics")
    if controller=="governor_v45" and actual_guard:
        gates["governor_command_contract"]=False
    if controller=="governor_v45" and not actual_guard:
        gates["governor_command_contract"]=True
    report=dict(path=runpath,source_sha256=sha(runpath),
      controller=controller,case=context["id"],
      actual_physx_samples=len(original),
      original_full_14s_completed=full,
      real_joint_velocity_overlimit_sample_joints=speeds["actual_speed_exceedance_sample_joint_count"],
      real_joint_velocity_max_ratio=speeds["maximum_actual_speed_ratio"],
      worst_actual_joint=speeds["worst_actual_speed"],
      first_contact_s=shot["contact_time_s"],
      correct_instep_impulse_fraction=shot["requested_surface_impulse_fraction"],
      real_outgoing_ball_speed_xy_mps=out_speed,
      actual_goal=shot["goal"],actual_standing_s=shot["standing_seconds"],
      original_task_failure_reasons=outcome["failure_reasons"],
      original_g1_task_gates=gates,
      original_strict_pass=all(gates.values()),
      actual_governor_nonzero_steps=braked,
      governor_eligible_steps=eligible,
      governor_dangerous_motion_control_steps=dangerous_count,
      governor_first_nonzero_s=first_nonzero,
      governor_max_reference_rad=peak,
      governor_max_joint_reference_change_rad=slew,
      governor_source_errors=sorted(set(actual_guard)),
      first_ball_and_robot_state=r["initial_robot_state"],
      initial_ball_xy=r["points"][0],
      audit_scope="Original PhysX, 5-ms observed joint speed and original contact, not a certified G1 hardware safety guarantee")
    return report

def compare_zero(study):
    source=study["zero_physics_trial"]
    path=source["output"]
    z=json.loads((ROOT/path).read_text())
    b=json.loads((ROOT/source["original_reference"]).read_text())
    instrumentation_only_old_fields={
      "root_velocity_w",
      "segment_angular_velocity_w_rad_s",
      "segment_quaternions_wxyz",
      "support_contact_capacity_exhausted",
      "support_foot_friction_force_w_N",
      "support_foot_ground_force_w_N",
      "support_foot_normal_force_w_N"
    }
    # The v41 frozen reference was made via the *instrumented* wrapper;
    # v45 zero used an exact frozen core runner with seven extra diagnostic
    # columns absent. Require EVERY shared true physics field at EVERY 5ms
    # sample to be exactly equal, and verify the omissions are exactly the
    # known extra sensor-recording columns (not hidden physical differences).
    raw_new=z["traces"][0];raw_old=b["traces"][0]
    if len(raw_new)!=2800 or len(raw_old)!=2800:
        raise ValueError("Zero G1 original full length changed")
    if set(raw_old[0])-set(raw_new[0])!=instrumentation_only_old_fields:
        raise ValueError("Unexpected missing original physical or diagnostic channels")
    if set(raw_new[0])-set(raw_old[0]):
        raise ValueError("Unexpected new channels in zero original trace")
    common=set(raw_new[0])&set(raw_old[0])
    if len(common)!=28:
        raise ValueError("Original 28 simultaneous core physical channels contract changed")
    for i,(new,old) in enumerate(zip(raw_new,raw_old)):
        if set(new)!=set(raw_new[0]) or set(old)!=set(raw_old[0]):
            raise ValueError("Mid-episode channel schema drift")
        if any(new[k]!=old[k] for k in common):
            raise ValueError("Zero governor changed actual common 5ms physics at "+str(i))
    for field in ("contacts","kinematics","records","sequence_records",
                  "running_evidence","stride_shot_evidence","stride_shot_success"):
        if z.get(field)!=b.get(field):raise ValueError("Disabled governor altered original "+field)
    logged=[]
    for row in z["joint_strike"]["feedback"]:
        governor=row["contact_recovery_governor_v045"]
        if governor["enabled"] is not False:raise ValueError("Disabled model accidentally active")
        if max(abs(v) for v in governor["actual_brake_rad"])>0:
            raise ValueError("Zero governor modified actual G1 29DoF")
        copy=dict(row)
        copy.pop("contact_recovery_governor_v045",None)
        logged.append(copy)
    if logged!=b["joint_strike"]["feedback"]:raise ValueError("Disabled governor changed original action history")
    return dict(original_reference=source["original_reference"],
        reference_sha256=sha(source["original_reference"]),observed_sha256=sha(path),
        original_shared_28_physical_channels_identical_every_5ms=True,
        original_instrumented_only_7_channels_absent_in_new_core_runner=sorted(instrumentation_only_old_fields),
        entire_json_byte_identical=False,
        original_actions_identical=True,
        zero_added_governor_control_steps=0,
        statement="All 28 shared physical channels over original 2800x5ms, all original contacts/kinematics/scoring and 700 G1 actions are EXACT; seven original diagnostic-only sensor channels are not present in the v45 core-runner output, so full trace schema is not byte-identical")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--zero-only",action="store_true")
    args=p.parse_args()
    output=ROOT/args.output
    if output.exists():raise FileExistsError(output)
    study=json.loads((ROOT/STUDY).read_text())
    if study["version"]!="0.1.45":
        raise ValueError("Wrong original fixed program")
    for path,expected in study["source_sha256"].items():
        if sha(path)!=expected:raise ValueError("Trial source code drift: "+path)
    zero=compare_zero(study)
    if args.zero_only:
        audit=dict(version="0.1.45",zero_exact=zero,ready_for_predeclared_trials=True)
    else:
        pairs=[]
        known=[]
        for c in study["known_development_diagnostics"]:
            row=validate_case(c["output"],study["governor_enabled_config"],"governor_v45",c,study)
            ref=json.loads((ROOT/c["reference"]).read_text())
            known.append(dict(case=c["id"],original_hazard_baseline_sha256=sha(c["reference"]),
                governor=row,prior_original_safety_success=ref["records"][0]["goal"] is True))
        for c in study["six_new_parameter_conditions"]:
            baseline=validate_case(c["outputs"]["style_v41"],study["original_baseline_config"],
                "style_v41",c,study)
            changed=validate_case(c["outputs"]["governor_v45"],study["governor_enabled_config"],
                "governor_v45",c,study)
            if baseline["first_ball_and_robot_state"]!=changed["first_ball_and_robot_state"]:
                raise ValueError("G1 initial joint state across paired simulations mismatches")
            if baseline["initial_ball_xy"]!=changed["initial_ball_xy"]:
                raise ValueError("Ball initial coordinate across paired simulations mismatches")
            for r in (baseline,changed):r.pop("first_ball_and_robot_state");r.pop("initial_ball_xy")
            pairs.append(dict(id=c["id"],ball_offset_xy_m=c["ball_offset_xy_m"],
                root_velocity_y_delta_mps=c["velocity_y_delta_mps"],
                original_style=baseline,contact_governor=changed))
        old=sum(x["original_style"]["original_strict_pass"] for x in pairs)
        new=sum(x["contact_governor"]["original_strict_pass"] for x in pairs)
        delta=dict(style_to_governor_improved=sum(not x["original_style"]["original_strict_pass"] and x["contact_governor"]["original_strict_pass"] for x in pairs),
                   style_to_governor_regressed=sum(x["original_style"]["original_strict_pass"] and not x["contact_governor"]["original_strict_pass"] for x in pairs))
        audit=dict(version="0.1.45",
            study_sha256=sha(STUDY),zero_exact=zero,
            known_old_development_diagnostics=known,
            fixed_new_physical_condition_count=len(pairs),
            paired_original_style_strict_pass=old,paired_contact_governor_strict_pass=new,
            paired_change=delta,
            original_real_limits_always_authoritative=True,
            original_V44_unreliable_forecaster_not_used=True,
            unseen_exact_parameters_not_independent_new_robot_domain=True,
            population_robustness_established=False,
            human_professional_explosive_instep_established=False,
            new_training_performed=False,
            physical_data_is_six_once_each=True,
            no_success_only_reporting=True,
            default_robot_controller_not_replaced=True,
            results=pairs)
    output.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
    if args.zero_only:
        print("V045_GOVERNOR_ZERO_PARITY_OK",json.dumps(zero),flush=True)
    else:
        print("V045_GOVERNOR_PHYSICS_FINISHED",json.dumps(dict(
            old=old,new=new,change=delta,
            samples=[dict(case=c["id"],
                old=c["original_style"]["original_strict_pass"],
                governor=c["contact_governor"]["original_strict_pass"],
                old_speed=c["original_style"]["real_outgoing_ball_speed_xy_mps"],
                governor_speed=c["contact_governor"]["real_outgoing_ball_speed_xy_mps"],
                original_qdot_violation=c["original_style"]["real_joint_velocity_overlimit_sample_joints"],
                new_qdot_violation=c["contact_governor"]["real_joint_velocity_overlimit_sample_joints"],
                governor_active_steps=c["contact_governor"]["actual_governor_nonzero_steps"],
                governor_failed_gates=[k for k,v in c["contact_governor"]["original_g1_task_gates"].items() if not v])
                for c in pairs]),ensure_ascii=False),flush=True)
if __name__=="__main__":main()
