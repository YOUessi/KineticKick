#!/usr/bin/env python3
"""v46 independently validate original free-root G1 14s shooting and landing.

ALL actual physical qdot substeps, original right instep contact, complete ball
goal, 14s, runup/shot, post-contact stance, sources and original 29DoF action
envelope. The newly fixed landing transfer is NOT a learned biomechanics policy
and must genuinely change the joint commands before any performance claim.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from strike_baseline_metrics import validated_episode,task_outcome,first_force_episode
from audit_p0_dynamic_strike import limit_observations
STUDY="configs/landing_transfer_study_v046.json"

def sha(rel):
 h=hashlib.sha256()
 with (ROOT/rel).open("rb") as f:
  for buf in iter(lambda:f.read(1<<20),b""):h.update(buf)
 return h.hexdigest()

def compare_zero(study):
 cfg=study["v46_zero_config"]
 p=study["zero_physical"]
 orig=study["zero_reference"]
 z=json.loads((ROOT/p).read_text())
 old=json.loads((ROOT/orig).read_text())
 if z["joint_strike"]["mode"]!="contact_landing_transfer_v046" or z["policy_sha256"]!=sha(cfg):
  raise ValueError("Wrong original v46 zero controller")
 if not z["joint_strike"]["config"]["contact_landing_transfer_v046"]["enabled"] is False:
  raise ValueError("Zero control must be physically disabled")
 a=z["traces"][0];b=old["traces"][0]
 if len(a)!=2800 or len(b)!=2800:raise ValueError("Full 14s zero PhysX contract")
 expected_extras={"root_velocity_w","segment_angular_velocity_w_rad_s","segment_quaternions_wxyz",
     "support_contact_capacity_exhausted","support_foot_friction_force_w_N",
     "support_foot_ground_force_w_N","support_foot_normal_force_w_N"}
 if set(b[0])-set(a[0])!=expected_extras or set(a[0])-set(b[0]):
  raise ValueError("Only known 7 read-only instrumented columns may differ")
 shared=set(a[0])&set(b[0])
 if len(shared)!=28:raise ValueError("Original 28 per-5ms PhysX body signals expected")
 for k,(new,prior) in enumerate(zip(a,b)):
  if set(new)!=set(a[0]) or set(prior)!=set(b[0]):raise ValueError("Midrun schema drift")
  for field in shared:
   if new[field]!=prior[field]:
    raise ValueError(f"Zero physics changed at {k*0.005:.3f}s in {field}")
 for field in ("contacts","kinematics","records","sequence_records",
               "running_evidence","stride_shot_evidence","stride_shot_success"):
  if z.get(field)!=old.get(field):raise ValueError("Zero case altered "+field)
 pruned=[]
 for i,frame in enumerate(z["joint_strike"]["feedback"]):
  delta=frame.get("contact_landing_transfer_v046")
  if not isinstance(delta,dict) or delta["enabled"] is not False:raise ValueError("Zero debug not disabled")
  if max(abs(v) for v in delta["actual_joint_delta_rad"])>0:raise ValueError("Zero added physical target")
  q=dict(frame);q.pop("contact_landing_transfer_v046")
  pruned.append(q)
 if pruned!=old["joint_strike"]["feedback"]:
  raise ValueError("Zero baseline parent actions changed")
 return dict(all_shared_original_physics_identical=True,
   shared_physical_fields_checked=28,physics_steps=2800,
   diagnostic_only_original_extra_fields=sorted(expected_extras),
   all_original_shot_contacts_scoring_equal=True,
   original_action_stream_identical=True,zero_additional_actuator_reference_steps=0,
   zero_output_sha256=sha(p),original_old_v41_sha256=sha(orig),
   full_json_byte_identical=False)

def validate_run(case,ctrl,rel,study):
 r=json.loads((ROOT/rel).read_text())
 cfgpath=study["v41_config"] if ctrl=="style_v41" else study["v46_active_config"]
 c=json.loads((ROOT/cfgpath).read_text())
 if r["root_fixed"] or r["post_initialization_state_writes"]:
  raise ValueError("G1 fixed-root or original physical state was overwritten")
 if r["policy_sha256"]!=sha(cfgpath) or r["joint_strike"]["config_sha256"]!=sha(cfgpath):
  raise ValueError("Runner did not load source-frozen controller config")
 if r["joint_strike"]["checkpoint_sha256"]!=study["frozen_v32_PPO_sha256"]:
  raise ValueError("Original physical G1 recovery PPO changed")
 mode="coordination_style_v041" if ctrl=="style_v41" else "contact_landing_transfer_v046"
 if r["joint_strike"]["mode"]!=mode:raise ValueError("Unexpected physical control actor")
 for f,v in r["joint_strike"]["source_sha256"].items():
  if sha("tools/"+f)!=v:raise ValueError("G1 physical runtime code drift: "+f)
 if not np.allclose(r["joint_strike"]["ball_offset_xy"],case["ball_offset_xy_m"],atol=1e-8,rtol=0):
  raise ValueError("Wrong ball placement")
 if abs(r["joint_strike"]["velocity_y_delta"]-case["velocity_y_delta_mps"])>1e-8:
  raise ValueError("Wrong initial robot forward velocity")
 data=validated_episode(r)
 if not data:raise ValueError("Original full-root physical trace is missing")
 measured=limit_observations(data,r)
 task=task_outcome(r)
 shot=r["records"][0]
 contact=first_force_episode(data)
 speed=None if contact is None else contact["outgoing_speed_xy_mps"]
 qd_limit_legal=measured["actual_speed_exceedance_sample_joint_count"]==0
 legal_surface=shot["measured_surface"]=="instep" and shot["surface_match"] is True and \
     float(shot["requested_surface_impulse_fraction"])>=.95 and \
     r["sequence_records"][0]["illegal_body_ball_contact"] is False
 completed=len(data)==2800 and abs(data[-1]["time_s"]-14.)<1e-6 and r["termination_reasons"]==[[]]
 gates=dict(
  original_full_14s=completed,
  original_real_joint_speed_limit=qd_limit_legal,
  original_legal_right_instep=bool(legal_surface),
  original_full_soccer_ball_goal=shot["goal"] is True,
  actual_recovery=shot["recovered"] is True and shot["standing_seconds"]>=1.,
  original_runup=bool(r["running_evidence"]["running_success"]),
  original_continuous_shot=bool(r["stride_shot_evidence"]["continuous_running_contact"]),
  original_scorer_success=bool(task["success"]),
  original_ball_speed_3_5_mps=speed is not None and speed>=3.5)
 actual_steps=0
 qualified_steps=0
 descent_steps=0
 earliest=None
 peak=0.
 slew=0.
 physical_issues=[]
 previous=np.zeros(29,dtype=float)
 if ctrl=="landing_v46":
  for frame in r["joint_strike"]["feedback"]:
   diag=frame.get("contact_landing_transfer_v046")
   if not isinstance(diag,dict) or not diag.get("enabled"):
    raise ValueError("No G1 v46 real command debug telemetry")
   after=np.asarray(diag["actual_joint_delta_rad"],float)
   before=np.asarray(diag["previous_joint_delta_rad"],float)
   original=np.asarray(diag["baseline_action"],float)
   final=np.asarray(diag["final_action"],float)
   if any(k.shape!=(29,) or not np.all(np.isfinite(k)) for k in (after,before,original,final)):
    raise ValueError("Bad 29DoF action contract")
   if np.max(abs(before-previous))>2e-6:
    physical_issues.append("previous_delta_not_continuous")
   if np.max(abs(after-before))>.00202:
    physical_issues.append("original_20ms_reference_slew")
   if np.max(abs(after))>.01602:
    physical_issues.append("original_reference_amplitude")
   if diag["ball_state_write"] is not False:
    physical_issues.append("external_ball_state_write")
   t=float(frame["time_s"])
   if shot["contact_time_s"] is None:
    if np.max(abs(after))>1e-7:
     physical_issues.append("landing_reference_without_ball_contact")
   elif t<shot["contact_time_s"]-1e-6 and np.max(abs(after))>1e-7:
    physical_issues.append("landing_reference_before_real_ball_contact")
   if float(np.max(abs(after)))>1e-7:
    actual_steps+=1
    if earliest is None: earliest=t
   qualified_steps+=int(bool(diag["qualified"]))
   descent_steps+=int(bool(diag["right_anticipated_landing"]))
   peak=max(peak,float(np.max(abs(after))))
   slew=max(slew,float(np.max(abs(after-before))))
   previous=after
  if completed and len(r["joint_strike"]["feedback"])!=700:
   physical_issues.append("policy_history_missing")
  gates["bounded_landing_reference"]=len(physical_issues)==0
 return dict(
  id=case["id"],controller=ctrl,
  source_path=rel,source_sha256=sha(rel),
  actual_physics_steps=len(data),
  original_strict_pass=all(gates.values()),
  original_task_gates=gates,
  original_task_failure_reasons=task["failure_reasons"],
  first_ball_contact_s=shot["contact_time_s"],
  measured_instep_impulse_fraction=shot["requested_surface_impulse_fraction"],
  goal=shot["goal"],
  outgoing_ball_speed_xy_mps=speed,
  original_qdot_ratio_peak=measured["maximum_actual_speed_ratio"],
  original_qdot_excess_samples=measured["actual_speed_exceedance_sample_joint_count"],
  original_max_qdot_sample=measured["worst_actual_speed"],
  recovered_standing_seconds=shot["standing_seconds"],
  landing_actual_nonzero_control_steps=actual_steps,
  landing_qualified_policy_steps=qualified_steps,
  landing_anticipated_right_foot_steps=descent_steps,
  first_landing_delta_time_s=earliest,
  peak_landing_joint_position_target_delta_rad=peak,
  peak_landing_joint_target_slew_per20ms_rad=slew,
  landing_command_contract_errors=sorted(set(physical_issues)),
  robot_initial_state=r["initial_robot_state"],
  ball_initial_xy=r["points"][0],
  limitations=["One PhysX run for this already fixed initial condition, not hardware or naturalness evidence",
    "The v46 wholebody contact-phase position target is an engineering candidate, not a learned professional kick"]
 )

def audit():
    cfg=json.loads((ROOT/STUDY).read_text())
    for p,h in cfg["source_sha256"].items():
        if sha(p)!=h:raise ValueError("Frozen study source changed: "+p)
    zero=compare_zero(cfg)
    old=[]
    for c in cfg["old_diagnostics"]:
        r=validate_run(c,"landing_v46",c["output"],cfg)
        old.append(dict(id=c["id"],result=r))
    pairs=[]
    for c in cfg["six_new_ball_conditions"]:
        a=validate_run(c,"style_v41",c["outputs"]["original_v41"],cfg)
        b=validate_run(c,"landing_v46",c["outputs"]["landing_v46"],cfg)
        if a["robot_initial_state"]!=b["robot_initial_state"] or a["ball_initial_xy"]!=b["ball_initial_xy"]:
            raise ValueError("Paired G1/ball initial physical states differ")
        for side in (a,b):
            side.pop("robot_initial_state")
            side.pop("ball_initial_xy")
        pairs.append(dict(id=c["id"],ball_offset_xy_m=c["ball_offset_xy_m"],
          velocity_y_delta_mps=c["velocity_y_delta_mps"],
          original_v41=a,landing_v46=b))
    old_count=sum(c["original_v41"]["original_strict_pass"] for c in pairs)
    new_count=sum(c["landing_v46"]["original_strict_pass"] for c in pairs)
    return dict(version="0.1.46",
       original_study_sha256=sha(STUDY),
       zero_parity=zero,old_diagnostics=old,
       exact_new_ball_parameter_conditions=len(pairs),
       fixed_original_v41_full_pass_count=old_count,
       new_landing_v46_full_pass_count=new_count,
       improved_cases=sum(not c["original_v41"]["original_strict_pass"] and c["landing_v46"]["original_strict_pass"] for c in pairs),
       regressed_cases=sum(c["original_v41"]["original_strict_pass"] and not c["landing_v46"]["original_strict_pass"] for c in pairs),
       new_landing_actual_nonzero_case_count=sum(c["landing_v46"]["landing_actual_nonzero_control_steps"]>0 for c in pairs),
       all_original_motor_qdot_limits_unchanged=True,
       once_only_each_condition=True,
       new_exact_ball_parameters_not_new_physics_domain=True,
       no_new_G1_PPO_training=True,no_success_filtering=True,
       new_landing_controller_not_certified_safe=True,
       new_professional_biomechanics_data=False,
       has_robot_hardware_validation=False,
       existing_default_v5_not_replaced=True,
       interpretation="Original PhysX paired comparison. Only genuine full task pass with no actual qdot over-limit counts; safety improvement is NOT implied by a few new ball conditions.",
       results=pairs)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--zero-only",action="store_true")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    p=ROOT/args.output
    if p.exists():raise FileExistsError(p)
    if args.zero_only:
        s=json.loads((ROOT/STUDY).read_text())
        for f,h in s["source_sha256"].items():
            if sha(f)!=h:raise ValueError("Source changed before zero audit "+f)
        result=dict(version="0.1.46",zero_parity=compare_zero(s),ready=True)
    else: result=audit()
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("V046_LANDING_PHYSX_AUDIT",json.dumps({
     "zero_pass":result["zero_parity"]["all_shared_original_physics_identical"],
     "old":result.get("fixed_original_v41_full_pass_count"),
     "new":result.get("new_landing_v46_full_pass_count"),
     "improved":result.get("improved_cases"),
     "regressed":result.get("regressed_cases"),
     "real_landing_case_count":result.get("new_landing_actual_nonzero_case_count"),
     "new_n":result.get("exact_new_ball_parameter_conditions"),
     "known_diagnostics":[{"id":x["id"],"gate":x["result"]["original_strict_pass"],
         "actuated":x["result"]["landing_actual_nonzero_control_steps"],
         "qdot_violation":x["result"]["original_qdot_excess_samples"]} for x in result.get("old_diagnostics",[])],
     "new_cases":[{"id":x["id"],"old":x["original_v41"]["original_strict_pass"],
         "new":x["landing_v46"]["original_strict_pass"],
         "actuated":x["landing_v46"]["landing_actual_nonzero_control_steps"],
         "old_over":x["original_v41"]["original_qdot_excess_samples"],
         "new_over":x["landing_v46"]["original_qdot_excess_samples"]} for x in result.get("results",[])]
    },ensure_ascii=False),flush=True)
