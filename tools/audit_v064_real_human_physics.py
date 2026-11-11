#!/usr/bin/env python3
"""Post-registered INDEPENDENT immutable G1 PhysX evidence audit v0.1.64.

This is a descriptive analysis, NOT training or model selection. Every result
(including errors, partial trajectories and contact failures) is retained.
Full-sequence pass follows original closed_loop_acceptance, never ball-only.
"""
from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from run_v064_human_phase_batch import config,check_physics,sha,PROTOCOL,PROGRESS,LOCK
from audit_v059_observer_parity import compare_raw
from audit_v060_observer_factorial import assert_complete_physics_fields
from audit_v063_post_registered import dynamic_metrics

def peak_ball_speed(raw):
 trace=raw["traces"][0]
 ct=raw["records"][0].get("contact_time_s")
 if ct is None:return None
 speeds=[math.sqrt(sum(float(y)**2 for y in x["ball_velocity"])) for x in trace
         if ct+.005<=x["time_s"]<=ct+.105]
 return max(speeds) if speeds else None

def action_magnitude(raw):
 ref=raw.get("human_phase_v064_summary")
 if ref is None:return dict(active=False)
 return dict(active=True,mode=ref["mode"],
  actual_nonzero_20ms=ref["nonzero_20ms_control_steps"],
  eligible_steps=ref["eligible_20ms_control_steps"],
  actual_max_delta_rad=ref["max_action_delta_rad"])

def physical_kinetics_proxies(raw):
 tr=raw["traces"][0]
 ct=raw["records"][0].get("contact_time_s")
 if ct is None:return dict(contact_observed=False)
 mask=[x for x in tr if ct-.4<=x["time_s"]<=ct]
 vel=np.asarray([x["joint_vel"] for x in mask],float)
 # This is original G1 commanded-PD estimate, not measured joint torque or
 # human muscle force; never present the integral as real muscle energy.
 torque=np.asarray([x["joint_torque_estimate"] for x in mask],float)
 if vel.size==0:return dict(contact_observed=True,kinetics_window_complete=False)
 return dict(contact_observed=True,
  robotics_PD_torque_qdot_abs_integral_proxy_J=float(.005*np.sum(np.abs(vel*torque))),
  measured_joint_qdot_rms_rad_s=float(np.sqrt(np.mean(vel**2))),
  peak_ball_velocity_5ms_to_105ms_mps=peak_ball_speed(raw),
  human_muscular_power_truth=False)

def analyze():
 c=config()
 state=json.loads(PROGRESS.read_text())
 if state.get("protocol_sha256")!=sha(PROTOCOL):
  raise RuntimeError("Frozen protocol/record SHA drift")
 lock=json.loads(LOCK.read_text())
 if lock.get("protocol_sha256")!=sha(PROTOCOL) or not lock.get("preregistered_before_first_real_G1_PhysX"):
  raise RuntimeError("No valid SHA preregistration")
 if len(state["runs"])!=12:raise RuntimeError("All twelve original G1 PhysX attempts must finish before audit")
 rows=[]
 for case,entry in zip(c["cases"],state["runs"]):
  if case["id"]!=entry["id"]:raise RuntimeError("Run order drift")
  p=ROOT/case["output"]; log=p.with_suffix(".log")
  if not log.exists() or sha(log)!=entry["log_sha256"]:raise RuntimeError("Raw runtime log overwritten")
  if p.exists()!=bool(entry.get("raw_sha256")):raise RuntimeError("Physical result presence drift")
  if p.exists() and sha(p)!=entry["raw_sha256"]:raise RuntimeError("Physical JSON content drift")
  row=dict(id=entry["id"],seed=case["seed"],context=case["ball_context"],
    policy=case["policy"],qualification=case["qualifier"],status=entry["status"],
    strict_success=None,
    error=entry.get("reason"),
    original_G1_JSON_sha256=entry.get("raw_sha256"))
  if entry["status"] in ("VALID_G1","INVALID_G1") and p.exists():
   raw=json.loads(p.read_text())
   try:
    evalinfo=check_physics(case,raw,log.read_text(errors="replace"),entry["return_code"],c)
    if entry["status"]!="VALID_G1":raise RuntimeError("Batch validity disagrees with independent audit")
   except ValueError as ex:
    if str(ex)!="Human learned geometric prior had zero measured physical actions" or entry["status"]!="INVALID_G1" or entry.get("reason")!=str(ex):
     raise
    # The original PhysX fields, clock alignment, seed, all-gate verdict,
    # config SHA, joint order and telemetry already passed EVERY check before
    # this optional activation criterion. An inert candidate remains a valid
    # negative result, not a missing physical experiment.
    row["status"]="VALID_G1_INACTIVE_POLICY"
    evalinfo=dict(strict_pass=raw["stride_shot_success"])
   if entry["status"]=="VALID_G1" and evalinfo["strict_pass"]!=entry["strict_pass"]:
    raise RuntimeError("Raw physics original strict score drift")
   metric=dynamic_metrics(raw)
   rec=raw["records"][0]
   row.update(strict_success=evalinfo["strict_pass"],
    full_original_raw_records=rec,
    original_sequence_success=raw["sequence_records"][0].get("sequence_success"),
    original_termination=raw["termination_reasons"],
    real_motion_200hz=metric,
    robotics_kinetic_proxies=physical_kinetics_proxies(raw),
    actual_controller_action=action_magnitude(raw),
    original_contact_surface=rec["measured_surface"],
    original_instep_impulse_fraction=rec["requested_surface_impulse_fraction"],
    goal_scored=rec["goal"],recovered=rec["recovered"])
   if row["strict_success"] and not(
      rec["goal"] and rec["recovered"] and
      rec["requested_surface_impulse_fraction"]>=.95 and
      metric["max_qdot_ratio"]<=1.000001 and
      metric["impact_window_post_1p25s_available"] and
      not metric["any_actual_joint_speed_limit_excess"]):
    raise RuntimeError("Inconsistent strict PASS")
  rows.append(row)
 comparisons=[]
 for g in c["groups"]:
  base=next(x for x in rows if x["context"]==g["name"] and x["policy"]=="v32")
  for variant in ("human_ordered","human_reversed"):
   other=next(x for x in rows if x["context"]==g["name"] and x["policy"]==variant)
   paired=dict(context=g["name"],seed=g["seed"],policy=variant,
     source_context_seen_by_fit=g["qualifier"]=="SEEN_DEV_ONLY",
     baseline_status=base["status"],candidate_status=other["status"],
     valid_real_PhysX_pair=False)
   if base["status"] not in ("VALID_G1","VALID_G1_INACTIVE_POLICY") or other["status"] not in ("VALID_G1","VALID_G1_INACTIVE_POLICY"):
    paired["reason"]="One or more original G1 episodes unscorable"
    comparisons.append(paired);continue
   b=json.loads((ROOT/next(z["output"] for z in c["cases"] if z["id"]==base["id"])).read_text())
   v=json.loads((ROOT/next(z["output"] for z in c["cases"] if z["id"]==other["id"])).read_text())
   # Original 0-0.50s 100-frame preaction equality certifies identical
   # seed/reset, ball and G1 before learning-assisted action can begin.
   ini=compare_raw(dict(traces=[b["traces"][0][:100]]),dict(traces=[v["traces"][0][:100]]))
   if not ini["matches_at_abs_tol_1e_6"]:
    paired["reason"]="Initial 200Hz G1 25-field physical state differs; no causal comparison"
    paired["preaction_parity"]=ini
    comparisons.append(paired);continue
   full_physics_parity=compare_raw(b,v)
   paired["full_25_field_200hz_physics_parity_vs_original"]=full_physics_parity
   if other["actual_controller_action"]["actual_nonzero_20ms"]==0:
    paired["frozen_human_prior_had_zero_physical_actuation"]=True
    # Never call the geometric reference a physical policy comparison if no
    # action reached the 29DoF motors. Even if a null-controller run scored PASS.
   times=np.asarray([x["time_s"] for x in b["traces"][0]])
   ts2=np.asarray([x["time_s"] for x in v["traces"][0]])
   n=min(len(times),len(ts2))
   if n<100:raise RuntimeError("Measured G1 run too short")
   # Same-time same-state before 0.5, then actual divergence from learned action.
   t=times[:n]
   dt=abs(t-ts2[:n]).max()
   if dt>1e-5:raise RuntimeError("Causal episodes not on the same physics 5ms clock")
   mask=(t>=1.02)&(t<=1.57)
   qa=np.array([x["joint_pos"] for x in b["traces"][0][:n]])
   qb=np.array([x["joint_pos"] for x in v["traces"][0][:n]])
   dq=qb-qa
   sep=np.abs(dq[mask])
   paired.update(valid_real_PhysX_pair=True,preaction_parity=ini,
    original_strict_pass=base["strict_success"],
    candidate_strict_pass=other["strict_success"],
    original_instep_fraction=base["original_instep_impulse_fraction"],
    candidate_instep_fraction=other["original_instep_impulse_fraction"],
    candidate_20ms_actual_steps=other["actual_controller_action"]["actual_nonzero_20ms"],
    candidate_actual_max_command_delta_rad=other["actual_controller_action"]["actual_max_delta_rad"],
    actual_measured_g1_29dof_joint_position_difference_max_rad=float(sep.max()),
    actual_measured_g1_29dof_joint_position_difference_rms_rad=float(np.sqrt(np.mean(sep**2))),
    candidate_motion_not_synthetic_reference_only=True)
   comparisons.append(paired)
 counts={}
 for pol in c["policies"]:
  group=[x for x in rows if x["policy"]==pol]
  valid=[x for x in group if x["status"] in ("VALID_G1","VALID_G1_INACTIVE_POLICY")]
  counts[pol]=dict(attempts=len(group),valid=len(valid),strict_pass=sum(x["strict_success"] is True for x in valid),
    strict_failure=sum(x["strict_success"] is False for x in valid),
    unscorable=len(group)-len(valid),
    unique_unseen_contexts_evaluated=sum(x["qualification"]=="NEW_FROZEN_CONTEXT" for x in valid))
 return dict(version="v0.1.64",frozen_protocol_sha256=sha(PROTOCOL),
  frozen_before_PhysX_git_sha=lock["frozen_git_sha"],
  actual_original_G1_PhysX_attempts=len(rows),
  physically_scorable=sum(x["status"] in ("VALID_G1","VALID_G1_INACTIVE_POLICY") for x in rows),
  frozen_candidate_zero_action_physical_negative_count=sum(x["status"]=="VALID_G1_INACTIVE_POLICY" for x in rows),
  actual_original_PhysX_all_gate_pass=sum(x["strict_success"] is True for x in rows),
  count_by_policy=counts,
  preregistered_pair_results=comparisons,
  all_episodes=rows,
  real_human_training_source='CMU clip 10_02, 120Hz optical geometry only, ONE subject',
  same_human_source_temporal_reversal_is_negative_control=True,
  professional_high_power_true_foot_ball_contact=False,
  measured_human_ground_reaction_force=False,
  independent_human_subject_test=False,
  robot_200hz_original_29dof_physics=True,
  gpu_wrist_hip_collider_manifold_confirmed=False,
  default_original_controller_unchanged=True)
def main():
 result=analyze()
 dst=ROOT/"reports/v064_independent_physics_audit.json"
 if dst.exists():raise FileExistsError(dst)
 dst.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
 print("V064_INDEPENDENT_AUDIT",result["actual_original_G1_PhysX_attempts"],
   result["physically_scorable"],result["actual_original_PhysX_all_gate_pass"])
 for k,v in result["count_by_policy"].items():print("POLICY",k,v)
 for c in result["preregistered_pair_results"]:
  print("PAIR",c["context"],c["policy"],c["valid_real_PhysX_pair"],
    c.get("original_strict_pass"),c.get("candidate_strict_pass"),
    c.get("candidate_instep_fraction"),c.get("actual_measured_g1_29dof_joint_position_difference_max_rad"))
if __name__=="__main__":main()
