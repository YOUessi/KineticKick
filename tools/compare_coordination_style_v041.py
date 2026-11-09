#!/usr/bin/env python3
"""v41: compare frozen original PhysX tasks and human-pose proxy style branch.

Never report repeated known-case deterministic physics as unseen skill
generalization or human biomechanics, even if 3/3 goal gates pass.
All loaded raw physical output is recomputed by an independent gate.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_coordination_style_v041 import sha as digest, audit
from strike_baseline_metrics import summarize_strike
from audit_p0_dynamic_strike import limit_observations
from strike_baseline_metrics import validated_episode

STUDY="configs/g1_coordination_style_physics_v041.json"

def assemble():
    cfg=json.loads((ROOT/STUDY).read_text())
    for f,h in cfg["source_sha256"].items():
        if digest(f)!=h:raise ValueError("Prespecified source drift: "+f)
    zero=cfg["zero_trial"]
    if not (ROOT/zero["report"]).exists():raise ValueError("Missing zero-parity PhysX run")
    z=audit(zero["case"],True,cfg)
    if not z["passed"] or z["additional_reference_active_decisions"]!=0:
        raise ValueError("Original zero path not physically equal")
    records=[]
    stop_reason=None
    started_original=False
    for i,row in enumerate(cfg["conditional_case_order"]):
        f=ROOT/row["candidate"]
        if not f.exists():
            if stop_reason is None:stop_reason="incomplete_candidate_record"
            for rest in cfg["conditional_case_order"][i:]:
                if (ROOT/rest["candidate"]).exists():raise ValueError("Trial order violated")
            break
        if stop_reason is not None:
            raise ValueError("Physical evaluations continued after failed condition")
        live=audit(row["id"],False,cfg)
        original=json.loads((ROOT/row["baseline"]).read_text())
        modified=json.loads(f.read_text())
        b=summarize_strike(original)
        c=summarize_strike(modified)
        orig_speed=b["contact"]["outgoing_speed_xy_mps"] if b["contact"] else None
        new_speed=c["contact"]["outgoing_speed_xy_mps"] if c["contact"] else None
        delta=None if orig_speed is None or new_speed is None else new_speed-orig_speed
        if live["passed"] and not live["original_goal"]:
            raise ValueError("Full-goal pass contradicts independent score")
        if live["passed"] and live["additional_reference_active_decisions"]<1:
            raise ValueError("Nominal motion not actually modified")
        records.append({
          "case":row["id"],"source_baseline":row["baseline"],
          "original_sha256":digest(row["baseline"]),
          "new_real_physx_source":row["candidate"],
          "new_sha256":digest(row["candidate"]),
          "original_goal":b["outcome"]["success"],
          "candidate_goal":c["outcome"]["success"],
          "new_physics_pass":live["passed"],
          "original_outgoing_speed_xy_mps":orig_speed,
          "new_outgoing_speed_xy_mps":new_speed,
          "ball_speed_difference_mps":delta,
          "new_speed_retention":live["outgoing_speed_fraction_of_prior"],
          "additional_29joint_control_decisions":live["additional_reference_active_decisions"],
          "peak_added_joint_reference_rad":live["maximum_new_joint_reference_rad"],
          "actual_joint_speed_ratio_max":live["actual_joint_speed_ratio_max"],
          "actual_joint_speed_exceedances":live["real_speed_exceedance_samples"],
          "continuous_standing_s":live["original_standing_s"],
          "full_policy_steps":live["real_policy_decisions"],
          "full_physical_steps":live["real_physics_samples"],
          "failed_gates":live["failed_gates"],
          "style_despite_zero_old_error_on_center":live["active_style_despite_zero_original_error_decisions"]
              if row["id"]=="center" else None,
          "legacy_error_gate_criterion_applicable":row["id"]=="center"
        })
        if not live["passed"]:
            stop_reason="candidate_failed_gate_"+row["id"]
    if not records:raise ValueError("No active G1 physical candidate evaluated")
    if len(records)==len(cfg["conditional_case_order"]) and not stop_reason:stop_reason="all_prespecified_conditions_completed"
    result={
      "version":"0.1.41","protocol_sha256":digest(STUDY),
      "zero_pass":z["passed"],"zero_source":cfg["zero_trial"]["report"],
      "zero_source_sha256":digest(zero["report"]),
      "original_model_unchanged":True,"human_source_is_verified_explosive_instep":False,
      "model_is_learned_human_conditional_policy":False,
      "the_only_human_pose_prior":"3DSP video-lifted 3D shot clip 00001 (train; without verified contact)",
      "n_evaluated_fixed_conditions":len(records),
      "prespecified_conditions_total":len(cfg["conditional_case_order"]),
      "all_evaluated_physically_pass":all(x["new_physics_pass"] for x in records),
      "original_goal_count":sum(bool(x["original_goal"]) for x in records),
      "candidate_goal_count":sum(bool(x["candidate_goal"]) for x in records),
      "original_failures_recovered_in_tested_cases":sum(not x["original_goal"] and x["candidate_goal"] for x in records),
      "candidate_full_pass_count":sum(x["new_physics_pass"] for x in records),
      "stopping_reason":stop_reason,
      "any_unseen_fresh_conditions_used":False,
      "hyperparameter_search":False,
      "robot_physics_and_actuator_limits_same_as_v32":True,
      "verified_lawful_instep_policy":all(x["new_physics_pass"] for x in records),
      "coordination_groundtruth_not_calibrated":True,
      "not_paper_naturalness_evidence":True,
      "cross_task_generalization_estimated":False,
      "stated_scope":"ONE fixed unqualified human 3DSP pose proxy and THREE previously known PhysX development conditions; no new models, no qualified 3D/impact data, no unseen variants.",
      "records":records
    }
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    dest=ROOT/args.output
    if dest.exists():raise FileExistsError(dest)
    result=assemble()
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("V041_PRESPECIFIED_COMPARISON",json.dumps({
      "zero":result["zero_pass"],
      "n":result["n_evaluated_fixed_conditions"],
      "all_pass":result["all_evaluated_physically_pass"],
      "original_goals":result["original_goal_count"],
      "new_goals":result["candidate_goal_count"],
      "recovered_known_misses":result["original_failures_recovered_in_tested_cases"],
      "stop":result["stopping_reason"],
      "all_source_sha256":{r["case"]:r["new_sha256"] for r in result["records"]}},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
