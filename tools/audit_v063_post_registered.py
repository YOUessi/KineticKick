#!/usr/bin/env python3
"""Post-registered INDEPENDENT evaluator for immutable v63 synthetic PhysX records.

Corrections: the preregistered initial audit_v063 script passed frames rather
than raw dictionaries into compare_raw. We retain that signed source and use
this separately hashed, repaired independent read-only evaluator instead.
"""
from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from run_v063_phase_batch import sha,check_real,config,PROTOCOL,PROGRESS
from audit_v059_observer_parity import compare_raw
from audit_v060_observer_factorial import assert_complete_physics_fields
def compare_frames(a,b):
    return compare_raw({"traces":[a]},{"traces":[b]})
def dynamic_metrics(data):
    trace=data["traces"][0]
    names=data["joint_names"]
    t=np.asarray([x["time_s"] for x in trace],float)
    vel=np.asarray([x["joint_vel"] for x in trace],float)
    q=np.asarray([x["joint_pos"] for x in trace],float)
    limits=np.asarray(data["joint_velocity_limits"],float).reshape(-1)
    assert q.shape==(len(trace),29) and vel.shape==q.shape and limits.shape==(29,)
    peak=np.max(abs(vel)/limits[None],axis=0)
    c=data["records"][0].get("contact_time_s")
    covered_before=bool(c is not None and min(t)<=c-.4+1e-5)
    covered_after=bool(c is not None and max(t)>=c+1.25-1e-5)
    pre=(t>=max(0,(c or 1.57)-.4))&(t<=(c or 1.57))
    if not pre.any():pre=t<=min(max(t),1.57)
    groups={
        "hip":("left_hip_pitch_joint","right_hip_pitch_joint","left_hip_roll_joint","right_hip_roll_joint","left_hip_yaw_joint","right_hip_yaw_joint"),
        "torso":("waist_yaw_joint","waist_roll_joint","waist_pitch_joint"),
        "swing_leg":("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"),
        "arms":("left_shoulder_pitch_joint","right_shoulder_pitch_joint","left_shoulder_roll_joint","right_shoulder_roll_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_elbow_joint","right_elbow_joint"),
    }
    group_metrics={}
    for key,member in groups.items():
        ids=[names.index(x) for x in member]
        group_metrics[key]=dict(
            measured_joint_speed_rms_precontact_rad_s=float(np.sqrt(np.mean(vel[np.ix_(pre,ids)]**2))),
            measured_joint_range_precontact_rad=float(np.mean(np.ptp(q[np.ix_(pre,ids)],axis=0))))
    supporting=np.array([[x["support_foot_force_N"]["left"],x["support_foot_force_N"]["right"]] for x in trace])
    w=np.array([x["left_wrist_hip_link_origin_distance_m"] for x in data["safe_body_v062"]["rows"]])
    nonfoot=np.array([x["nonfoot_contact_N"] for x in trace])
    return dict(physical_rows=len(trace),end_time_s=float(t[-1]),observed_impact_s=c,
        impact_window_pre_0p4s_available=covered_before,
        impact_window_post_1p25s_available=covered_after,
        all_29_measured_qdot_ratios=peak.tolist(),
        max_qdot_ratio=float(peak.max()),
        any_actual_joint_speed_limit_excess=bool((peak>1+1e-6).any()),
        min_wrist_hip_origin_m=float(w.min()),
        maximum_net_nonfoot_force_N=float(nonfoot.max()),
        maximum_left_stance_force_N=float(supporting[:,0].max()),
        maximum_right_stance_force_N=float(supporting[:,1].max()),
        group_proxy_measures_not_naturalness_labels=group_metrics)
def analyze():
    c=config()
    state=json.loads(PROGRESS.read_text())
    assert state["protocol_sha256"]==sha(PROTOCOL)
    if len(state["runs"])!=8:raise RuntimeError("Incomplete preregistered batch")
    by_id={};reports=[]
    for case,old in zip(c["cases"],state["runs"]):
        if case["id"]!=old["id"]:raise RuntimeError("Case reorder")
        path=ROOT/case["output"];log=path.with_suffix(".log")
        if not log.exists() or sha(log)!=old["log_sha256"]:raise RuntimeError("Log hash drift")
        if old["status"]=="VALID_G1":
            if not path.exists() or sha(path)!=old["raw_sha256"]:raise RuntimeError("Physical JSON drift")
            raw=json.loads(path.read_text())
            evidence=check_real(case,raw,log.read_text(errors="replace"),old["return_code"],c)
            if not evidence["strict_pass"]==old["strict_pass"]:raise RuntimeError("Scoring drift")
            metrics=dynamic_metrics(raw)
            report=dict(id=case["id"],status="VALID_G1",strict_pass=evidence["strict_pass"],
                controls=case["policy"],seed=case["seed"],case=case["ball_case"],
                actual_motion=metrics,original_evaluation_record=raw["records"][0],
                original_sequence_success=raw["sequence_records"][0].get("sequence_success"),
                termination=raw["termination_reasons"],
                candidate_v063=raw.get("phase_v063_summary"),
                sha256=sha(path))
            by_id[case["id"]]=raw
        else:report=dict(id=case["id"],status=old["status"],
            strict_pass=None,reason=old.get("reason","no valid physical evidence"))
        reports.append(report)
    comparisons=[]
    for group in ("center","hold_a","hold_b","hold_c"):
        b=by_id.get(group+"_v32");p=by_id.get(group+"_synth")
        if not b or not p:
            comparisons.append(dict(case=group,certified_pair=False,reason="Missing one physical episode"))
            continue
        pre=compare_frames(b["traces"][0][:100],p["traces"][0][:100])
        # Before first possible 0.86s phase action, physics must agree within
        # same seed and initial state for causal candidate comparison.
        if not pre["matches_at_abs_tol_1e_6"]:
            comparisons.append(dict(case=group,certified_pair=False,
                reason="No valid same-seed pre-action physics parity",initial_parity=pre))
            continue
        action=p["phase_v063_summary"]
        comparisons.append(dict(case=group,certified_pair=True,
            original_v32_pass=b["stride_shot_success"],synthetic_candidate_pass=p["stride_shot_success"],
            initial_0_to_0p5_s_all_25_physical_fields=pre,
            candidate_effective_nonzero_steps=action["actual_nonzero_action_steps"],
            candidate_eligible_steps=action["eligible_steps"],
            candidate_max_joint_applied_rad=action["actual_max_joint_delta_rad"],
            all_guard_accept_counts=action["safety_gate_true_counts"],
            candidate_not_human_trained=True))
    historical=None
    if "center_v32" in by_id:
        original=json.loads((ROOT/"reports/v062_cuda_v32_safe_full.json").read_text())
        historical=compare_raw(by_id["center_v32"],original)
    return dict(version="v0.1.63",source="8 frozen original G1/PhysX outcomes, NOT human motion learning",
        protocol_sha256=sha(PROTOCOL),frozen_git_sha=json.loads((ROOT/"reports/v063_protocol_lock.json").read_text())["frozen_git_sha"],
        real_PhysX_attempts=len(reports),
        scorable=sum(x["status"]=="VALID_G1" for x in reports),
        strict_successes=sum(x.get("strict_pass") is True for x in reports),
        synthetic_human_real_instep_contact_truth=False,
        professional_kinetics_and_GRF_truth_available=False,
        GPU_left_wrist_hip_physx_contact_pair_certified=False,
        baseline_center_replay_vs_v062=historical,
        independent_pairs=comparisons,episodes=reports)
def main():
    result=analyze();o=ROOT/"reports/v063_independent_recheck.json"
    if o.exists():raise FileExistsError(o)
    o.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("V063_AUDIT_SUMMARY",result["real_PhysX_attempts"],result["scorable"],result["strict_successes"])
    for row in result["independent_pairs"]:
        print(row["case"],row["certified_pair"],
              row.get("original_v32_pass"),row.get("synthetic_candidate_pass"),
              row.get("candidate_eligible_steps"),row.get("candidate_max_joint_applied_rad"))
if __name__=="__main__":main()
