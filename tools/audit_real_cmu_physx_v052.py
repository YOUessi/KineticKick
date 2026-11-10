#!/usr/bin/env python3
"""Read-only, strict whole-episode PhysX audit for the two frozen real-CMU v52 refs.

Do not confuse a low *pose proxy* MSE with verified human-like power or style.
In particular, joint speed must be read from the complete 5ms physics trace
including >1s AFTER ball impact; only reference slew is insufficient.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from build_cmu_g1_candidate_v052 import (ROOT, CONFIG, OUTPUT, ROBOT,
    digest, phase_window, g1_six_differentiable)
from cmu_g1_semantic_bridge_v051 import human_six_timed, CMU
from g1_contact_coordination_projection_v040 import TorchG1Relations, load_robot_source, interpolate_source
from strike_baseline_metrics import validated_episode, summarize_strike
from audit_p0_dynamic_strike import limit_observations
from free_shooting_metrics import ShotTracker

BASELINE = ROOT / "reports/v032_learned_center.json"
BASELINE_FREEZE = ROOT / "configs/g1_coordination_style_physics_v041.json"


def numeric_six(robot, times):
    root,q=interpolate_source(robot,times)
    fk=TorchG1Relations(robot["joint_names"])
    with torch.no_grad():
        return g1_six_differentiable(fk,torch.tensor(root,dtype=torch.float64),
                                     torch.tensor(q,dtype=torch.float64)).numpy()


def evaluate(clip):
    cfg=json.loads(CONFIG.read_text())
    if cfg["human_clips"]!=["10_02","10_05"] or clip not in cfg["human_clips"]:
        raise ValueError("No new selection after original CMU trial")
    # Exact pinned original G1 task baseline from the preceding independent
    # v41 study: distinct from the v48 original trace used OFFLINE for IK.
    old_pins=json.loads(BASELINE_FREEZE.read_text())["source_sha256"]
    if digest(BASELINE)!=old_pins["reports/v032_learned_center.json"]:
        raise ValueError("Physical baseline frozen v32 original source changed")
    audit_path=OUTPUT/clip/"candidate_audit.json"
    cand=json.loads(audit_path.read_text())
    reference=ROOT/cand["output_file"]
    phys_cfg=ROOT/cand["physics_config"]
    output=ROOT/("reports/v052_cmu_"+clip+"_center_physx.json")
    result=ROOT/("reports/v052_cmu_"+clip+"_physics_audit.json")
    if result.exists():
        raise FileExistsError("Published independent v52 physical verdict is immutable")
    if digest(reference)!=cand["output_sha256"] or digest(phys_cfg)!=cand["physics_config_sha256"]:
        raise ValueError("Candidate or physical config changed since original frozen offline generation")
    c=json.loads(phys_cfg.read_text())
    if c["coordination_style_v041"]["enabled"] is not True or c["human_coordination_reference"]["enabled"] is not True:
        raise ValueError("Original independent physical activation must actually be on")
    if c["human_coordination_reference"]["trajectory_sha256"]!=digest(reference):
        raise ValueError("Physical config is not running the frozen new-CMU reference")
    old=json.loads(BASELINE.read_text())
    new=json.loads(output.read_text())
    if new["root_fixed"] or new["post_initialization_state_writes"]:
        raise ValueError("Faked root/ball physical state rewrite not allowed")
    if new["joint_strike"]["mode"]!="coordination_style_v041":
        raise ValueError("Wrong verified original G1 reference controller")
    if new["joint_strike"]["config_sha256"]!=digest(phys_cfg) or new["policy_sha256"]!=digest(phys_cfg):
        raise ValueError("Recorder did not use frozen physical config")
    if new["joint_strike"]["checkpoint_sha256"]!=old_pins["experiments/post_shot_v032/ppo_s3201/last_model.pt"]:
        raise ValueError("Frozen original postshot balance policy changed")
    for name, expected in new["joint_strike"]["source_sha256"].items():
        if digest(ROOT/"tools"/name)!=expected:
            raise ValueError("Runtime physical controller/source mismatch: "+name)
    full=validated_episode(new)
    prior=validated_episode(old)
    if len(prior)!=2800 or len(old["kinematics"][0])!=700:
        raise ValueError("Baseline not fully physical, independent original 14 seconds")
    measurements=limit_observations(full,new)
    cstr=summarize_strike(new)
    bstr=summarize_strike(old)
    shot=new["records"][0]
    prior_shot=old["records"][0]
    physics_ok=(len(full)==2800 and abs(full[-1]["time_s"]-14.)<1e-6
                and len(new["kinematics"][0])==700
                and new["termination_reasons"]==[[]])
    feedback=new["joint_strike"]["feedback"]
    if len(feedback)!=len(new["kinematics"][0]) or len(feedback)<80:
        raise ValueError("Physical feedback count must match all recorded 20ms controls, including failed episode")
    corrections=[]
    nonzero=0
    eligible=0
    previous=np.zeros(29,dtype=float)
    anomalies=[]
    for index, f in enumerate(feedback):
        h=f.get("human_coordination_reference")
        s=f.get("coordination_style_v041")
        if not isinstance(h,dict) or not isinstance(s,dict) or not h.get("enabled") or not s.get("enabled"):
            raise ValueError("Not actually using the independent STYLE physical controller")
        if h["ball_state_write"] is not False or s["ball_state_write"] is not False:
            anomalies.append("nonphysical_ball_state_write")
        delta=np.asarray(h["applied_delta_rad"],float)
        if delta.shape!=(29,) or not np.isfinite(delta).all():
            raise ValueError("Malformed actual G1 joint target")
        if np.max(abs(delta))>.006002:
            anomalies.append("unbounded_actual_applied_joint_reference")
        if np.max(abs(delta-previous))>.00121:
            anomalies.append("unbounded_actual_control_slew")
        if index*.02<1.05-1e-6 and np.max(abs(delta))>1e-7:
            anomalies.append("new_control_modified_protected_early_prefix")
        if index*.02>=1.82-1e-6 and np.max(abs(delta))>1e-7:
            anomalies.append("new_control_failed_postcontact_release")
        if np.max(abs(delta))>1e-7:
            nonzero+=1
        if s.get("eligible"):
            eligible+=1
        corrections.append(delta)
        previous=delta
    corrections=np.asarray(corrections)
    physical_pose=np.asarray([r["joint_pos"] for r in new["kinematics"][0]],float)
    baseline_pose=np.asarray([r["joint_pos"] for r in old["kinematics"][0]],float)
    physical_time=np.asarray([r["time_s"] for r in new["kinematics"][0]],float)
    pre_mask=(physical_time>=1.17)&(physical_time<=1.55)
    if len(physical_pose)>len(baseline_pose) or not np.allclose(physical_time,
         np.asarray([r["time_s"] for r in old["kinematics"][0][:len(physical_pose)]],float),atol=1e-6):
        raise ValueError("Early-terminated physics trace does not share original 20ms timeline")
    actual_nonzero=float(np.max(abs(physical_pose[pre_mask]-baseline_pose[:len(physical_pose)][pre_mask])))
    actual_mean=float(np.mean(abs(physical_pose[pre_mask]-baseline_pose[:len(physical_pose)][pre_mask])))
    # Compare measured PHYSICAL G1 segment geometry at the SAME original v32
    # robot impact-aligned timestamps to avoid posthoc shifting two outputs.
    human=json.loads((ROOT/CMU/"quality_audit.json").read_text())
    rel=np.arange(-.4,0,.02).round(8)
    raw_human,_=human_six_timed(clip,rel,human)
    reference_robot=load_robot_source(ROBOT)  # offline IK reference
    original_robot=load_robot_source(BASELINE)  # independent physics control
    if len(new["kinematics"][0])==700:
        candidate_robot=load_robot_source(output)
    else:
        # A physically failed early-terminated rollout is critical evidence,
        # not a missing-data error or permission to fabricate future frames.
        rows=new["kinematics"][0]
        candidate_robot=dict(times=np.asarray([r["time_s"] for r in rows],float),
            q=np.asarray([r["joint_pos"] for r in rows],float),
            root=np.asarray([r["root_pose"] for r in rows],float),
            joint_names=new["joint_names"],contact_time=shot["contact_time_s"])
    time=original_robot["contact_time"]+rel
    source_geo=numeric_six(reference_robot,time)
    baseline_geo=numeric_six(original_robot,time)
    candidate_geo=numeric_six(candidate_robot,time)
    gate=phase_window(rel)
    target=np.clip(source_geo+cfg["relative_gain"]*gate[:,None]*
                   np.clip(raw_human-raw_human[:1],-.30,.30),-1,1)
    mask=gate>.05
    pose_error_before=float(np.mean((baseline_geo[mask]-target[mask])**2))
    pose_error_after=float(np.mean((candidate_geo[mask]-target[mask])**2))
    # Re-run original whole-ball goal reporter independently on original 5ms trace.
    replay=ShotTracker(initial_root=new["initial_robot_state"]["root_position"][0],
        initial_ball=[*new["points"][0],.113],goal=new["goals"][0],
        forward=new["trials"][0]["goal_forward"],style=new["trials"][0]["style"],
        width=new["goal_width_m"],height=new["goal_height_m"])
    for row in new["traces"][0]:
        replay.update(row)
    replay_same=(replay.summary()==shot)
    old_speed=bstr["contact"]["outgoing_speed_xy_mps"] if bstr["contact"] else None
    new_speed=cstr["contact"]["outgoing_speed_xy_mps"] if cstr["contact"] else None
    ratio=None if not old_speed or not new_speed else float(new_speed/old_speed)
    gates=dict(
        complete_14s_original_PhysX=physics_ok,
        original_frozen_postshot_checkpoint_untouched=True,
        actual_new_reference_nonzero=(nonzero>0 and actual_nonzero>1e-7),
        actual_original_G1_joint_speed_within_limits=measurements["actual_speed_exceedance_sample_joint_count"]==0,
        correct_real_instep_and_no_illegal_ball_contact=(shot["measured_surface"]=="instep"
            and float(shot["requested_surface_impulse_fraction"])>=.95
            and new["sequence_records"][0]["illegal_body_ball_contact"] is False),
        whole_sphere_goal_and_original_task=(shot["goal"] is True
            and cstr["outcome"]["success"] is True and new["stride_shot_success"] is True),
        retained_outgoing_ball_speed=(ratio is not None and ratio>=.95),
        recovered_standing=(shot["recovered"] is True and shot["standing_seconds"]>=1.),
        original_running_and_continuous_contact=(
            new["running_evidence"]["running_success"] is True and
            new["stride_shot_evidence"]["continuous_running_contact"] is True),
        original_goal_replay_matches_recorded=replay_same,
        control_envelope_and_ball_state_intact=(len(anomalies)==0))
    result_doc=dict(
       version="0.1.52",clip=clip,physics_source=str(output.relative_to(ROOT)),
       physics_source_sha256=digest(output),
       original_physical_baseline=str(BASELINE.relative_to(ROOT)),
       original_baseline_sha256=digest(BASELINE),
       source_reference_sha256=cand["output_sha256"],
       original_full_14sec_5ms_steps=len(full),original_control_20ms_steps=len(feedback),
       true_PhysX_robot_ball_impact_s=float(shot["contact_time_s"]),
       human_true_foot_ball_impact_available=False,
       task_success_all_gates=all(gates.values()),gates=gates,
       nonzero_applied_reference_control_steps=nonzero,
       eligible_style_control_steps=eligible,
       actual_robot_joint_pose_peak_delta_from_base_rad=actual_nonzero,
       actual_robot_joint_pose_mean_abs_delta_from_base_rad=actual_mean,
       actual_applied_reference_peak_rad=float(np.max(abs(corrections))),
       actual_applied_reference_max_slew_rad=float(np.max(abs(np.diff(corrections,axis=0)))),
       real_5ms_joint_speed_limit_exceedance_samples=measurements["actual_speed_exceedance_sample_joint_count"],
       ball_xy_outgoing_speed_mps=new_speed,
       baseline_ball_xy_outgoing_speed_mps=old_speed,
       relative_ball_xy_speed_vs_baseline=ratio,
       first_true_robot_instep_impulse_Ns=float(shot["contact_impulse_by_surface_Ns"].get("instep",0)),
       baseline_instep_impulse_Ns=float(prior_shot["contact_impulse_by_surface_Ns"].get("instep",0)),
       standing_recovered_s=float(shot["standing_seconds"]),
       full_original_physics_termination=new["termination_reasons"],
       phase_target_6d_proxy_MSE_physics_baseline=pose_error_before,
       phase_target_6d_proxy_MSE_physics_candidate=pose_error_after,
       physically_realized_proxy_MSE_change=pose_error_after-pose_error_before,
       original_v32_offline_vs_replayed_actual_root_is_same=False,
       limitations=[
        "Two clips are the SAME human subject; neither verified foot-ball impact nor instep/ball spin/ground reaction force provided",
        "Optimization and physics compare six non-anatomically equivalent segment-angle cosines; proxy MSE is not human naturalness or professional kinetic-chain transfer",
        "Only one known development center-ball state, no randomized held-out ball positions, players, blinded aesthetics or statistical improvement",
        "Original v32 off-line IK root comes from v48 archived physical run; live runner reproduces its own separate v32 baseline",
        "Real measured qdot protection and task contact are independently audited; action joint reference slew alone never certifies physical safety",
        "Neither measured 3D optical ball launch nor original G1 force contact implies a matched true human-G1 collision time"],
       control_anomalies=sorted(set(anomalies)))
    result.write_text(json.dumps(result_doc,ensure_ascii=False,indent=2)+"\n")
    print("V052_STRICT_REAL_PHYSX",json.dumps({k:result_doc[k] for k in
       ("clip","task_success_all_gates","nonzero_applied_reference_control_steps",
        "actual_robot_joint_pose_peak_delta_from_base_rad",
        "real_5ms_joint_speed_limit_exceedance_samples",
        "ball_xy_outgoing_speed_mps","relative_ball_xy_speed_vs_baseline",
        "physically_realized_proxy_MSE_change")}),flush=True)
    if not result_doc["task_success_all_gates"]:
        print("V052_PHYSX_GATE_FAILURES", [k for k,v in gates.items() if not v],flush=True)
    return result_doc


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--clip",choices=["10_02","10_05"],required=True)
    a=parser.parse_args()
    evaluate(a.clip)
