#!/usr/bin/env python3
"""v43: physically measured kick-event aligned G1 whole-body safety evidence.

NO training or control-system mutation. Uses ONLY original v42 paired actual
PhysX logs, pinned by SHA256, and emits a tiny event-aligned dataset for
scientific study. Real 5-ms qdot measurements matter; pre-impact constraints
cannot guarantee delayed landing joint speed. Do not use source trial pairs as
an independent held-out dataset once results are inspected.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/contact_event_safety_v043.json"
PAIRED="reports/v042_paired_novel_ball_audit.json"

def sha(name):
    h=hashlib.sha256()
    with (ROOT/name).open("rb") as s:
        for b in iter(lambda:s.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def indices(raw,impact,relative):
    t=np.asarray([x["time_s"] for x in raw],dtype=float)
    expected=impact+relative
    wanted=np.searchsorted(t,expected,side="left")
    idx=np.clip(wanted,0,len(t)-1)
    close=np.clip(idx-1,0,len(t)-1)
    use_previous=(abs(t[close]-expected)<abs(t[idx]-expected))
    idx=np.where(use_previous,close,idx)
    if np.max(abs(t[idx]-expected))>5e-7 or not np.all(np.diff(idx)==1):
        raise ValueError("Not truly contiguous 5-ms original PhysX episode near contact")
    return idx

def one(item,kind,relative):
    path=item[kind]["path"]
    if sha(path)!=item[kind]["sha256"]:
        raise ValueError("Original recorded v42 physical evidence SHA changed: "+path)
    r=json.loads((ROOT/path).read_text())
    contact=r["records"][0]["contact_time_s"]
    if contact is None:raise ValueError("Cannot align an unverified foot-ball contact")
    raw=r["traces"][0]
    if len(raw)!=2800 or r["post_initialization_state_writes"] is not False:
        raise ValueError("Only uninterrupted original 14s physical episodes")
    assert all(abs(float(x["physics_dt_s"])-.005)<1e-9 for x in raw)
    idx=indices(raw,float(contact),relative)
    samples=[raw[i] for i in idx]
    names=r["joint_names"]
    if len(names)!=29:raise ValueError("Original G1 joint order")
    def field(k,shape):
        v=np.asarray([s[k] for s in samples],dtype=np.float64)
        if v.shape!=shape or not np.all(np.isfinite(v)):
            raise ValueError("Invalid original 5-ms field "+k)
        return v
    length=len(samples)
    q=field("joint_pos",(length,29))
    v=field("joint_vel",(length,29))
    target=field("joint_target",(length,29))
    torque=field("joint_torque_estimate",(length,29))
    foot_speed=field("right_foot_point_velocity",(length,3))
    ball_v=field("ball_velocity",(length,3))
    com_v=field("whole_body_com_velocity",(length,3))
    com_pos=field("whole_body_com_xyz",(length,3))
    root=field("root_xyz",(length,3))
    forces=np.asarray([[float(s["support_foot_force_N"]["left"]),float(s["support_foot_force_N"]["right"])]
                         for s in samples],dtype=np.float64)
    if forces.shape!=(length,2) or not np.all(np.isfinite(forces)):raise ValueError("Missing foot load")
    nonfoot=np.asarray([s["nonfoot_contact_N"] for s in samples],float)
    if np.any(nonfoot<0) or not np.isfinite(nonfoot).all():raise ValueError("Invalid nonfoot")
    limits=np.asarray(r["joint_velocity_limits"][0],float)
    if limits.shape!=(29,) or np.min(limits)<=0:raise ValueError("Original qdot limits invalid")
    ratio=np.abs(v)/limits[None]
    signed_power=v*torque
    pre=relative<0
    after=(relative>=0)&(relative<=1.0+1e-9)
    actual_min=float((1.-ratio[pre].max()))
    actual_post=float(ratio[after].max())
    pre_above_09=bool((ratio[pre]>.9).any())
    post_actual_over=bool((ratio[after]>1.+1e-6).any())
    which=np.where((ratio[after]>1.+1e-6).any(axis=1))[0]
    first_overspeed_relative=None
    first_overspeed_joint=None
    if len(which)>0:
        relative_sub=relative[after]
        first_overspeed_relative=float(relative_sub[which[0]])
        bad_step=np.where(after)[0][which[0]]
        first_overspeed_joint=names[int(np.argmax(ratio[bad_step]))]
    # Full original safety gate scans ALL 2800 physical steps, not only this
    # contact-aligned 1.25-second research window.
    from audit_p0_dynamic_strike import limit_observations
    from strike_baseline_metrics import validated_episode
    full=limit_observations(validated_episode(r),r)
    data=dict(q=q,v=v,torque=torque,target=target,foot_speed=foot_speed,ball_v=ball_v,
              com_v=com_v,com_pos=com_pos,root=root,support=forces,nonfoot=nonfoot,
              qdot_ratio=ratio,power=signed_power)
    summary=dict(
       source_path=path,source_sha256=sha(path),
       observed_source_is_true_original_PhysX=True,
       original_clipped_PD_torque_not_real_hardware=True,
       first_physx_foot_ball_contact_time_s=float(contact),
       signed_ball_velocity_at_impact=ball_v[np.argmin(abs(relative))].tolist(),
       precontact_qdot_max_ratio=float(ratio[pre].max()),
       contact_relative_0to1s_qdot_max_ratio=actual_post,
       had_precontact_90_percent_limit=pre_above_09,
       violated_qdot_after_contact_1s=post_actual_over,
       pre_under_90_but_post_over_100=(not pre_above_09) and post_actual_over,
       first_overspeed_relative_s=first_overspeed_relative,
       first_overspeed_joint=first_overspeed_joint,
       full_episode_actual_qdot_limit_exceedances=int(full["actual_speed_exceedance_sample_joint_count"]),
       full_episode_worst_qdot_ratio=float(full["maximum_actual_speed_ratio"]),
       full_episode_worst_joint=full["worst_actual_speed"]["joint"],
       full_episode_worst_t=full["worst_actual_speed"]["time_s"],
       whole_body_positive_clipped_PD_work_pre_J=float(np.maximum(signed_power[pre],0).sum(axis=1).sum()*.005),
       whole_body_positive_clipped_PD_work_post1s_J=float(np.maximum(signed_power[after],0).sum(axis=1).sum()*.005),
       maximum_stance_left_load_after_contact_N=float(forces[after,0].max()),
       maximum_swing_right_load_after_contact_N=float(forces[after,1].max()),
       maximum_nonfoot_contact_N=float(nonfoot.max()),
       strict_functional_pass=item[kind]["strict_functional_pass"],
       legal_instep=item[kind]["original_physx_gates"]["legal_instep"],
       shot_goal=item[kind]["goal"],
       first_impact_world_t_minus_theoretical_clock_s=None,
       note="qdot is PHYSX actually measured, not just a generated joint position reference."
    )
    return data,summary,names,limits

def build():
    config=json.loads((ROOT/STUDY).read_text())
    if config["version"]!="0.1.43" or config["no_new_physics"] is not True or config["precontact_warn_ratio"]!=.9:
        raise ValueError("Wrong read-only study or originally predeclared risk threshold")
    for name,dig in config["source_sha256"].items():
        if sha(name)!=dig:raise ValueError("Previously certified physics/source SHA drift: "+name)
    paired=json.loads((ROOT/PAIRED).read_text())
    if len(paired["trials"])!=6 or paired["protocol_sha256"]!=config["previous_v42_protocol_sha256"]:
        raise ValueError("Wrong original v42 physical conditions")
    rel=np.linspace(config["window_start_s"],config["window_end_s"],
                    int(round((config["window_end_s"]-config["window_start_s"])/.005))+1)
    if len(rel)!=251 or not np.allclose(np.diff(rel),.005,atol=1e-11):
        raise ValueError("Wrong original PhysX event window")
    all_data={}
    output=[]
    labels=None
    physical_speed_limits=None
    for trial in paired["trials"]:
        for kind in ("original_v32","style_v41"):
            metrics,metadata,names,limits=one(trial,kind,rel)
            if labels is None:
                labels=names
                physical_speed_limits=limits
            if labels!=names or not np.array_equal(physical_speed_limits,limits):
                raise ValueError("G1 joint semantics or original physical bounds changed")
            for name,arr in metrics.items():
                all_data.setdefault(name,[]).append(arr.astype(np.float32))
            output.append(dict(id=trial["id"],controller=kind,**metadata))
    if len(output)!=12:raise ValueError("Not complete 12-case original paired physics")
    source_dir=ROOT/config["output_directory"]
    source_dir.mkdir(exist_ok=True,parents=True)
    target=source_dir/"event_aligned_qdot_dynamics.npz"
    index=source_dir/"event_aligned_audit.json"
    if target.exists() or index.exists():raise FileExistsError("Read-only dataset cannot overwrite")
    payload={k:np.stack(v) for k,v in all_data.items()}
    if any(v.shape[0]!=12 or v.shape[1]!=251 for v in payload.values()):
        raise ValueError("Inconsistent actual sampling alignment")
    np.savez_compressed(target,**payload,
      relative_contact_s=rel.astype(np.float32),
      joint_names=np.asarray(labels),original_joint_velocity_limits_rad_s=physical_speed_limits,
      source_case=np.asarray([x["id"] for x in output]),
      controller=np.asarray([x["controller"] for x in output]),
      verified_real_g1_hardware=np.asarray(False),
      exact_original_physx_5ms_samples=np.asarray(True))
    brief=dict(version="0.1.43",schema=1,
      original_v42_protocol_sha256=config["previous_v42_protocol_sha256"],
      v43_study_sha256=sha(STUDY),output_sha256=sha(config["output_directory"]+"/event_aligned_qdot_dynamics.npz"),
      all_source_episodes_verified_by_sha=True,
      phase_time_zero="original PhysX FIRST real foot-ball contact from original shot record, not 3DSP shot annotation",
      full_episode_qdot_certified_by_original_v42_audit=True,
      physical_hardware_status="simulation only",
      source_robot_count=1,source_human_motion_ground_truth=False,
      labels_100percent_safe=False,
      dt_s=.005,contact_relative_window_s=list(config["window_start_end_s"]),
      sample_count_per_episode=251,episode_count=12,
      foot_plant_joint_power_definition="measured IsaacLab qdot*estimated clipped PD torque; not real motor/human mechanics",
      risk_threshold_pre_safety_ratio=config["precontact_warn_ratio"],
      events=output,
      signals=sorted(payload),
      limitations=[
        "These 12 already inspected and audited episodes are DEVELOPMENT evidence only. Do not treat a v43 control algorithm validated on them as new held-out generalization.",
        "Only 6 distinct ball initial states and two controllers, one deterministic physical run each; event bins within one run are highly correlated.",
        "Only 1.25s around contact, not a general natural gait dataset",
        "Human biomechanics and joint impulse transfer are not calibrated; G1 internal PD torque estimate is not hardware-rated power",
        "Physical support load and post-contact qdot can change nonlinearly even after 0.006rad/20ms small reference commands",
        "Cannot certify a new control policy by offline event alignment or a pre-contact threshold alone"
      ])
    index.write_text(json.dumps(brief,ensure_ascii=False,indent=2)+"\n")
    print("V043_EVENT_SAFETY_DATA_BUILT",json.dumps(dict(
       episodes=len(output),time_bins=len(rel),signals=sorted(payload),
       output_sha=brief["output_sha256"],
       pre_safe_post_unsafe=[(x["id"],x["controller"]) for x in output if x["pre_under_90_but_post_over_100"]],
       post_joint_violations=[(x["id"],x["controller"],x["first_overspeed_joint"],x["first_overspeed_relative_s"])
                             for x in output if x["violated_qdot_after_contact_1s"]]),ensure_ascii=False),flush=True)
    return brief

if __name__=="__main__":build()
