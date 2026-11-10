#!/usr/bin/env python3
"""Independent strict original 29DoF PhysX gate audit; no false pair evidence.

The revised recorded matrix deliberately uses only the same previously
validated 200Hz trace and named sensor as v54. Native contact pair was NOT
obtained in three failed non-scored instrumentation pilots.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from run_pair_dynamics_batch_v055 import CONFIG,PROGRESS,ROOT,sha,freeze
from audit_cmu_physics_matrix_v053 import assess,wilson
from audit_named_nonfoot_diagnostics_v054 import COMPARE
OUTPUT=ROOT/"reports/v055_full_physics_dynamics_audit.json"

def measure_200hz(report):
    """Original G1 200Hz all-joint/root/COM/support/inter-segment dynamics."""
    rows=report["traces"][0]
    if len(rows)<20:raise ValueError("No actual original PhysX trajectory")
    time=np.asarray([row["time_s"] for row in rows],float)
    normal_dt=np.diff(time)
    # The original G1 evaluator appends a terminal sentinel at t+1us after
    # nonfoot early stop. It is NOT another 200Hz physics integration frame.
    early_stop = report["termination_reasons"] != [[]]
    integration_dt = normal_dt[:-1] if early_stop else normal_dt
    if (not np.all(np.isfinite(time)) or not np.all(normal_dt>0) or
        not np.all(np.abs(integration_dt-.005)<1e-5) or
        (early_stop and not (0 < normal_dt[-1] <= .00501))):
        raise ValueError("Original 200Hz cadence missing (except terminal sentinel)")
    q=np.asarray([row["joint_pos"] for row in rows],float)
    dq=np.asarray([row["joint_vel"] for row in rows],float)
    target=np.asarray([row["joint_target"] for row in rows],float)
    com=np.asarray([row["whole_body_com_xyz"] for row in rows],float)
    com_vel=np.asarray([row["whole_body_com_velocity"] for row in rows],float)
    if not (q.shape==dq.shape==target.shape==(len(rows),29) and
            com.shape==com_vel.shape==(len(rows),3)):
        raise ValueError("29DoF actual motion, target and 3D COM must all be recorded")
    if not all(np.isfinite(x).all() for x in (q,dq,target,com,com_vel)):
        raise ValueError("NaN/Inf in original real G1 motion")
    labels=report["joint_names"]
    lim=np.asarray(report["joint_velocity_limits"][0],float)
    if len(labels)!=29 or len(lim)!=29 or np.any(lim<=0):raise ValueError("Original G1 qdot contract missing")
    group={
      "upper_arm_and_waist":[i for i,n in enumerate(labels) if any(k in n for k in ("shoulder","elbow","wrist","waist"))],
      "lower_limb":[i for i,n in enumerate(labels) if any(k in n for k in ("hip","knee","ankle"))]
    }
    if len(group["upper_arm_and_waist"])+len(group["lower_limb"])!=29:raise ValueError("G1 29DoF body assignment incomplete")
    g=np.asarray([[row["support_foot_force_N"][s] for s in ("left","right")] for row in rows],float)
    if g.shape!=(len(rows),2) or np.min(g)<-1e-5:raise ValueError("Ground force matrix missing")
    t0=float(report["records"][0]["contact_time_s"])
    before=(time>=t0-.4)&(time<t0)
    after=(time>=t0)&(time<=t0+1.25)
    pre_complete=bool(before.any() and time[before][0]<=t0-.4+.011)
    post_complete=bool(after.any() and time[after][-1]>=t0+1.25-.011)
    # Never impute the 1.25s recovery window if original physical_nonfoot
    # terminated the episode. Retain as a counted, censored negative.
    qr=np.abs(dq)/lim
    peak_by_group={}
    for name,idx in group.items():
        # Arm/leg timings are measured robot-specific, not validated human kinesthetics.
        values=np.abs(dq[:,idx]);where=np.unravel_index(int(values.argmax()),values.shape)
        nearvalues=np.abs(dq[before|after][:,idx])
        near_time=time[before|after]
        nearwhere=np.unravel_index(int(nearvalues.argmax()),nearvalues.shape)
        peak_by_group[name]=dict(
          global_peak_joint=labels[idx[where[1]]],global_peak_abs_qdot_rad_s=float(values[where]),
          global_peak_s=float(time[where[0]]),
          near_impact_peak_joint=labels[idx[nearwhere[1]]],
          near_impact_peak_abs_qdot_rad_s=float(nearvalues[nearwhere]),
          near_impact_peak_relative_s=float(near_time[nearwhere[0]]-t0))
    phase_support={}
    for label,mask in (("pre_400ms",before),("post_1250ms",after)):
        force=g[mask];phase_support[label]=dict(
           mean_left_ground_normal_only_N=float(force[:,0].mean()),
           mean_right_ground_normal_only_N=float(force[:,1].mean()),
           left_only_fraction=float(((force[:,0]>80)&(force[:,1]<40)).mean()),
           right_only_fraction=float(((force[:,1]>80)&(force[:,0]<40)).mean()),
           double_support_fraction=float(((force[:,0]>80)&(force[:,1]>80)).mean()),
           both_unloaded_fraction=float(((force[:,0]<40)&(force[:,1]<40)).mean()))
    named=report.get("nonfoot_contact_diagnostics")
    if not isinstance(named,list):raise ValueError("Missing original read-only named sensor list")
    limitmax=np.unravel_index(np.argmax(qr),qr.shape)
    metric=dict(
       cadence_Hz=200,original_physics_samples=len(time),
       actual_200Hz_integrated_steps=(len(time)-1 if early_stop else len(time)),
       physical_terminal_sentinel_present=bool(early_stop),
       end_time_s=float(time[-1]),
       real_all_29dof_pose_velocity_target=True,
       contact_time_s=t0,preimpact_400ms_observed_samples=int(before.sum()),
       preimpact_400ms_complete=pre_complete,
       postimpact_1250ms_observed_samples=int(after.sum()),
       postimpact_1250ms_complete=post_complete,
       actual_postimpact_observed_until_s=(
           float(time[after][-1]-t0) if after.any() else None),
       early_physical_stop_recovery_window_censored=bool(early_stop and not post_complete),
       all_qdot_over_limit_sample_joint_count=int((qr>1+1e-6).sum()),
       max_actual_qdot_to_original_limit_ratio=float(qr.max()),
       worst_joint=labels[limitmax[1]],worst_joint_time_s=float(time[limitmax[0]]),
       com_height_preimpact_mean_m=float(com[before,2].mean()),
       com_height_postimpact_mean_m=float(com[after,2].mean()),
       com_speed_xy_preimpact_mean_mps=float(np.linalg.norm(com_vel[before,:2],axis=1).mean()),
       com_speed_xy_postimpact_mean_mps=float(np.linalg.norm(com_vel[after,:2],axis=1).mean()),
       peak_joint_group_timings=peak_by_group,
       support_dynamics=phase_support,
       nonfoot_named_events=named,nonfoot_max_original_sensor_N=float(max(x["nonfoot_contact_N"] for x in rows)),
       body_pair_manifold_verified=False,actual_collider_paths_available=False,
       wrist_hip_origin_distance_200Hz_available=False,
       full_original_14s_end=abs(float(time[-1])-14.)<1e-6 and report["termination_reasons"]==[[]],
       termination=report["termination_reasons"],
       actual_terminal_root_height_m=float(rows[-1]["root_xyz"][2]),
       actual_terminal_root_tilt_deg=float(rows[-1]["root_tilt_deg"]),
       measured_human_grf=False,measured_human_instep_contact=False,
       ground_normal_semantics="Original PhysX ankle-ground filtered force matrix normal only, not full ground reaction wrench or support polygon",
       causal_limit="No collider manifold, no friction-complete GRF or signed arm/hip mesh separation; cannot confirm self-collision or explain causality from phase proxies alone")
    return metric

def audit():
    if OUTPUT.exists():raise FileExistsError("Never overwrite primary experiment audit")
    c=json.loads(CONFIG.read_text());freeze(c)
    progress=json.loads(PROGRESS.read_text())
    if progress["protocol_sha256"]!=sha(CONFIG) or len(progress["runs"])!=len(c["cases"]):
        raise ValueError("Partial, reordered or modified frozen original PhysX run")
    if c["pair_manifold_attempt_status"]!="FAILED_PREPHYSICS_PILOT_NO_PAIR_CONTACT_TRUTH":
        raise ValueError("Cannot mislabel native collider evidence")
    v53=json.loads((ROOT/"configs/cmu_g1_physics_matrix_v053.json").read_text())
    rows=[]
    for case,original in zip(c["cases"],progress["runs"]):
        path=ROOT/case["out"]
        if original["id"]!=case["id"] or original["sha256"]!=sha(path):
            raise ValueError("Frozen original PhysX file SHA mismatch")
        d=json.loads(path.read_text())
        if d["seed"]!=case["seed"] or d["root_fixed"] or d["post_initialization_state_writes"]:
            raise ValueError("Original free-root physics, real seeded environment required")
        if d["joint_strike"]["ball_offset_xy"]!=case["ball_offset_xy_m"]:
            raise ValueError("Changed original ball position")
        if abs(d["joint_strike"]["velocity_y_delta"]-case["velocity_y_delta_mps"])>1e-8:
            raise ValueError("Changed initial velocity")
        if d["policy_sha256"]!=sha(ROOT/c["policies"][case["policy"]]):
            raise ValueError("Source config drift")
        spec=dict(ball_offset_xy_m=case["ball_offset_xy_m"],velocity_y_delta_mps=case["velocity_y_delta_mps"],outputs={case["policy"]:case["out"]})
        strict=assess(spec,case["policy"],v53)
        measured=measure_200hz(d)
        parity=None
        if "prior" in case:
            prev=json.loads((ROOT/case["prior"]).read_text())
            parity={field:d.get(field)==prev.get(field) for field in COMPARE}
        rows.append(dict(id=case["id"],policy=case["policy"],seed=case["seed"],
            original_physx_trace_sha256=sha(path),matched_ball_offset_xy_m=case["ball_offset_xy_m"],
            measured_root_velocity_delta_mps=case["velocity_y_delta_mps"],
            exact_physx_fields_equal_original_v54=parity,
            strict_task_and_all_safety_gates=strict,dynamics=measured))
    center=rows[:3];holdout=rows[3:]
    all_parity=all(x["exact_physx_fields_equal_original_v54"] and all(
        x["exact_physx_fields_equal_original_v54"].values()) for x in center)
    results={}
    for p in c["policies"]:
        vals=[r for r in holdout if r["policy"]==p]
        wins=sum(v["strict_task_and_all_safety_gates"]["success"] for v in vals)
        results[p]=dict(strict_successes=wins,n_real_holdout_runs=len(vals),rate=wins/len(vals),
           descriptive_wilson_95=wilson(wins,len(vals)),
           illegal_nonfoot_stops=sum("physical_nonfoot" in v["dynamics"]["termination"][0] for v in vals),
           joint_qdot_overspeed_conditions=sum(v["dynamics"]["all_qdot_over_limit_sample_joint_count"]>0 for v in vals),
           legal_instep_cases=sum(v["strict_task_and_all_safety_gates"]["gates"]["legal_right_instep_contact"] for v in vals),
           goal_cases=sum(v["strict_task_and_all_safety_gates"]["gates"]["entire_ball_goal"] for v in vals),
           full_recovery_conditions=sum(v["strict_task_and_all_safety_gates"]["gates"]["recovered_and_stable"] for v in vals))
    matrix_by_id={}
    for row in holdout:
        key=row["id"].rsplit("_",1)[0].replace("_cmu_10_02","").replace("_cmu_10_05","").replace("_frozen_v32","")
        if row["policy"]=="frozen_v32":base=row["id"].removesuffix("_frozen_v32")
        else:base=row["id"].removesuffix("_"+row["policy"])
        matrix_by_id.setdefault(base,{})[row["policy"]]=row["strict_task_and_all_safety_gates"]["success"]
    for p in ("cmu_10_02","cmu_10_05"):
        results[p]["baseline_fail_new_pass"]=sum(not v["frozen_v32"] and v[p] for v in matrix_by_id.values())
        results[p]["baseline_pass_new_fail"]=sum(v["frozen_v32"] and not v[p] for v in matrix_by_id.values())
    data=dict(version="0.1.55",protocol_sha256=sha(CONFIG),original_3x_parity=all_parity,
       n_new_real_physx_episodes=len(rows),n_center_repeat_diagnostics=len(center),n_new_holdout_matched_rollouts=len(holdout),
       n_holdout_conditions=4,all_sources_and_results_locked=True,none_of_the_new_scenarios_used_to_fit_policy=True,
       no_new_training_or_policy_promotion=True,
       original_simulator_collision_filter_or_termination_unchanged=True,
       physical_pair_manifold_collider_paths_not_proven=True,
       failed_native_pair_instrumentation_pilots_not_scored_as_real_trials=3,
       prior_original_PhySX_reanalysis=True,
       holdout_strict_summary=results,results=rows)
    OUTPUT.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    print("V055_STRICT_REAL_PHYSX_AUDIT",json.dumps(dict(
       n_new=data["n_new_real_physx_episodes"],parity=all_parity,
       pair_confirmed=False,summary=results)),flush=True)
    return data
if __name__=="__main__":audit()
