#!/usr/bin/env python3
"""Independent all-29DoF/supported COM read-only reanalysis of frozen v54 source.

Uses ONLY original complete 200Hz physical rollout time series and v54 named
sensor events. This file never substitutes a link-origin gap for a manifold.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCES={name:ROOT/("reports/v054_"+name+"_contact_diagnostics.json") for name in
         ("frozen_v32","cmu_10_02","cmu_10_05")}
OUT=ROOT/"reports/v055_existing_200hz_coupled_analysis.json"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def analyze_one(src):
    d=json.loads(src.read_text())
    rows=d["traces"][0];names=d["joint_names"];limit=np.array(d["joint_velocity_limits"][0])
    if len(names)!=29 or len(limit)!=29:raise ValueError("Original 29DoF joint order required")
    t=np.array([x["time_s"] for x in rows])
    q=np.array([x["joint_pos"] for x in rows]);dq=np.array([x["joint_vel"] for x in rows])
    com=np.array([x["whole_body_com_xyz"] for x in rows])
    support=np.array([[x["support_foot_force_N"][side] for side in ("left","right")] for x in rows])
    normal=np.array([x["nonfoot_contact_N"] for x in rows])
    if q.shape!=(len(t),29) or dq.shape!=q.shape or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError("Corrupted 29D 200Hz original physical telemetry")
    dt=np.diff(t)
    valid=(dt>=.00499)&(dt<=.00501)
    if valid.mean()<.998:raise ValueError("Physical step coverage insufficient")
    impact=float(d["records"][0]["contact_time_s"])
    window=(t>=impact-.4)&(t<=impact+1.25)
    if t[window][0]>(impact-.4)+.0051 or t[window][-1]<(impact+1.25)-.0051:
        raise ValueError("Missing pre .40s / post 1.25s complete physical evidence")
    ratio=np.abs(dq)/limit
    worst=np.unravel_index(np.argmax(ratio),ratio.shape)
    termination=d["termination_reasons"]
    named=d.get("nonfoot_contact_diagnostics",[])
    arm_cols=[i for i,n in enumerate(names) if any(k in n for k in ("shoulder","elbow","wrist","waist"))]
    leg_cols=[i for i,n in enumerate(names) if any(k in n for k in ("hip","knee","ankle"))]
    if len(arm_cols)+len(leg_cols)!=29:raise ValueError("Unrecognized G1 joint taxonomy")
    left_frac=float(np.mean((support[:,0]>80)&(support[:,1]<40)))
    right_frac=float(np.mean((support[:,1]>80)&(support[:,0]<40)))
    double_frac=float(np.mean((support[:,0]>80)&(support[:,1]>80)))
    flight_frac=float(np.mean((support[:,0]<40)&(support[:,1]<40)))
    return dict(source=str(src.relative_to(ROOT)),source_sha256=sha(src),
       original_real_physx=True,source_sampling_hz=200,full_29dof_actual_qdot=True,
       physical_frames=len(t),original_time_s=float(t[-1]),
       impact_s=impact,
       contact_before_s=.4,contact_after_s=1.25,actual_impact_window_frames=int(window.sum()),
       true_ground_filtered_normal_only_support_forces_N=dict(
         min_left=float(support[:,0].min()),max_left=float(support[:,0].max()),
         min_right=float(support[:,1].min()),max_right=float(support[:,1].max())),
       support_state_fraction_80N=dict(left_only=left_frac,right_only=right_frac,
            both=double_frac,below_40N_both=flight_frac),
       near_contact_support_left_mean_N=float(support[window,0].mean()),
       near_contact_support_right_mean_N=float(support[window,1].mean()),
       max_overall_qdot_to_original_limit=float(np.max(ratio)),
       actual_qdot_overspeed_5ms_joint_samples=int(np.sum(ratio>1+1e-6)),
       worst_joint=names[worst[1]],worst_joint_time_s=float(t[worst[0]]),
       arm_waist_peak_qdot_rad_s=float(np.abs(dq[:,arm_cols]).max()),
       hips_knees_ankles_peak_qdot_rad_s=float(np.abs(dq[:,leg_cols]).max()),
       com_z_impact_m=float(np.interp(impact,t,com[:,2])),
       com_z_final_m=float(com[-1,2]),
       com_velocity_xy_impact_proxy_m_s=float(np.linalg.norm(
           np.gradient(com[:,:2],t,axis=0)[np.argmin(np.abs(t-impact))])),
       max_real_nonfoot_net_force_N=float(normal.max()),
       original_terminal_root_height_m=float(rows[-1]["root_xyz"][2]),
       original_terminal_tilt_deg=float(rows[-1]["root_tilt_deg"]),
       original_last_ground_filtered_support_normal_N=rows[-1]["support_foot_force_N"],
       named_nonfoot_sensor_events=named,
       physical_nonfoot_stop=("physical_nonfoot" in termination[0]),
       full_14s=(len(t)>=2800 and abs(t[-1]-14)<1e-5 and termination==[[]]),
       strict_instep_fraction=float(d["records"][0]["requested_surface_impulse_fraction"]),
       legal_instep=bool(d["records"][0]["measured_surface"]=="instep" and
                        d["records"][0]["requested_surface_impulse_fraction"]>=.95),
       goal=bool(d["records"][0]["goal"]),
       recovered=bool(d["records"][0]["recovered"]),
       trajectory_29dof_truth_not_human=True,
       note="Ground filtered foot force matrix is normal-only estimate, not friction-complete GRF; no precise human impact time or collider pair in v54.")

def analyze():
    if OUT.exists():raise FileExistsError("Do not overwrite original reanalysis")
    rows={k:analyze_one(p) for k,p in SOURCES.items()}
    data=dict(version="0.1.55",prior_original_PhysX_runs=3,new_PhySX_rollouts_in_this_analysis=0,
              these_are_diagnostic_development_center_only=True,results=rows)
    OUT.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    print("V055_PRIOR_ORIGINAL_DYNAMICS",json.dumps({k:dict(
        frames=x["physical_frames"],end=x["original_time_s"],
        over_limit=x["actual_qdot_overspeed_5ms_joint_samples"],
        nonfoot=x["physical_nonfoot_stop"],root_z=x["original_terminal_root_height_m"],
        tilt=x["original_terminal_tilt_deg"]) for k,x in rows.items()}),flush=True)
    return data
if __name__=="__main__":analyze()
