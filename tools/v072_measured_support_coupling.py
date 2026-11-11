#!/usr/bin/env python3
"""v72 independent postregistered 200Hz full-body phase and support forensic audit.

The v67 episodes have *already been seen*. Statistics describe measured PhysX
only; lagged kinematic covariation cannot prove human causal force transfer.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V67 = Path("/home/you/projects/kinetickick-wt-v067/reports")
LOCK = ROOT / "configs/g1_physx_readonly_evidence_ledger_v071.json"
DEST = ROOT / "reports/v072_measured_200hz_wholebody_coupling.json"
PARTS = {
    "plant_legs": [0,3,6,9,13,17],
    "swing_legs": [1,4,7,10,14,18],
    "pelvis_waist": [2,5,8],
    "left_arm": [11,15,19,21,23,25,27],
    "right_arm": [12,16,20,22,24,26,28],
}
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def safe_rms(data):
    a=np.asarray(data,dtype=float)
    if a.size==0 or not np.isfinite(a).all():
        raise ValueError("Missing/invalid true 200Hz physical values")
    return float(np.sqrt(np.mean(a*a)))

def lagged_signed_cov(a,b,lags=(-20,-10,0,10,20)):
    """100ms total lag grid in 200Hz samples, descriptive not causal."""
    x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float)
    if x.shape!=y.shape or len(x)<50: return None
    rows=[]
    for lag in lags:
        p=x[max(0,lag):len(x)+min(0,lag)]
        q=y[max(0,-lag):len(y)-max(0,lag)]
        if len(p)<30 or p.std()<1e-8 or q.std()<1e-8:
            rows.append(dict(lag_s=lag*.005,corr=None))
        else:
            rows.append(dict(lag_s=lag*.005,corr=float(np.corrcoef(p,q)[0,1])))
    return rows

def window_stats(rows, start, end, *, observation=None):
    subset=[(i,r) for i,r in enumerate(rows) if start-1e-8<=r["time_s"]<=end+1e-8]
    if not subset:return None
    idx,rec=zip(*subset)
    velocity=np.array([r["joint_vel"] for r in rec],dtype=float)
    if velocity.shape!=(len(rec),29): raise ValueError("No full 29D velocity")
    support=np.array([[r["support_foot_force_N"]["left"],r["support_foot_force_N"]["right"]] for r in rec],dtype=float)
    com=np.array([r["whole_body_com_xyz"] for r in rec],dtype=float)
    if com.shape!=(len(rec),3) or not np.isfinite(com).all():raise ValueError("No original PhysX CoM")
    total=support.sum(axis=1)
    res=dict(first_s=rec[0]["time_s"],last_s=rec[-1]["time_s"],actual_rows=len(rec),
             left_support_mean_N=float(support[:,0].mean()),right_support_mean_N=float(support[:,1].mean()),
             left_support_fraction_mean=float(np.mean(support[:,0]/np.maximum(total,1.e-6))),
             unsupported_frame_count=int(np.sum(total<10.0)),
             com_mean_xyz_m=np.mean(com,axis=0).tolist(),
             com_z_min_m=float(com[:,2].min()),
             nonfoot_peak_N=max(float(r["nonfoot_contact_N"]) for r in rec),
             measured_rms_qdot_rad_s={name:safe_rms(velocity[:,indices]) for name,indices in PARTS.items()},
             signed_velocity_lags_left_shoulder_vs_waist_yaw=lagged_signed_cov(velocity[:,11],velocity[:,2]))
    if observation:
        gaps=[observation[i]["left_wrist_hip_link_origin_distance_m"] for i in idx]
        res["minimum_wrist_hip_origin_gap_m"]=float(min(gaps))
        res["gap_is_not_mesh_distance"]=True
    else:res["minimum_wrist_hip_origin_gap_m"]=None
    return res

def audit_one(name, expected):
    source=V67/name
    if SHA(source)!=expected:raise RuntimeError("SOURCE_DRIFT "+name)
    raw=json.loads(source.read_text())
    names=raw.get("joint_names",[])
    if len(names)!=29 or len(set(names))!=29:raise ValueError("No original G1 29DOF")
    rows=raw["traces"][0]
    if not rows:raise ValueError("No physical frames")
    t=np.array([r["time_s"] for r in rows],dtype=float)
    # Original early-terminated G1 recordings include one terminal sentinel at
    # last time + 1us. Retain that real output row, but never call it a 5ms
    # independent step or pretend a censored 14s recovery completed.
    diffs=np.diff(t)
    abnormal=np.where(np.abs(diffs-.005)>=1.0e-4)[0]
    terminal_sentinel=(len(abnormal)==1 and abnormal[0]==len(diffs)-1
                       and 0<diffs[-1]<=1e-5)
    if len(abnormal) and not terminal_sentinel:raise ValueError("Broken original 200Hz clock")
    qdot=np.asarray([r["joint_vel"] for r in rows],dtype=float)
    if qdot.shape!=(len(rows),29) or not np.isfinite(qdot).all():raise ValueError("Original joint velocity lost")
    limits=np.asarray(raw["joint_velocity_limits"],dtype=float)
    if limits.shape!=(1,29) or np.any(limits<=0):raise ValueError("Original 29DoF speed limits lost")
    ratio=np.abs(qdot)/limits
    worst_index=np.unravel_index(np.argmax(ratio),ratio.shape)
    obs=raw.get("safe_body_v062")
    observation=None
    if obs and obs.get("mode")=="safe_full" and len(obs["rows"])==len(rows):
        observation=obs["rows"]
        for i,(a,b) in enumerate(zip(rows,observation)):
            if abs(a["time_s"]-b["state_aligned_time_s"])>1e-6 or b["state_alignment_source_index"]!=i:
                raise ValueError("Clock offset / observer data not aligned")
    rec=raw["records"][0]
    contact=rec.get("contact_time_s")
    result=dict(filename=name,original_sha256=expected,backend_requested=raw.get("v057_requested_physics_backend"),
        original_physics_rows=len(rows),terminal_extra_sample_1us=bool(terminal_sentinel),
        unique_physical_200hz_steps=len(rows)-int(terminal_sentinel),actual_end_s=float(t[-1]),
        original_29_joint_qdot_limit_max_ratio=float(ratio.max()),
        worst_joint=names[worst_index[1]],worst_qdot_t_s=float(t[worst_index[0]]),
        contact_real_s=contact,missing_contact_not_imputed=contact is None,
        observed_14s=bool(t[-1]>=14.-1e-3),physical_nonfoot_early_censored=bool(t[-1]<14.-1e-3),
        original_record_goal=rec.get("goal"),original_record_inst_fraction=rec.get("requested_surface_impulse_fraction"),
        attribution="POSTREGISTERED_SEEN_NOT_HOLDOUT",original_contact_pair_directly_certified=False)
    if contact is not None:
        start=contact-.4;end=contact+1.25
        result["pre_impact"]=window_stats(rows,start,contact,observation=observation)
        result["post_impact_1p25"]=window_stats(rows,contact,end,observation=observation)
        result["complete_required_window"]=bool(t[0]<=start and t[-1]>=end-1e-4)
        if result["complete_required_window"] and result["post_impact_1p25"] is None:
            raise ValueError("Window contradiction")
    else:
        result.update(pre_impact=None,post_impact_1p25=None,complete_required_window=False)
    return result

def run():
    lock=json.loads(LOCK.read_text())
    cases=[audit_one(n,h) for n,h in sorted(lock["v67_raw_sha256"].items())]
    valid=0
    for c in cases:
        if c["observed_14s"] and c["original_record_goal"] and (
            c["original_record_inst_fraction"] is not None and c["original_record_inst_fraction"]>=.95
        ) and c["original_29_joint_qdot_limit_max_ratio"]<=1:
            valid+=1
    data=dict(version="v0.1.72",classification="postregistered nine original CUDA G1 episodes, not a novel experiment",
              source="Frozen v67 original 200Hz PhysX; no new G1 trials",episodes_original=9,
              episodes_new=0,strict_partial_gate_count_not_full_qualification=valid,
              no_hip_wrist_true_pair_certified=True,
              full_standing_only_when_original_verified=True,
              correlation_not_causation=True,
              these_recordings_are_not_professional_human_biomechanics=True,
              support_force_is_original_foot_net_not_CoP_or_polygon=True,
              cases=cases)
    with DEST.open("x") as f:json.dump(data,f,indent=2,ensure_ascii=False);f.write("\n")
    print("V072_HISTORICAL",len(cases),"ROWS",sum(c["original_physics_rows"] for c in cases),
          "COMPLETE_14S",sum(c["observed_14s"] for c in cases),
          "CONTACT_WINDOWS",sum(c["complete_required_window"] for c in cases),
          "PARTIAL_GATE_COUNT",valid,"SHA",SHA(DEST))
    for c in cases:print(c["filename"],c["actual_end_s"],c["contact_real_s"],
                       "pre_left_share",round(c["pre_impact"]["left_support_fraction_mean"],3) if c["pre_impact"] else "NA",
                       "pre_waist_rms",round(c["pre_impact"]["measured_rms_qdot_rad_s"]["pelvis_waist"],3) if c["pre_impact"] else "NA")
    return data
if __name__=="__main__":run()
