#!/usr/bin/env python3
"""Audit actual 200Hz G1 29DoF support, COM and phase timing without imputation.

First-contact time is a genuine G1 PhysX event. Human 120Hz time is
NOT a measured strike event and is never aligned as ground truth here.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from wholebody_safety_interface_v055 import G1_JOINT_ORDER
GROUPS={
    "stance_left_leg":("left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint"),
    "swing_right_leg":("right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"),
    "trunk":("waist_yaw_joint","waist_roll_joint","waist_pitch_joint"),
    "arms":tuple(n for n in G1_JOINT_ORDER if ("shoulder" in n or "elbow" in n or "wrist" in n)),
}
RIGHT_CHAIN=("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint")
def source_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()
def trimmed_200hz(raw):
    frames=raw["traces"][0]
    if len(frames)<32:raise ValueError("Too few original PhysX samples")
    if frames[-1].get("terminated") is True and 0<frames[-1]["time_s"]-frames[-2]["time_s"]<.003:
        frames=frames[:-1] # microsecond terminal marker is not a 200Hz step
    if len(raw["joint_names"])!=29 or tuple(raw["joint_names"])!=G1_JOINT_ORDER:
        raise ValueError("Not original G1 29DoF joint order")
    for f in frames:
        if len(f.get("joint_vel",[]))!=29 or len(f.get("joint_pos",[]))!=29:
            raise ValueError("Unobserved joint state")
    if any(not .0048<b["time_s"]-a["time_s"]<.0052 for a,b in zip(frames,frames[1:])):
        raise ValueError("Invalid 200Hz sample spacing")
    return frames
def summarize(raw):
    frames=trimmed_200hz(raw)
    contact=raw["records"][0].get("contact_time_s")
    if contact is None or not math.isfinite(contact):raise ValueError("Missing measured G1 foot-ball collision time")
    time=np.asarray([f["time_s"] for f in frames])
    qdot=np.asarray([f["joint_vel"] for f in frames],dtype=float)
    q=np.asarray([f["joint_pos"] for f in frames],dtype=float)
    limits=np.asarray(raw["joint_velocity_limits"][0],dtype=float)
    if qdot.shape[1]!=29 or limits.shape!=(29,) or not np.all(limits>0):
        raise ValueError("Original velocity limit missing")
    joint={name:i for i,name in enumerate(raw["joint_names"])}
    support_left=np.asarray([f["support_foot_force_N"]["left"] for f in frames])
    support_right=np.asarray([f["support_foot_force_N"]["right"] for f in frames])
    nonfoot=np.asarray([f["nonfoot_contact_N"] for f in frames])
    com=np.asarray([f["whole_body_com_xyz"] for f in frames])
    root=np.asarray([f["root_xyz"] for f in frames])
    if not all(np.isfinite(x).all() for x in [time,qdot,q,limits,support_left,support_right,nonfoot,com,root]):
        raise ValueError("Nonfinite observed physical states")
    idx=int(np.argmin(abs(time-contact)))
    if abs(time[idx]-contact)>.011:raise ValueError("Physical contact not sampled")
    masks={
        "before_0p40":(time>=contact-.4-1e-6)&(time<=contact),
        "after_1p25":(time>=contact)&(time<=contact+1.25+1e-6),
        "after_until_end":(time>=contact),
    }
    coverage={
        "before_0p40_complete":bool(time[0]<=contact-.4+1e-6),
        "after_1p25_complete":bool(time[-1]>=contact+1.25-1e-6),
        "recovery_to_14s_observed":bool(time[-1]>=13.999),
    }
    results={}
    for name,mask in masks.items():
        if not np.any(mask):raise ValueError("Missing physical time window")
        if name=="after_1p25" and not coverage["after_1p25_complete"]:
            # Do not pretend this censored window includes 1.25s of data.
            results[name]={"complete":False,"observed_to_s":float(time[-1]),
                           "reason":"early_physical_termination"}
            continue
        out={"complete":True,"sample_count":int(mask.sum()),
             "max_29dof_velocity_over_original_limit":float(np.max(abs(qdot[mask])/limits)),
             "max_nonfoot_net_contact_N":float(nonfoot[mask].max()),
             "left_support_force_min_N":float(support_left[mask].min()),
             "right_support_force_max_N":float(support_right[mask].max()),
             "com_z_min_m":float(com[mask,2].min()),
             "root_z_min_m":float(root[mask,2].min())}
        for group,names in GROUPS.items():
            inds=[joint[x] for x in names]
            out[group+"_rms_qdot_rad_s"]=float(np.sqrt(np.mean(qdot[mask][:,inds]**2)))
            out[group+"_peak_abs_qdot_rad_s"]=float(np.max(abs(qdot[mask][:,inds])))
        results[name]=out
    pre=masks["before_0p40"]; chain={}
    for n in RIGHT_CHAIN:
        i=joint[n]
        local=np.flatnonzero(pre)[np.argmax(abs(qdot[pre,i]))]
        chain[n]=dict(peak_time_relative_to_impact_s=float(time[local]-contact),
                      signed_velocity_at_peak_rad_s=float(qdot[local,i]))
    return dict(physx_200hz_rows=len(frames),contact_time_s=contact,
                end_s=float(time[-1]),coverage=coverage,
                joint_order_sha256=hashlib.sha256(json.dumps(raw["joint_names"]).encode()).hexdigest(),
                kinetic_chain_peaks=chain,
                stance_support_at_contact_N=dict(left=float(support_left[idx]),right=float(support_right[idx])),
                nonfoot_net_N_at_contact=float(nonfoot[idx]),
                root_com_vertical_offset_m_at_contact=float(root[idx,2]-com[idx,2]),
                windows=results,
                cannot_claim_cop_zmp_or_true_foot_grf=True,
                no_professional_motion_or_contact_truth_used=True)
def run(paths,output):
    if output.exists():raise FileExistsError(output)
    records=[]
    for p in paths:
        raw=json.loads(p.read_text())
        row=summarize(raw)
        row.update(raw_path=str(p),raw_sha256=source_hash(p),
                   policy_sha256=raw["policy_sha256"],strict_success=raw["stride_shot_success"],
                   termination_reasons=raw["termination_reasons"])
        records.append(row)
    with output.open("x") as f:json.dump(dict(version="v0.1.61",records=records),f,indent=2,ensure_ascii=False)
    return records
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=run(a.input,a.output)
    for x in r:print(x["raw_path"],"impact",x["contact_time_s"],"complete",x["coverage"],"strict",x["strict_success"])
