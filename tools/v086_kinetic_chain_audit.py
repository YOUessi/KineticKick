#!/usr/bin/env python3
"""Blind-to-physics-outcome, retrospective 200 Hz KineticKick wholebody timing audit.

Independent policy controls only; contact-reader on/off pairs are NOT separate
subjects/resets. Purely observational kinematic velocity/torque proxies.
"""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np

CASES=("center_v32_off","center_cmu02_off","center_cmu05_off")
GROUPS={
"support_left":["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
                "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint"],
"swing_right":["right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
               "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"],
"trunk":["waist_yaw_joint","waist_roll_joint","waist_pitch_joint"],
"counter_arm_left":["left_shoulder_pitch_joint","left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint","left_elbow_joint",
                    "left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint"],
"right_arm":["right_shoulder_pitch_joint","right_shoulder_roll_joint",
             "right_shoulder_yaw_joint","right_elbow_joint",
             "right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]}
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def physical_steps(rows,dt=.005):
    if not rows:raise ValueError("EMPTY_TRACE")
    true=[]
    for row in rows:
        if not isinstance(row.get("time_s"),(float,int)):
            raise ValueError("MISSING_TIME")
        if not true:
            if abs(row["time_s"]-dt)>1e-6:raise ValueError("FIRST_TIME_NOT_REAL_200HZ")
            true.append(row);continue
        gap=float(row["time_s"])-float(true[-1]["time_s"])
        if abs(gap-dt)<1e-6:
            true.append(row)
        elif (len(rows)==len(true)+1 and bool(row.get("terminated")) and
              0<gap<dt*0.01 and
              row.get("joint_pos")==true[-1].get("joint_pos") and
              row.get("joint_vel")==true[-1].get("joint_vel") and
              row.get("whole_body_com_xyz")==true[-1].get("whole_body_com_xyz")):
            continue
        else:raise ValueError("NONREAL_TIME_STEP")
    return true
def inspect_episode(raw):
    names=raw["joint_names"]
    if len(names)!=29 or len(set(names))!=29:raise ValueError("INVALID_ORIGINAL_29_JOINTS")
    indices={g:[names.index(s) for s in expected] for g,expected in GROUPS.items()}
    trace=physical_steps(raw["traces"][0])
    t=np.asarray([r["time_s"] for r in trace])
    q=np.asarray([r["joint_pos"] for r in trace],dtype=float)
    dq=np.asarray([r["joint_vel"] for r in trace],dtype=float)
    tau=np.asarray([r["joint_torque_estimate"] for r in trace],dtype=float)
    com=np.asarray([r["whole_body_com_xyz"] for r in trace],dtype=float)
    fs=np.asarray([[r["support_foot_force_N"]["left"],r["support_foot_force_N"]["right"]] for r in trace],dtype=float)
    if any(a.shape!=(len(t),n) for a,n in ((q,29),(dq,29),(tau,29),(com,3),(fs,2))):
        raise ValueError("SHAPE_MISMATCH")
    if not all(np.isfinite(a).all() for a in (t,q,dq,tau,com,fs)):
        raise ValueError("NONFINITE_PHYSICAL_TRACE")
    contact=float(raw["records"][0]["contact_time_s"])
    window=(t>=contact-.4-1e-6)&(t<=contact+1.25+1e-6)
    if window.sum()!=331:raise ValueError("CONTACT_WINDOW_CENSORED_OR_MISSING")
    pre=(t>=contact-.4-1e-6)&(t<=contact+1e-6)
    post=(t>=contact-1e-6)&(t<=contact+1.25+1e-6)
    support=(fs>20.)
    groups={}
    for name,idx in indices.items():
        v=dq[pre][:,idx]
        magnitude=np.linalg.norm(v,axis=1)
        peak=int(np.argmax(magnitude))
        ts=t[pre]
        groups[name]={"rms_joint_speed_rad_s":float(np.sqrt(np.mean(v*v))),
                      "preimpact_speed_peak_time_relative_s":float(round(ts[peak]-contact,4)),
                      "preimpact_peak_group_speed_norm":float(magnitude[peak]),
                      "preimpact_abs_tau_dq_work_proxy_J":float(np.sum(np.abs(tau[pre][:,idx]*v))*.005),
                      "postimpact_rms_joint_speed_rad_s":float(np.sqrt(np.mean(dq[post][:,idx]**2)))}
    nonfoot=[r for r in trace if float(r.get("nonfoot_contact_N") or 0.)>=20.]
    summary={
        "true_physics_steps":len(trace),"source_rows":len(raw["traces"][0]),
        "terminal_annotation_rows":len(raw["traces"][0])-len(trace),
        "contact_s":contact,"complete_contact_pre0p4_post1p25":True,
        "episode_complete_14s":bool(abs(t[-1]-14.)<1e-5),
        "last_physics_s":float(t[-1]),
        "wholebody_com_z_at_contact":float(com[np.argmin(abs(t-contact)),2]),
        "support_preimpact_left_only_fraction":float(np.mean(support[pre,0]&~support[pre,1])),
        "support_preimpact_right_only_fraction":float(np.mean(~support[pre,0]&support[pre,1])),
        "support_preimpact_double_fraction":float(np.mean(support[pre].all(axis=1))),
        "support_preimpact_flight_fraction":float(np.mean(~support[pre].any(axis=1))),
        "measured_qdot_max_abs_rad_s":float(np.max(np.abs(dq))),
        "first_nonfoot_ge20_time_s":float(nonfoot[0]["time_s"]) if nonfoot else None,
        "peak_nonfoot_N":float(max(r.get("nonfoot_contact_N",0) or 0 for r in trace)),
        "groups":groups,
        "not_truth":["not a human/GRF/impulse label","not an anthropomorphic naturalness metric",
                     "joint torque is original recorder estimate, not motor calibrated power",
                     "no shape-level wrist hip clearance measured"]}
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v85-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("WRITE_ONCE_OUTPUT")
    prior=json.loads((a.v85_root/"reports/v085_g1_physics_final.audit.json").read_text())
    expected={row["name"]:row["raw_sha256"] for row in prior["runs"]}
    result={"source_v85_report_sha256":sha(a.v85_root/"reports/v085_g1_physics_final.audit.json"),
            "source_witnesses":{},"unique_policy_conditions":3,
            "same_seed_all_cases":True,"new_real_G1_executions":0,
            "this_is_retrospective_physics_not_new_experiment":True,
            "policy_conditions":{}}
    for case in CASES:
        src=a.v85_root/"reports"/("v085_"+case+".json")
        h=sha(src)
        if h!=expected[case]:raise RuntimeError("RAW_SOURCE_DRIFT:"+case)
        raw=json.loads(src.read_text())
        result["source_witnesses"][case]=h
        result["policy_conditions"][case]=inspect_episode(raw)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    for name,value in result["policy_conditions"].items():
        print("V086_COUPLING",name,"true",value["true_physics_steps"],
              "terminal_markers",value["terminal_annotation_rows"],
              "nonfoot",value["first_nonfoot_ge20_time_s"],
              "phasePeaks",{k:v["preimpact_speed_peak_time_relative_s"] for k,v in value["groups"].items()},flush=True)
if __name__=="__main__":main()
