#!/usr/bin/env python3
"""SHA-verified real v54 G1 200 Hz window; pure read-only CPU reanalysis.

1.570s is an old nominal strike anchor, NOT a measured collision manifold.
The 5.140s early terminal is not extended or filled. Net foot sensor magnitude
is never interpreted as ground-only signed GRF or CoP.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from v146_real_200hz_coupled_audit import (
    verified_sources,split_real_5ms_rows,verify_g1_metadata)
from v154_cpu_feasibility_campaign import qualified_history

EVENT_T=1.570
WINDOW=(1.170,2.820)
TERMINAL=(4.740,5.140)
SAMPLES=331


def analyze():
    sources=verified_sources()
    prior=qualified_history()
    result={}
    total_states=0
    for name,path in sources.items():
        raw=path.read_bytes()
        data=json.loads(raw)
        limits,torque=verify_g1_metadata(data)
        rows,terminal=split_real_5ms_rows(data["traces"][0],
            prior[name]["true_historical_200hz_rows"])
        t=np.array([r["time_s"] for r in rows])
        block=[r for r in rows if WINDOW[0]-1e-8<=r["time_s"]<=WINDOW[1]+1e-8]
        if len(block)!=SAMPLES or abs(block[0]["time_s"]-WINDOW[0])>1e-6 or abs(block[-1]["time_s"]-WINDOW[1])>1e-6:
            raise ValueError("V155_ACTUAL_200HZ_CONTACT_WINDOW_INCOMPLETE")
        pos=np.array([r["joint_pos"] for r in block])
        vel=np.array([r["joint_vel"] for r in block])
        tau=np.array([r["joint_torque_estimate"] for r in block])
        com=np.array([r["whole_body_com_xyz"] for r in block])
        root=np.array([r["root_xyz"] for r in block])
        ball=np.array([r["ball_xyz"] for r in block])
        net=np.array([r["nonfoot_contact_N"] for r in block])
        support=np.array([[r["support_foot_force_N"][x] for x in ("left","right")] for r in block])
        ratio=np.max(abs(vel)/np.asarray(limits))
        tau_ratio=np.max(abs(tau)/np.asarray(torque))
        if not all(np.isfinite(x).all() for x in (pos,vel,tau,com,root,ball,net,support)):
            raise ValueError("V155_NONFINITE_HISTORICAL_WINDOW")
        breakdown=[]
        for window_name,first,last in (("pre_nominal",1.170,1.565),
                                       ("post_nominal",1.570,2.820)):
            z=[r for r in block if first-1e-8<=r["time_s"]<=last+1e-8]
            breakdown.append({
                "name":window_name,"actual_sample_count":len(z),
                "time_first":z[0]["time_s"],"time_last":z[-1]["time_s"],
                "max_nonfoot_net_N":max(r["nonfoot_contact_N"] for r in z),
                "max_root_tilt_deg":max(r["root_tilt_deg"] for r in z)
            })
        d={"original_sha256":hashlib.sha256(raw).hexdigest(),
           "original_total_200hz_states":len(rows),
           "actual_window_states":len(block),
           "actual_window_first_s":block[0]["time_s"],
           "actual_window_last_s":block[-1]["time_s"],
           "actual_window_sample_period_s":float(np.median(np.diff([r["time_s"] for r in block]))),
           "max_abs_all29_qdot_limit_ratio":float(ratio),
           "max_abs_all29_effort_limit_ratio":float(tau_ratio),
           "com_z_min_m":float(com[:,2].min()),
           "com_z_max_m":float(com[:,2].max()),
           "max_root_tilt_deg":max(r["root_tilt_deg"] for r in block),
           "max_nonfoot_net_contact_N":float(net.max()),
           "max_left_right_foot_net_N":np.max(support,axis=0).tolist(),
           "foot_sensor_net_not_ground_grf_or_cop":True,
           "root_tilt_cannot_certify_foot_recovery":True,
           "pre_post":breakdown,
           "terminal_note_count":len(terminal),
           "human_collision_step_native_verified":False}
        if name=="cmu_10_05":
            last=[r for r in rows if TERMINAL[0]-1e-8<=r["time_s"]<=TERMINAL[1]+1e-8]
            if len(last)!=81 or rows[-1]["time_s"]!=5.14:
                raise ValueError("V155_TERMINAL_WINDOW_MISSING")
            d["actual_early_terminal_200hz_states"]=len(last)
            d["last_complete_full29_t_s"]=rows[-1]["time_s"]
            d["max_nonfoot_last_400ms_N"]=max(r["nonfoot_contact_N"] for r in last)
            d["named_sensor_5145ms_has_full29_state"]=False
        result[name]=d
        total_states+=len(rows)
    if total_states!=6628:
        raise ValueError("V155_ALL_HISTORY_SHA_STATES_DRIFT")
    return {"real_historical_200hz_states_verified":total_states,
            "nominal_strike_anchor_not_contact_truth_s":EVENT_T,
            "window":WINDOW,
            "three_runs_window_states":sum(x["actual_window_states"] for x in result.values()),
            "no_off_on_physics_parity_yet":True,
            "no_ground_unique_grf_or_cop":True,
            "native_left_wrist_left_hip_manifold":"UNKNOWN",
            "new_physics_runs":0,
            "runs":result}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    out=analyze()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:v for k,v in out.items() if k!="runs"},ensure_ascii=False))

if __name__=="__main__":
    main()
