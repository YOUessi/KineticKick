#!/usr/bin/env python3
"""Independent SHA-pinned original 200Hz G1 coupling / censoring, no physics writes."""
from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from v080_native_poll_batch import CFG,CFG_SHA,sha

def audit():
    config=json.loads(CFG.read_text())
    if sha(CFG)!=CFG_SHA:raise RuntimeError("PROTOCOL_SHA_DRIFT")
    output=[]
    for trial in config["trial_order"]:
        for mode in config["mode_order"]:
            stem=f"v080_original_g1_poll_pair_{trial['case']}_{mode}"
            rawpath=ROOT/"reports"/(stem+".json")
            auditpath=ROOT/"reports"/(stem+".audit.json")
            logpath=ROOT/"reports"/(stem+".log")
            side=json.loads(auditpath.read_text())
            if sha(rawpath)!=side["json_sha256"] or sha(logpath)!=side["log_sha256"]:
                raise RuntimeError("RAW_OR_LOG_SHA_DRIFT:"+stem)
            raw=json.loads(rawpath.read_text())
            if not side["certified_original_200hz_G1"]:raise RuntimeError("NOT_ORIGINAL_G1:"+stem)
            rows=raw["traces"][0]
            T=np.array([r["time_s"] for r in rows],dtype=float)
            Q=np.array([r["joint_vel"] for r in rows],dtype=float)
            limits=np.array(raw["joint_velocity_limits"],dtype=float).reshape(-1)
            if Q.shape!=(len(rows),29) or limits.shape!=(29,) or np.any(limits<=0):
                raise ValueError("ORIGINAL_29QDOT_NOT_PRESENT")
            COM=np.array([r["whole_body_com_xyz"] for r in rows],dtype=float)
            feet=np.array([[r["support_foot_force_N"]["left"],
                            r["support_foot_force_N"]["right"]] for r in rows],dtype=float)
            NF=np.array([r["nonfoot_contact_N"] for r in rows],dtype=float)
            if not all(np.isfinite(x).all() for x in (T,Q,COM,feet,NF,limits)):
                raise ValueError("NONFINITE_200HZ")
            hit=raw["records"][0].get("contact_time_s")
            start=T.min();end=T.max()
            pre=bool(hit is not None and hit-start>=0.4-0.005)
            post=bool(hit is not None and end-hit>=1.25-0.005)
            wpre=(T>=hit-0.4)&(T<=hit) if hit is not None else np.zeros(len(T),dtype=bool)
            wpost=(T>=hit)&(T<=hit+1.25) if hit is not None else np.zeros(len(T),dtype=bool)
            measured_force=[]
            for e in raw.get("nonfoot_contact_diagnostics",[]):
                if not isinstance(e,dict):continue
                f=e.get("force_w_N",[])
                if len(f)==3 and all(math.isfinite(v) for v in f):
                    measured_force.append((e["time_s"],e["body"],math.sqrt(sum(v*v for v in f))))
            high=sorted(measured_force,key=lambda x:x[2],reverse=True)[:8]
            overall=float(np.max(np.abs(Q)/limits))
            row=dict(case=trial["case"],mode=mode,
                actual_raw_json_sha256=sha(rawpath),actual_log_sha256=sha(logpath),
                original_200hz_physics_frames=len(rows),
                time_start_s=float(start),time_end_s=float(end),
                true_g1_14s_observed=bool(end>=13.995),
                real_ball_contact_s=hit,pre_contact_0p4s_observed=pre,
                post_contact_1p25s_observed=post,
                original_29_qdot_over_limit_max_ratio=overall,
                measured_original_29_qdot_all_frames=True,
                pre_contact_qdot_ratio=float(np.max(np.abs(Q[wpre])/limits)) if np.any(wpre) else None,
                post_contact_qdot_ratio=float(np.max(np.abs(Q[wpost])/limits)) if np.any(wpost) else None,
                com_height_range_m=[float(COM[:,2].min()),float(COM[:,2].max())],
                mean_left_support_sensor_N=float(feet[:,0].mean()),
                mean_right_support_sensor_N=float(feet[:,1].mean()),
                both_feet_supported_fraction=float(np.mean((feet[:,0]>10)&(feet[:,1]>10))),
                no_foot_load_fraction=float(np.mean((feet[:,0]<=10)&(feet[:,1]<=10))),
                peak_nonfoot_net_N=float(NF.max()),
                named_nonfoot_net_force_top_8=[{"time_s":t,"body":n,"norm_N":v} for t,n,v in high],
                actual_nonfoot_termination="physical_nonfoot" in str(raw.get("termination_reasons",[])),
                strict_hard_gate_pass=side["strict_hard_gates"]["hard_gate_pass"],
                strict_failed_gates=side["strict_hard_gates"].get("failed_gates"),
                original_29dof_first_to_last_traces_are_measured=True,
                native_poll=side["native_poll_summary"],
                human_ground_truth_GRF_or_professional_actor=False,
                torso_arm_lag_not_causal_power_attribution=True,
                body_origin_distance_not_collider_mesh_gap=True)
            output.append(row)
    summary=dict(version="v0.1.80",frozen_protocol_sha256=CFG_SHA,
        genuinely_measured_gpu_original_g1=len(output),
        physics_episode_full_14s=sum(x["true_g1_14s_observed"] for x in output),
        physics_episode_early_censored=sum(not x["true_g1_14s_observed"] for x in output),
        elligible_real_contact_full_window=sum(x["pre_contact_0p4s_observed"] and
                          x["post_contact_1p25s_observed"] for x in output),
        strictly_hard_gates_pass=sum(x["strict_hard_gate_pass"] for x in output),
        native_complete_wrist_hip_witness=sum(x["native_poll"].get("wrist_hip_direct_contact_certified")
                          is True for x in output),
        split_warning="repeated same seed for read-only parity; not independent training holdouts",
        episodes=output)
    return summary

def graphs(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (ROOT/"media").mkdir(exist_ok=True)
    names=[]
    for t in result["episodes"]:
        if t["mode"]!="poll":continue
        stem=f"v080_original_g1_poll_pair_{t['case']}_poll"
        raw=json.loads((ROOT/"reports"/(stem+".json")).read_text())
        trace=raw["traces"][0]
        ts=np.array([r["time_s"] for r in trace],dtype=float)
        com=np.array([r["whole_body_com_xyz"][2] for r in trace],dtype=float)
        nf=np.array([r["nonfoot_contact_N"] for r in trace],dtype=float)
        feet=np.array([[r["support_foot_force_N"]["left"],r["support_foot_force_N"]["right"]] for r in trace])
        q=np.array([r["joint_vel"] for r in trace]);lim=np.array(raw["joint_velocity_limits"])
        fig,ax=plt.subplots(4,1,sharex=True,figsize=(11,10))
        ax[0].plot(ts,com);ax[0].set_ylabel("G1 COM z [m]")
        ax[1].plot(ts,feet[:,0],label="L foot");ax[1].plot(ts,feet[:,1],label="R foot")
        ax[1].legend();ax[1].set_ylabel("foot sensor force [N]")
        ax[2].plot(ts,nf);ax[2].axhline(20,ls="--");ax[2].set_ylabel("non-foot force [N]")
        ax[3].plot(ts,np.max(np.abs(q)/lim,axis=1));ax[3].axhline(1,ls="--")
        ax[3].set_ylabel("max original |qdot| / limit");ax[3].set_xlabel("actual PhysX t [s]")
        hit=t["real_ball_contact_s"]
        if hit is not None:
            for subplot in ax:
                subplot.axvline(hit,ls=":",alpha=.5)
                subplot.axvspan(hit-0.4,hit+1.25,alpha=.05)
        fig.suptitle(stem+" (original 29DoF 200Hz, not human GRF)")
        fig.tight_layout()
        dest=ROOT/"media"/(stem+"_true_fullbody_dynamics.png")
        if not dest.exists():
            with dest.open("xb") as fp:fig.savefig(fp,dpi=130)
        plt.close(fig)
        names.append(str(dest))
    return names

if __name__=="__main__":
    result=audit()
    result["local_write_once_figures"]=graphs(result)
    p=ROOT/"reports/v080_independent_real_200hz.audit.json"
    with p.open("x") as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write("\n")
    print("V080_INDEPENDENT_ACTUAL_200HZ",result["genuinely_measured_gpu_original_g1"],
          "FULL",result["physics_episode_full_14s"],
          "EARLY",result["physics_episode_early_censored"],
          "STRICT_PASS",result["strictly_hard_gates_pass"],flush=True)
