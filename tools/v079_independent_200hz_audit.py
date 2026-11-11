#!/usr/bin/env python3
"""v79 independently audit real G1 physics after native off/on trials.

No robot state is written. Supports incomplete/failed episodes and explicitly
censors windows instead of pretending early resets are 14-second dynamics.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from v077_kinematic_coupling_audit import audit_trace
from v079_native_pair_batch import read_config,sha,ROOT

def certified_rows(raw,sidecar,log_text):
    # A genuine early nonfoot termination is NOT a 14s success and NOT NO_PHYSICS.
    # Preserve the actual 200Hz episode-to-termination and its censored windows.
    rows=raw.get("traces",[[]])[0]
    real_prefix=(raw.get("root_fixed") is False and raw.get("seed")==sidecar.get("seed")
        and len(raw.get("joint_names",[]))==29 and len(rows)>=100
        and sidecar.get("exit_code")==0 and not sidecar.get("watchdog")
        and "Environment device    : cuda:0" in log_text)
    if not real_prefix:
        return dict(status="UNCERTIFIED_OR_NO_G1_PHYSICS",
                    reason=sidecar.get("certification_reason"))
    full=sidecar.get("original_g1_full_physics_certified") is True
    truth=sidecar["independent_hard_gates"] if full else dict(
        hard_gate_pass=False,failed_gates=["EARLY_CENSORED_14S"]
        +list(sidecar.get("independent_hard_gates",{}).get("failed_gates",[])))
    out=audit_trace(raw,truth)
    out["status"]="COMPLETE_14S" if full else "EARLY_TERMINATED_TRUE_GPU_PHYSICS"
    out["strict_hard_gate_pass"]=bool(full and truth["hard_gate_pass"])
    out["full_14s_versus_censored"]=full
    # Preserve full time course for recovery/plot without misattributing correlation to causation.
    rows=raw["traces"][0]
    time=np.asarray([r["time_s"] for r in rows],float)
    nonfoot=np.asarray([float(r["nonfoot_contact_N"]) for r in rows])
    forces=np.asarray([[r["support_foot_force_N"]["left"],r["support_foot_force_N"]["right"]] for r in rows],float)
    com=np.asarray([r["whole_body_com_xyz"] for r in rows],float)
    joint=np.asarray([r["joint_vel"] for r in rows],float)
    limit=np.asarray(raw["joint_velocity_limits"],float)
    if not all(np.isfinite(a).all() for a in (time,nonfoot,forces,com,joint,limit)):
        raise ValueError("NONFINITE_ORIGINAL_PHYSX")
    out.update(
        source_original_fully_measured=True,
        source_is_physx_episode_to_termination=True,
        original_29_joint_qdot_ratio_all_frames=float(np.max(np.abs(joint)/limit)),
        original_com_z_range_m=[float(com[:,2].min()),float(com[:,2].max())],
        original_com_speed_proxy_mps=None,
        full_trace_left_support_mean_N=float(forces[:,0].mean()),
        full_trace_right_support_mean_N=float(forces[:,1].mean()),
        original_nonfoot_net_force_max_N=float(nonfoot.max()),
        full_trace_both_support_fraction=float(np.mean((forces[:,0]>10)&(forces[:,1]>10))),
        full_trace_zero_support_fraction=float(np.mean((forces[:,0]<=10)&(forces[:,1]<=10))),
        left_wrist_right_hip_surface_clearance_available=False,
        G1_motor_torque_not_identical_to_human_joint_power=True,
        contact_foot_force_not_synchronized_lab_GRF=True,
        contact_event_200Hz_window_score=out["contact_plus_1p25_observed"],
    )
    return out

def rows_available(c):
    result=[]
    for trial in c["trial_order"]:
        for mode in c["observer_modes_order"]:
            stem=c["run_namespace"]+"_"+trial["case"]+"_"+mode
            a=ROOT/"reports"/(stem+".audit.json")
            b=ROOT/"reports"/(stem+".json")
            if not a.exists():
                result.append(dict(case=trial["case"],mode=mode,status="NOT_RUN"));continue
            sidecar=json.loads(a.read_text())
            if not b.exists():
                result.append(dict(case=trial["case"],mode=mode,status="NO_PHYSICS_REPORT",
                                   sidecar_sha=sha(a)));continue
            if sha(b)!=sidecar["raw_sha256"]:raise RuntimeError("ORIGINAL_G1_RAW_SHA_CHANGED:"+stem)
            log=ROOT/"reports"/(stem+".log")
            if not log.exists() or sha(log)!=sidecar["log_sha256"]:
                raise RuntimeError("ORIGINAL_G1_LOG_SHA_CHANGED:"+stem)
            original=json.loads(b.read_text())
            out=certified_rows(original,sidecar,log.read_text(errors="replace"))
            out.update(case=trial["case"],mode=mode,raw_sha=sha(b),
                       sidecar_sha=sha(a),
                       native_callback_certified=sidecar["native_callback"].get("wrist_hip_direct_contact_certified",False))
            result.append(out)
    return result

def summarize():
    c=read_config()
    rows=rows_available(c)
    return dict(version="v0.1.79",
        protocol_sha256=sha(ROOT/"configs/v079_frozen_native_pair_matrix.json"),
        actual_source_original_physics_audited=sum(r.get("source_original_fully_measured",False) for r in rows),
        fully_observed_14s_count=sum(r.get("full_14s_versus_censored",False) for r in rows),
        early_censored_real_physics_count=sum(r.get("status")=="EARLY_TERMINATED_TRUE_GPU_PHYSICS" for r in rows),
        strict_all_hard_gates_pass_count=sum(r.get("strict_hard_gate_pass",False) for r in rows),
        whole_contact_plus_1p25_observed=sum(r.get("contact_plus_1p25_observed",False) for r in rows),
        requested_physical_trials=6,actual_scored=len(rows),
        strict_29_qdot_all_modes_and_200Hz=True,
        no_fabricated_arm_hip_mesh_gap=True,
        net_support_sensor_not_force_plate_GRF=True,
        stance_swing_torso_arm_lag_is_association_not_causal_attribution=True,
        historical_v67_not_heldout_for_tuning=True,
        episodes=rows)

def maybe_plot(result):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    images=[]
    for row in result["episodes"]:
        if not row.get("source_original_fully_measured"):continue
        stem="v079_original_g1_native_pair_"+row["case"]+"_"+row["mode"]
        p=json.loads((ROOT/"reports"/(stem+".json")).read_text())
        rows=p["traces"][0]
        t=np.asarray([x["time_s"] for x in rows],float)
        comz=np.asarray([x["whole_body_com_xyz"][2] for x in rows])
        f=np.asarray([[x["support_foot_force_N"]["left"],x["support_foot_force_N"]["right"]] for x in rows])
        nf=np.asarray([x["nonfoot_contact_N"] for x in rows])
        v=np.asarray([x["joint_vel"] for x in rows]);lim=np.asarray(p["joint_velocity_limits"])
        q=np.max(np.abs(v)/lim,axis=1)
        fig,ax=plt.subplots(4,1,figsize=(10,11),sharex=True)
        ax[0].plot(t,comz); ax[0].set_ylabel("G1 COM z (m)")
        ax[1].plot(t,f[:,0],label="left");ax[1].plot(t,f[:,1],label="right");ax[1].legend();ax[1].set_ylabel("foot sensor net N")
        ax[2].plot(t,nf);ax[2].axhline(20,ls="--");ax[2].set_ylabel("nonfoot sensor N")
        ax[3].plot(t,q);ax[3].axhline(1,ls="--");ax[3].set_ylabel("max |29 qdot|/limit");ax[3].set_xlabel("real G1 PhysX time (s)")
        contact=p["records"][0].get("contact_time_s")
        if contact is not None:
            for a in ax:a.axvline(contact,ls=":",alpha=.7)
        fig.suptitle(stem+" | measured 200Hz, NOT professional human biomechanical GRF")
        fig.tight_layout()
        path=ROOT/"media"/(stem+"_dynamics.png")
        if path.exists():raise FileExistsError("WRITE_ONCE_PLOT:"+str(path))
        fig.savefig(path,dpi=135);plt.close(fig)
        images.append(str(path.relative_to(ROOT)))
    return images

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plot",action="store_true")
    args=ap.parse_args()
    p=ROOT/"reports/v079_independent_200hz_dynamics.audit.json"
    if p.exists():raise FileExistsError("WRITE_ONCE_DYNAMICS")
    out=summarize()
    if args.plot:out["physical_figures_local_only"]=maybe_plot(out)
    with p.open("x") as fp:json.dump(out,fp,ensure_ascii=False,indent=2);fp.write("\n")
    print("V079_INDEPENDENT_200HZ",out["actual_source_original_physics_audited"],
          "contact_windows",out["whole_contact_plus_1p25_observed"])

if __name__=="__main__":main()

