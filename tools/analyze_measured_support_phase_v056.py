#!/usr/bin/env python3
"""Measured 200Hz body-group phase coupling with censoring and foot normal loads.

Descriptive physics-only timing features, NOT human naturalness or force plates.
No fitting, action changes, old evidence edits, or invented post-failure frames.
"""
import json
from pathlib import Path
import numpy as np
from audit_real_batch_v056 import ROOT,OUTPUT as AUDIT,CFG,sha

OUT=ROOT/"reports/v056_coupled_temporal_dynamics.json"
MEDIA=ROOT/"media"
GROUPS={
 "stance_left":("left_hip","left_knee","left_ankle"),
 "swing_right":("right_hip","right_knee","right_ankle"),
 "trunk":("waist_",),
 "counter_arm_left":("left_shoulder","left_elbow","left_wrist"),
 "counter_arm_right":("right_shoulder","right_elbow","right_wrist")
}

def smooth(values,width=9):
    if width<1 or width%2!=1:raise ValueError("Odd smoothing width required")
    padding=np.pad(np.asarray(values,dtype=float),(width//2,width//2),mode="edge")
    return np.convolve(padding,np.ones(width)/width,mode="valid")

def lag_ms(a,b,dt=.005,max_lag_s=.2):
    """Best bounded temporal activation lag, not physics causality."""
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)!=len(b) or len(a)<20 or np.std(a)<1e-9 or np.std(b)<1e-9:
        return None
    steps=int(max_lag_s/dt);samples=[]
    for lag in range(-steps,steps+1):
        if abs(lag)>=len(a)//2:continue
        aa=a[max(0,lag):len(a)+min(0,lag)]
        bb=b[max(0,-lag):len(b)-max(0,lag)]
        if len(aa)<20 or np.std(aa)<1e-9 or np.std(bb)<1e-9:continue
        corr=float(np.corrcoef(aa,bb)[0,1])
        if np.isfinite(corr):samples.append((corr,lag))
    if not samples:return None
    corr,lag=max(samples,key=lambda row:(row[0],-abs(row[1])))
    return dict(lag_s=round(lag*dt,4),normalized_correlation=round(corr,5),
                caveat="descriptive lagged group speed envelope, not human causal sequencing")

def extract(report):
    rows=report["traces"][0]
    t=np.array([row["time_s"] for row in rows])
    q=np.asarray([row["joint_vel"] for row in rows])
    lim=np.asarray(report["joint_velocity_limits"][0])
    if len(report["joint_names"])!=29 or q.shape!=(len(t),29):
        raise ValueError("Actual G1 29DoF velocity required")
    if np.max(np.abs(q)/lim)>1.000001:
        # Record this violation, never suppress it by normalization.
        overrun=True
    else:overrun=False
    recorded_contact=report["records"][0].get("contact_time_s")
    if recorded_contact is None:
        # Do not fabricate a contact-aligned phase or 1.25s follow-through.
        times=np.array([row["time_s"] for row in rows],dtype=float)
        forces=np.asarray([[row["support_foot_force_N"][side] for side in ("left","right")]
            for row in rows],dtype=float)
        return dict(contact_time_s=None,available_last_physics_s=float(times[-1]),
            pre_window_observed=False,post_1250ms_observed=False,
            right_censored_by_physical_termination=True,
            no_foot_ball_contact_before_physical_termination=True,
            all_original_qdot_limit_violated=overrun,
            group_activation=None,lag_to_swing_right=None,
            ground_support_pre_400ms=None,
            ground_support_post_at_most_1250ms=None,
            available_ground_normal_N_means=[float(x) for x in np.mean(forces,axis=0)],
            human_grf_or_cop_verified=False,no_mesh_clearance_distance_claim=True)
    t0=float(recorded_contact)
    sel=(t>=t0-.4)&(t<=t0+.30)
    times=t[sel]
    groups={}
    envelopes={}
    for key,tokens in GROUPS.items():
        idx=[i for i,name in enumerate(report["joint_names"])
             if any(token in name for token in tokens)]
        if not idx:raise ValueError("Missing G1 joint group "+key)
        speed=np.sqrt(np.mean((q[sel][:,idx]/lim[idx])**2,axis=1))
        envelope=smooth(speed,9)
        envelopes[key]=envelope
        peak=int(np.argmax(envelope))
        groups[key]=dict(n_real_original_joints=len(idx),
                         measured_peak_relative_contact_s=round(float(times[peak]-t0),5),
                         max_normalized_group_speed=float(envelope[peak]),
                         integrated_speed_ratio_s=float(np.trapz(envelope,times)))
    pairwise={label:lag_ms(envelopes[label],envelopes["swing_right"])
        for label in GROUPS if label!="swing_right"}
    force=np.array([[r["support_foot_force_N"][side] for side in ("left","right")]
                    for r in rows],float)
    support=(t>=t0-.4)&(t<=min(t0+1.25,t[-1]))
    before=(t>=t0-.4)&(t<t0)
    after=(t>=t0)&(t<=t0+1.25)
    def stats(mask):
        f=force[mask]
        if len(f)==0:return None
        return dict(n=len(f),mean_left_normal_N=float(np.mean(f[:,0])),
              mean_right_normal_N=float(np.mean(f[:,1])),
              left_support_80N_fraction=float(np.mean(f[:,0]>=80)),
              dual_support_80N_fraction=float(np.mean((f[:,0]>=80)&(f[:,1]>=80))))
    return dict(contact_time_s=t0,available_last_physics_s=float(t[-1]),
                pre_window_observed=bool(before.any() and t[before][0]<=t0-.39),
                post_1250ms_observed=bool(after.any() and t[after][-1]>=t0+1.24),
                right_censored_by_physical_termination=bool(t[-1]<t0+1.24),
                all_original_qdot_limit_violated=overrun,
                group_activation=groups,lag_to_swing_right=pairwise,
                ground_support_pre_400ms=stats(before),
                ground_support_post_at_most_1250ms=stats(after),
                human_grf_or_cop_verified=False,
                no_mesh_clearance_distance_claim=True)

def run():
    if OUT.exists():raise FileExistsError("Preserve original 200Hz derived audit")
    audit=json.loads(AUDIT.read_text())
    cfg=json.loads(CFG.read_text())
    cases={x["id"]:x for x in cfg["cases"]}
    result=[]
    for item in audit["results"]:
        case=cases[item["id"]]
        filename=ROOT/case["output"]
        if sha(filename)!=item["original_sha256"]:
            raise ValueError("Raw physical SHA drift")
        raw=json.loads(filename.read_text())
        result.append(dict(id=item["id"],policy=item["policy"],
                           strict_success=item["strict_success"],**extract(raw)))
    out=dict(version="0.1.56",observed_real_physx_episodes=len(result),
        physics_dt=.005,information_not_inferred_if_censored=True,
        proxy_temporal_lag_does_not_measure_human_naturalness=True,
        cases=result)
    with OUT.open("x") as f:json.dump(out,f,indent=2,ensure_ascii=False);f.write("\n")
    # Print a physics-dynamics figure for every distinct failure type: baseline
    # center and one first new seed. Raw sources stay on Tang.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for item in (result[0],next((r for r in result if r["right_censored_by_physical_termination"] and r["contact_time_s"] is not None),None)):
        if not item:continue
        original=json.loads((ROOT/cases[item["id"]]["output"]).read_text())
        trace=original["traces"][0]
        t=np.array([r["time_s"] for r in trace]);t0=item["contact_time_s"]
        mask=(t>=t0-.4)&(t<=min(t0+1.25,t[-1]))
        forces=np.array([[r["support_foot_force_N"][side] for side in ("left","right")]
                         for r in trace])
        plt.figure(figsize=(11,4))
        plt.plot(t[mask]-t0,forces[mask,0],label="left contact normal (not human GRF)")
        plt.plot(t[mask]-t0,forces[mask,1],label="right contact normal (not human GRF)")
        plt.axvline(0,color="k",linestyle="--",label="recorded ball contact")
        plt.xlabel("Seconds relative to actual ball collision")
        plt.ylabel("Original PhysX foot-ground normal force (N)")
        plt.title(item["id"]+(" (truncated by physical termination)" if item["right_censored_by_physical_termination"] else ""))
        plt.legend();plt.tight_layout()
        name=MEDIA/("v056_measured_support_"+item["id"]+".png")
        if name.exists():raise FileExistsError(name)
        plt.savefig(name,dpi=150);plt.close()
    print("V056_COUPLED_29DOF_TEMPORAL_OBSERVATIONS",
        json.dumps(dict(n=len(result),
          incomplete_post_windows=sum(r["right_censored_by_physical_termination"] for r in result),
          observed_qdot_violations=sum(r["all_original_qdot_limit_violated"] for r in result))),flush=True)

if __name__=="__main__":run()
