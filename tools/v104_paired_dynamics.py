#!/usr/bin/env python3
"""Historical original 200 Hz 29DoF paired dynamics: SHA verified, no fabricated recovery.

Constrained to frozen v97 G/H and v98 I/J/K. Not elite human imitation proof.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_v067_hard_gates import physically_scored
from v100_coupled_dynamics import physical_grid, describe_episode, require_joint_groups

def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()

def load_source_manifest(root,version):
    master_p=Path(root)/"reports"/f"{version}_g1_physics_final.audit.json"
    master=json.loads(master_p.read_text())
    assert master["version"]=="v0.1."+version[-2:], "UNEXPECTED_MASTER_VERSION"
    cases={}
    for e in master["runs"]:
        if not e["true_original_200Hz_G1_CUDA_PhysX"]:
            raise ValueError("UNVERIFIED_G1_SOURCE")
        name=e["name"]
        if name in cases:raise ValueError("DUPLICATE_CASE")
        cases[name]=(Path(root)/"reports"/f"{version}_{name}.json",e)
    return cases,sha_file(master_p)

def extract(raw):
    names=raw["joint_names"]
    ix=require_joint_groups(names)
    tr=raw["traces"][0]
    good,duplicates=physical_grid(tr)
    rows=[tr[int(i)] for i in good]
    t=np.array([r["time_s"] for r in rows],dtype=float)
    q=np.array([r["joint_pos"] for r in rows],dtype=float)
    dq=np.array([r["joint_vel"] for r in rows],dtype=float)
    limits=np.array(raw["joint_velocity_limits"],dtype=float).reshape(-1)
    if q.shape!=(len(t),29) or dq.shape!=q.shape or limits.shape!=(29,):
        raise ValueError("NOT_FULL_29DOF")
    if (not np.isfinite(q).all() or not np.isfinite(dq).all()
            or not np.isfinite(limits).all() or np.any(limits<=0)):
        raise ValueError("UNQUALIFIED_29DOF_STATE")
    return dict(t=t,q=q,dq=dq,limits=limits,ix=ix,duplicate_count=duplicates)

def match_prefix(a,b,atol=0.00001,dt=0.005):
    """Compare actual measured timestamps only; no interpolation or tail filling."""
    ta,tb=a["t"],b["t"]
    if not np.isclose(ta[0],tb[0],atol=1e-6):
        raise ValueError("DIFFERENT_TIME_ORIGINS")
    k=min(len(ta),len(tb))
    if k<2:raise ValueError("SHORT_PAIRED_PHYSICAL_PREFIX")
    if not np.allclose(ta[:k],tb[:k],rtol=0,atol=0.00015):
        raise ValueError("MISALIGNED_ACTUAL_200HZ_PREFIX")
    if np.max(np.abs(np.diff(ta[:k])-dt))>.00015:
        raise ValueError("NOT_200HZ")
    if a["q"].shape[1]!=29 or b["q"].shape[1]!=29:
        raise ValueError("INCOMPLETE_G1")
    # These two vectors refer to identical known source joint order, checked by caller.
    dpos=a["q"][:k]-b["q"][:k]
    dvel=a["dq"][:k]-b["dq"][:k]
    absdiff=np.maximum(np.max(np.abs(dpos),axis=1),np.max(np.abs(dvel),axis=1))
    ix=np.flatnonzero(absdiff>atol)
    first=float(ta[ix[0]]) if len(ix) else None
    drift={}
    for key,indices in a["ix"].items():
        diff=dvel[:,indices]
        drift[key]=float(np.sqrt(np.mean(diff**2)))
    return dict(observed_matched_200hz_steps=k,common_observed_duration_s=float(ta[k-1]),
         first_full29_state_divergence_s=first,unchanged_within_fixed_tolerance=bool(not len(ix)),
         max_abs_joint_position_delta_rad=float(np.abs(dpos).max()),
         max_abs_joint_velocity_delta_rad_s=float(np.abs(dvel).max()),
         rms_joint_position_delta_rad=float(np.sqrt(np.mean(dpos*dpos))),
         rms_joint_velocity_delta_rad_s=float(np.sqrt(np.mean(dvel*dvel))),
         group_rms_joint_velocity_delta_rad_s=drift,
         unobserved_tail_not_extrapolated=True,
         earlier_termination_causes_pair_right_censoring=bool(len(ta)!=len(tb)))

def phase_correlation(x,y,maxlag=20):
    """Exploratory cross-correlation of measured energy, not proof of naturalness."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)!=len(y) or len(x)<80:raise ValueError("SHORT_PHASE_SAMPLE")
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("INVALID_PHASE")
    vals=[]
    for lag in range(-maxlag,maxlag+1):
        aa=x[max(0,lag):len(x)+min(0,lag)]
        bb=y[max(0,-lag):len(y)-max(0,lag)]
        a0=aa-aa.mean();b0=bb-bb.mean()
        denom=float(np.linalg.norm(a0)*np.linalg.norm(b0))
        vals.append(float(a0@b0/denom) if denom>1e-8 else float("nan"))
    if not any(np.isfinite(z) for z in vals):
        return dict(observable=False,reason="INSUFFICIENT_ACTIVITY_VARIANCE")
    valid=np.array(vals)
    best=int(np.nanargmax(valid))
    return dict(observable=True,lag_arm_relative_right_leg_s=round((best-maxlag)*.005,3),
                max_correlation=float(valid[best]),
                zero_lag_correlation=float(vals[maxlag]) if np.isfinite(vals[maxlag]) else None,
                signal_is_group_abs_joint_velocity_not_angular_momentum=True)

def first_unsafe_nonfoot(raw,series,threshold_N=20.):
    # Measured net force is NOT a true collider-pair manifold certificate.
    rows=raw["traces"][0]
    ix,_=physical_grid(rows)
    vals=np.asarray([rows[int(i)]["nonfoot_contact_N"] for i in ix],dtype=float)
    if len(vals)!=len(series["t"]) or not np.isfinite(vals).all():
        raise ValueError("UNQUALIFIED_NONFOOT_SENSOR")
    hit=np.flatnonzero(vals>threshold_N)
    if not len(hit):
        return dict(first_over_20N_s=None,first_force_N=None,
                    no_event_observed_before_termination_only=True)
    j=int(hit[0])
    return dict(first_over_20N_s=float(series["t"][j]),
                first_force_N=float(vals[j]),true_native_collision_pair_certified=False)

def observed_phase(raw,series):
    c=raw["records"][0].get("contact_time_s")
    if c is None:return dict(observable=False,reason="NO_CONTACT")
    c=float(c);t=series["t"]
    mask=(t>=c-.4-.0001)&(t<=c+.0001)
    if not (t[0]<=c-.4+1e-4 and mask.sum()>=79):
        return dict(observable=False,reason="CONTACT_PREWINDOW_CENSORED")
    group={}
    for name,ix in series["ix"].items():
        group[name]=np.mean(np.abs(series["dq"][mask][:,ix])/series["limits"][None,ix],axis=1)
    return {"left_counterarm_right_leg":phase_correlation(group["left_arm"],group["right_leg"]),
            "waist_right_leg":phase_correlation(group["waist"],group["right_leg"]),
            "left_support_right_leg":phase_correlation(group["left_leg"],group["right_leg"]),
            "start_time_s":float(t[mask][0]),"contact_time_s":c,
            "true_precontact_samples":int(mask.sum()),
            "not_expert_human_naturalness_metric":True}

def main(config_path:Path,output:Path):
    if output.exists():raise FileExistsError("WRITE_ONCE_OUTPUT")
    c=json.loads(config_path.read_text())
    if c["version"]!="v0.1.104" or c["gpu_allowed"] or not c["historical_only"]:
        raise ValueError("PROTOCOL_DRIFT")
    if len(c["sites"])!=5 or len(c["controllers"])!=3 or len(c["pairing"])!=2:
        raise ValueError("UNFROZEN_STUDY_DESIGN")
    imported={};masters={}
    for version,path in c["source_roots"].items():
        imported[version],masters[version]=load_source_manifest(path,version)
    episodes={};series={}
    for site,version in c["sites"].items():
        for control in c["controllers"]:
            name=site+"_"+control
            p,e=imported[version][name]
            if not p.is_file() or sha_file(p)!=e["raw_sha256"]:
                raise ValueError("SOURCE_SHA_MISMATCH:"+name)
            raw=json.loads(p.read_text())
            if raw["joint_names"]!=next(iter(episodes.values()))["joint_names"] if episodes else False:
                raise ValueError("DIFFERENT_ORIGINAL_29DOF_ORDER")
            original=physically_scored(raw,wrapper_certified=True)
            if bool(original["hard_gate_pass"])!=bool(e["gates"]["hard_gate_pass"]):
                raise ValueError("ORIGINAL_GATE_DISAGREEMENT:"+name)
            s=extract(raw)
            d=describe_episode(raw)
            episodes[name]=dict(source_version=version,source_sha256=e["raw_sha256"],
                joint_names=raw["joint_names"],certified_original_G1_physx_200Hz=True,
                hard_gate_pass=bool(original["hard_gate_pass"]),
                failed_gates=list(original["failed_gates"]),
                actual_steps=d["actual_unique_200hz_steps"],
                terminal_duplicate_not_physical_step=d["terminal_duplicate_rows"],
                true_postcontact_1p25_observed=d["post_1p25_observed"],
                true_precontact_400ms_observed=d["pre_400ms_observed"],
                completed_full_14s=d["completed_full_14s"],
                max_source29_qdot_limit_ratio=d["actual29_qdot_limit_max_ratio"],
                contact_time_s=raw["records"][0].get("contact_time_s"),
                legal_instep_fraction=raw["records"][0].get("requested_surface_impulse_fraction"),
                nonfoot_peak_N=d["nonfoot_peak_N"],
                first_nonfoot_risk=first_unsafe_nonfoot(raw,s),
                support_sensor_foot_force_N_not_human_GRF=d["support_sensor_N_not_expert_GRF"],
                exploratory_group_phase=observed_phase(raw,s))
            series[name]=s
    if len(episodes)!=15:raise ValueError("INCOMPLETE_15")
    pairs={}
    for site in c["sites"]:
        for base,other in c["pairing"]:
            ref_a=site+"_"+base;ref_b=site+"_"+other
            aa=series[ref_a];bb=series[ref_b]
            if episodes[ref_a]["joint_names"]!=episodes[ref_b]["joint_names"]:
                raise ValueError("WRONG_JOINT_ORDER")
            pair=match_prefix(aa,bb,c["divergence_absolute_tolerance"])
            if pair["first_full29_state_divergence_s"] is not None:
                pair["measured_lead_to_contact_s"]={
                    control:(None if episodes[site+"_"+control]["contact_time_s"] is None
                    else float(episodes[site+"_"+control]["contact_time_s"]-pair["first_full29_state_divergence_s"]))
                    for control in (base,other)}
            pairs[f"{site}_{base}_vs_{other}"]=pair
    succeeded=[k for k,v in episodes.items() if v["hard_gate_pass"]]
    result=dict(version=c["version"],protocol_sha256=sha_file(config_path),
      frozen_parent_sha=c["parent_commit"],historical_raw_source_audits_sha256=masters,
      original_sha_verified_episode_count=len(episodes),
      original_strict_pass_count=len(succeeded),original_strict_pass_names=succeeded,
      total_historical_true_200Hz_physics_steps=sum(v["actual_steps"] for v in episodes.values()),
      historical_paired_comparisons=len(pairs),new_G1_PhysX_runs=0,
      new_isolated_PhysX_runs=0,new_shooting_success_denominator=None,
      observed_post_1p25_count=sum(v["true_postcontact_1p25_observed"] for v in episodes.values()),
      censored_post_1p25_count=sum(not v["true_postcontact_1p25_observed"] for v in episodes.values()),
      entire_14s_recovery_observed_count=sum(v["completed_full_14s"] for v in episodes.values()),
      actual_29_qdot_overlimit_episode_count=sum(v["max_source29_qdot_limit_ratio"]>1.000001 for v in episodes.values()),
      human_elite_motion_or_GRF_ground_truth=False,
      validated_native_wrist_hip_actor_collider_manifold=False,
      physical_actuation_of_new_learned_policy=False,
      causal_improvement_or_naturalness_proven=False,
      episodes=episodes,pairs=pairs)
    with output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print("V104_AUDIT",json.dumps({k:v for k,v in result.items() if k not in ("episodes","pairs")}),flush=True)
    return result

def plot_pair_grid(result,png:Path):
    if png.exists():raise FileExistsError("WRITE_ONCE_PLOT")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    key=list(result["pairs"])
    data=[result["pairs"][k] for k in key]
    fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True)
    axes[0].bar(range(len(key)),[p["rms_joint_velocity_delta_rad_s"] for p in data])
    axes[0].set_ylabel("Matched 200Hz prefix RMS difference in all 29 qdot (rad/s)")
    axes[1].bar(range(len(key)),[p["observed_matched_200hz_steps"] for p in data])
    axes[1].set_ylabel("Observed paired prefix (real 200Hz frames)")
    axes[1].set_xticks(range(len(key)),key,rotation=55,ha="right")
    fig.suptitle("Original G1 PhysX historical paired differences: not new physics or human-motion quality")
    fig.tight_layout();fig.savefig(png,dpi=130);plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--plot",type=Path)
    a=p.parse_args()
    res=main(a.config,a.output)
    if a.plot:plot_pair_grid(res,a.plot)
