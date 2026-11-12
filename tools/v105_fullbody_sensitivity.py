#!/usr/bin/env python3
"""SHA-qualified original 200Hz G1 full-body onset sensitivity and falsification.

Historical G-K comparison ONLY. Not new physics, new policy, or elite-human evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v104_paired_dynamics import load_source_manifest, sha_file, extract
from audit_v067_hard_gates import physically_scored

def onset(t, dpos, dvel, indices, cutoff):
    idx=np.asarray(indices,dtype=int)
    if idx.size==0 or np.any(idx<0) or np.any(idx>=29):
        raise ValueError("INVALID_JOINT_GROUP")
    amplitude=np.maximum(np.max(np.abs(dpos[:,idx]),axis=1),
                         np.max(np.abs(dvel[:,idx]),axis=1))
    hit=np.flatnonzero(amplitude>cutoff)
    return float(t[hit[0]]) if len(hit) else None

def sensitivity(base, other, cutoffs, contact_time):
    t=np.asarray(base["t"])
    k=min(len(t),len(other["t"]))
    if k<2 or k!=len(other["t"][:k]) or not np.allclose(t[:k],other["t"][:k],rtol=0,atol=.00015):
        raise ValueError("UNMATCHED_ACTUAL_TIME")
    if np.max(np.abs(np.diff(t[:k])-.005))>.00015:
        raise ValueError("NOT_200HZ")
    if base["q"].shape[1]!=29 or other["q"].shape[1]!=29:
        raise ValueError("NOT_ALL_ORIGINAL_29DOF")
    p=base["q"][:k]-other["q"][:k]
    v=base["dq"][:k]-other["dq"][:k]
    if not (np.isfinite(p).all() and np.isfinite(v).all()):
        raise ValueError("NONFINITE_PHYSICAL_STATES")
    result=[]
    for z in cutoffs:
        if not np.isfinite(z) or z<=0:
            raise ValueError("INVALID_THRESHOLD")
        first=onset(t[:k],p,v,range(29),z)
        groups={g:onset(t[:k],p,v,inds,z) for g,inds in base["ix"].items()}
        result.append(dict(threshold=z,all29_first_onset_s=first,lead_before_contact_s=
                           None if first is None or contact_time is None else float(contact_time-first),
                           group_first_onset_s=groups))
    return dict(common_real_200hz_steps=k,observed_duration_s=float(t[k-1]),
                right_censored_by_shorter_trajectory=len(base["t"])!=len(other["t"]),
                self_observed_null=bool(np.max(np.abs(p))==0 and np.max(np.abs(v))==0),
                max_original_joint_position_delta_rad=float(np.abs(p).max()),
                max_original_joint_velocity_delta_rad_s=float(np.abs(v).max()),
                thresholds=result)

def prereg(c):
    expected=(1e-7,1e-6,1e-5,1e-4,1e-3,1e-2)
    if c["version"]!="v0.1.105" or c["parent_commit"]!="8d149cb8693ef19cbd42e8b4d9cc3d4358e11e00":
        raise ValueError("SOURCE_PROTOCOL_CHANGED")
    if tuple(c["full29_threshold_sweep"])!=expected or c["time_step_s"]!=.005:
        raise ValueError("UNREGISTERED_TOLERANCE")
    if c["mandatory_matched_sites"]!=list("GHIJK") or c["mandatory_controls"]!=["v32","cmu02","cmu05"]:
        raise ValueError("UNREGISTERED_EPISODES")
    if c["safe_policy_actuation_allowed"]:raise ValueError("UNSAFE_ACTUATION_ENABLED")
    return expected

def evaluate(c, source_roots, out):
    if out.exists() or out.is_symlink():raise FileExistsError(out)
    tolerances=prereg(c)
    sites={"G":"v097","H":"v097","I":"v098","J":"v098","K":"v098"}
    manifests={};master_sha={}
    for version,root in source_roots.items():
        manifests[version],master_sha[version]=load_source_manifest(Path(root),version)
    shots={};states={};pairs={}
    for site,v in sites.items():
        for controller in c["mandatory_controls"]:
            name=f"{site}_{controller}"
            path,e=manifests[v][name]
            if sha_file(path)!=e["raw_sha256"]:raise ValueError("RAW_SHA_MISMATCH:"+name)
            raw=json.loads(path.read_text())
            gate=physically_scored(raw,wrapper_certified=True)
            if bool(gate["hard_gate_pass"])!=bool(e["gates"]["hard_gate_pass"]):
                raise ValueError("GATE_MISMATCH:"+name)
            states[name]=extract(raw)
            shots[name]=dict(original_physx_source_sha256=e["raw_sha256"],
                original29_joint_order_sha256=hashlib.sha256(json.dumps(raw["joint_names"]).encode()).hexdigest(),
                existing_strict_hard_gate_pass=bool(gate["hard_gate_pass"]),
                original_hard_gate_failed=list(gate["failed_gates"]),
                true_contact_time_s=raw["records"][0].get("contact_time_s"),
                actual_observed_200hz_steps=len(states[name]["t"]))
    for site in sites:
        a=states[f"{site}_v32"]
        ref_contact=shots[f"{site}_v32"]["true_contact_time_s"]
        for other in ("cmu02","cmu05"):
            b=states[f"{site}_{other}"]
            key=f"{site}_v32_vs_{other}"
            ab=sensitivity(a,b,tolerances,ref_contact)
            ba=sensitivity(b,a,tolerances,ref_contact)
            aa=sensitivity(a,a,tolerances,ref_contact)
            if not aa["self_observed_null"] or any(e["all29_first_onset_s"] is not None for e in aa["thresholds"]):
                raise AssertionError("SELF_PAIR_NONZERO")
            if ab["thresholds"]!=ba["thresholds"]:
                raise AssertionError("NON_SYMMETRIC_COORDINATION_ONSET")
            ab["reverse_pair_agrees_exactly"]=True
            ab["self_pair_no_onset_at_every_threshold"]=True
            pairs[key]=ab
    nominal=[x["thresholds"][2]["all29_first_onset_s"] for x in pairs.values()]
    bytol={str(z):{"observed_pairs":sum(x["thresholds"][i]["all29_first_onset_s"] is not None for x in pairs.values()),
        "first_onsets_s":[x["thresholds"][i]["all29_first_onset_s"] for x in pairs.values()]}
        for i,z in enumerate(tolerances)}
    result=dict(version=c["version"],frozen_protocol_sha256=sha_file(ROOT/"configs/v105_frozen_protocol.json"),
                independent_historical_g1_source_masters_sha256=master_sha,
                historical_verified_200hz_G1_episodes=len(shots),
                previous_original_strict_success_count=sum(x["existing_strict_hard_gate_pass"] for x in shots.values()),
                new_original_G1_physics_episodes=0,new_isolated_physics_episodes=0,
                original_G1_observer_off_on_physics_parity_newly_verified=False,
                human_expert_synchronized_GRF_truth=False,
                native_wrist_hip_collider_manifold_certified=False,
                human_shot_naturalness_or_causal_improvement_proven=False,
                nominal_1e_minus5_first_onsets_s=nominal,threshold_sensitivity=bytol,
                shots=shots,pairs=pairs)
    with out.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    return result

def plot(res,p):
    if p.exists():raise FileExistsError("WRITE_ONCE_PLOT")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs=list(res["pairs"])
    cutoffs=[x["threshold"] for x in res["pairs"][xs[0]]["thresholds"]]
    matrix=np.array([[float(t["all29_first_onset_s"]) if t["all29_first_onset_s"] is not None else np.nan
       for t in res["pairs"][k]["thresholds"]] for k in xs])
    fig,ax=plt.subplots(figsize=(11,6))
    for i,x in enumerate(xs):ax.plot(cutoffs,matrix[i,:],marker="o",label=x)
    ax.set_xscale("log");ax.set_xlabel("absolute original q/qdot divergence threshold")
    ax.set_ylabel("first measured 200Hz full29 state difference (s)")
    ax.set_title("Historical G1 measured causal-prefix sensitivity; NOT human style evidence")
    ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(p,dpi=150);plt.close(fig)

if __name__=="__main__":
    pa=argparse.ArgumentParser();pa.add_argument("--out",required=True);pa.add_argument("--plot")
    args=pa.parse_args();c=json.loads((ROOT/"configs/v105_frozen_protocol.json").read_text())
    source={"v097":"/home/you/projects/kinetickick-wt-v097",
            "v098":"/home/you/projects/kinetickick-wt-v098"}
    result=evaluate(c,source,Path(args.out))
    if args.plot:plot(result,Path(args.plot))
    print(json.dumps({k:v for k,v in result.items() if k not in ("shots","pairs")},ensure_ascii=False))
