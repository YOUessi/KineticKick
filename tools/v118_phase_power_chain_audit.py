#!/usr/bin/env python3
"""SHA-verified measured G1 support/power chronology: falsification, not naturalness or human-force proof.

No new policy is trained or actuated. Dynamics rows, 29 qdot, and joint torque
come from existing genuine 200Hz free-root G1 PhysX. False inference from
partial traces or uncalibrated human kinematics must fail closed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from v113_phase_biomechanics import load_physical_arrays
from v117_causal_safety_statistics import sha, true_joint_hazard, measured_support_and_work
ROOT=Path(__file__).resolve().parents[1]
GROUPS=("waist","left_leg","right_leg","left_arm","right_arm")

def _window_indices(t, start, stop):
    return np.flatnonzero((t>=start-1e-7)&(t<stop-1e-7))

def _first_sustained_effort(t, work, low, high, frames=3):
    """Onset >max(20% own peak, 0.1J/s) in frames consecutive; no invented onset."""
    ix=_window_indices(t,low,high)
    if len(ix)<frames:return None
    y=np.asarray(work[ix],float)
    if not np.isfinite(y).all():raise ValueError("NONFINITE_ACTUAL_POWER")
    peak=float(y.max())
    if peak<0.5:return None
    above=y>=max(0.2*peak,0.5)
    for j in range(len(ix)-frames+1):
        if bool(np.all(above[j:j+frames])):return float(t[ix[j]])
    return None

def phase_chain(a):
    t=np.asarray(a["t"],float); impact=float(a["impact"]); dt=np.diff(t)
    if t.ndim!=1 or len(t)<100 or not np.allclose(dt,0.005,rtol=0,atol=1e-6):
        raise ValueError("NON_200HZ_REAL_TIME_GRID")
    n=len(t); v=np.asarray(a["vel"],float); tau=np.asarray(a["tau"],float)
    support=np.asarray(a["support"],float)
    if v.shape!=(n,29) or tau.shape!=(n,29) or support.shape!=(n,2):
        raise ValueError("MISSING_OR_INVALID_FULLBODY_PHYSICAL_CHANNELS")
    if not all(np.isfinite(x).all() for x in (v,tau,support)):
        raise ValueError("NONFINITE_PHYSICS")
    pre=_window_indices(t,impact-0.4,impact)
    post=_window_indices(t,impact,impact+1.25+0.005)
    if len(pre)<79:raise ValueError("CENSORED_PRECONTACT_400MS")
    p=np.maximum(v*tau,0.0)
    groups={name:np.sum(p[:,a["groups"][name]],axis=1) for name in GROUPS}
    parts=(("pre_early",impact-.4,impact-.2),
           ("pre_late",impact-.2,impact),
           ("post_early",impact,impact+.2),
           ("post_late",impact+.2,impact+1.25))
    windows={}
    for name,lo,hi in parts:
        ix=_window_indices(t,lo,hi)
        windows[name]=dict(real_rows=int(len(ix)),partial=bool(len(ix) < int(round((hi-lo)/.005))-1),
           positive_estimated_motor_work_J={g:float(groups[g][ix].sum()*.005) for g in GROUPS},
           left_support_force_proxy_integral_Ns=float(support[ix,0].sum()*.005),
           right_support_force_proxy_integral_Ns=float(support[ix,1].sum()*.005))
    onsets={g:_first_sustained_effort(t,groups[g],impact-.4,impact) for g in GROUPS}
    # Temporal negative control: deliberately reverse the pre-impact power
    # sequence, retaining original times; not a simulated trajectory.
    reversed_onsets={}
    for g in GROUPS:
        rev=groups[g].copy();rev[pre]=rev[pre][::-1]
        reversed_onsets[g]=_first_sustained_effort(t,rev,impact-.4,impact)
    hazards=true_joint_hazard(a)
    return dict(source="actual_200hz_original_G1_PhysX",
        impact_time_s=impact,physical_rows=n,
        contact_pre400_rows=len(pre),post1250_rows_observed=len(post),
        post1250_censored=bool(t[-1] < impact+1.25-1e-6),
        phase_windows=windows,first_sustained_precontact_positive_motor_power_s=onsets,
        reversed_chronology_negative_control_onsets_s=reversed_onsets,
        chronology_onset_differs_from_reversal=any(
            onsets[g] is not None and reversed_onsets[g] is not None and
            abs(onsets[g]-reversed_onsets[g])>=.005 for g in GROUPS),
        positive_motor_work_proxy_not_biological_power=True,
        foot_support_force_proxy_not_athlete_forceplate_GRF=True,
        cannot_identify_professional_instep_kinetic_chain=True,
        qdot=hazards)

def audit(root=ROOT):
    path=root/"reports/v118_original_g1_batch.audit.json"
    master=json.loads(path.read_text())
    if master["version"]!="v0.1.118":raise ValueError("WRONG_FROZEN_BATCH")
    records=[];independent=[]
    for row in master["runs"]:
        if not row["certified_original_G1_cuda_200Hz"]:
            records.append(dict(name=row["name"],certified=False,reason=row["certification"]))
            continue
        rawpath=root/"reports"/("v118_"+row["name"]+".json")
        if sha(rawpath)!=row["raw_sha256"]:raise ValueError("ORIGINAL_PHYSX_SHA_DRIFT")
        raw=json.loads(rawpath.read_text())
        raw["joint_velocity_limits"]=np.asarray(raw["joint_velocity_limits"],dtype=float).reshape(-1).tolist()
        a=load_physical_arrays(raw)
        if len(a["t"])!=row["actual_traced_200hz_rows"]:
            raise ValueError("UNQUALIFIED_ACTUAL_FRAME_COUNT")
        measured=phase_chain(a)
        rec=dict(name=row["name"],scene=row["scene"],control=row["control"],
                 observer=row["observer"],raw_sha256=row["raw_sha256"],
                 measured=measured,hard_gates=row["hard_gates"])
        records.append(rec)
        if not row["observer"]:independent.append(rec)
    return dict(version="v0.1.118",protocol_sha256=sha(root/"configs/v118_frozen_matrix.json"),
        source_master_sha256=sha(path),actual_physx_executions=len([x for x in records if x.get("measured")]),
        actual_200hz_g1_frames=sum(x["measured"]["physical_rows"] for x in records if x.get("measured")),
        independent_cells=len(independent),
        independent_strict_pass=sum(r["hard_gates"]["hard_gate_pass"] for r in independent),
        all_runs_including_parity_strict_pass=sum(x["hard_gates"]["hard_gate_pass"] for x in records if x.get("measured")),
        post_contact_censored_count=sum(x["measured"]["post1250_censored"] for x in records if x.get("measured")),
        native_collider_pair_geometry_certified=False,qualified_elite_human_kick_GRFCOP=False,
        learned_wholebody_motor_policy_actuated=False,episodes=records)

if __name__=="__main__":
    result=audit()
    target=ROOT/"reports/v118_phase_power_chain.audit.json"
    with target.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V118_PHASE_POWER",json.dumps({k:v for k,v in result.items() if k!="episodes"}),flush=True)
