#!/usr/bin/env python3
"""Paired measured all-29 original G1 qdot onset; 200Hz actual, no naturalness claims."""
import json,numpy as np
from pathlib import Path
from v113_phase_biomechanics import load_physical_arrays
from v118_independent_200hz_audit import sha
ROOT=Path(__file__).resolve().parents[1]

def onset(ref,alt,threshold=1e-7):
    t0=np.asarray(ref["t"],float);t1=np.asarray(alt["t"],float)
    n=min(len(t0),len(t1))
    if n==0 or not np.array_equal(t0[:n],t1[:n]):
        raise ValueError("DIFFERENT_REAL_PHYSICS_TIME_GRID")
    a=np.asarray(ref["vel"],float);b=np.asarray(alt["vel"],float)
    if a.shape[1:]!=(29,) or b.shape[1:]!=(29,):
        raise ValueError("NOT_29DOF")
    diff=np.max(np.abs(a[:n]-b[:n]),axis=1)
    active=np.flatnonzero(diff>threshold)
    return dict(measured_first_29dof_qdot_diff_s=float(t0[active[0]]) if len(active) else None,
        common_real_200hz_rows=n,max_measured_qdot_difference_radps=float(diff.max()),
        comparison_after_shorter_trace_is_censored=bool(len(t0)!=len(t1)))

def audit(root=ROOT):
    batch=json.loads((root/"reports/v118_original_g1_batch.audit.json").read_text())
    decoded={}
    for row in batch["runs"]:
        if not row["certified_original_G1_cuda_200Hz"] or row["observer"]:continue
        p=root/"reports"/("v118_"+row["name"]+".json")
        if sha(p)!=row["raw_sha256"]:raise ValueError("MISMATCH_RAW_SHA")
        raw=json.loads(p.read_text())
        raw["joint_velocity_limits"]=np.asarray(raw["joint_velocity_limits"],float).reshape(-1).tolist()
        decoded[(row["scene"],row["control"])]=load_physical_arrays(raw)
    contrasts=[]
    for scene in ("AM","AN"):
        for comp in ("cmu02","cmu05"):
            a=decoded[(scene,"v32")];b=decoded[(scene,comp)]
            v=onset(a,b)
            impact=float(b["impact"])
            v.update(scene=scene,comparator=comp,
                real_ball_contact_s=impact,
                difference_lead_before_impact_s=(float(impact-v["measured_first_29dof_qdot_diff_s"])
                 if v["measured_first_29dof_qdot_diff_s"] is not None else None))
            contrasts.append(v)
    return dict(version="v0.1.118",source_master_sha256=sha(root/"reports/v118_original_g1_batch.audit.json"),
                independent_controller_pairs=len(contrasts),
                positive_early_full_approach_coordination_proven=False,
                nonzero_velocity_difference_is_not_human_coordinated_learning=True,
                contrasts=contrasts)
if __name__=="__main__":
    result=audit()
    with (ROOT/"reports/v118_earliest_controller_effect.audit.json").open("x") as f:
        json.dump(result,f,indent=2);f.write("\n")
    print("V118_ONSET_AUDIT",json.dumps(result),flush=True)
