#!/usr/bin/env python3
"""Original 200 Hz G1 first-qdot-hazard vs measured controller intervention.

Read-only audit of SHA-pinned PhysX data. No fictitious motor policy, human
GRF/CoP, contact manifold or recovery after censoring.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from v116_200hz_coupled_audit import group_indices, strip_terminal_event_only, require_200hz

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-7

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def physical_arrays(raw):
    if raw.get("root_fixed") is not False:
        raise ValueError("NOT_ORIGINAL_FREE_ROOT_G1")
    names = raw["joint_names"]
    groups = group_indices(names)
    tr, nterminal = strip_terminal_event_only(raw["traces"][0])
    t = require_200hz(tr)
    v = np.array([r["joint_vel"] for r in tr], dtype=float)
    q = np.array([r["joint_pos"] for r in tr], dtype=float)
    lim = np.asarray(raw["joint_velocity_limits"], dtype=float).reshape(-1)
    if q.shape != (len(t), 29) or v.shape != (len(t), 29) or lim.shape != (29,):
        raise ValueError("INVALID_29DOF_PHYSICS_SHAPE")
    if not all(np.isfinite(x).all() for x in (q, v, lim)) or np.any(lim <= 0):
        raise ValueError("NONFINITE_OR_NONPOSITIVE_SOURCE_QDOT_LIMIT")
    contact = float(raw["records"][0]["contact_time_s"])
    if not np.isfinite(contact) or not t[0] <= contact <= t[-1]+0.005001:
        raise ValueError("INVALID_PHYSICAL_CONTACT_TIME")
    return dict(names=names, groups=groups, t=t, q=q, v=v, lim=lim,
                contact=contact, terminal_markers=nterminal)

def violations(a):
    ratio = np.abs(a["v"]) / a["lim"][None, :]
    bad = np.argwhere(ratio > 1 + 1e-7)
    bad_joints = np.flatnonzero((ratio > 1 + 1e-7).any(axis=0))
    first = int(bad[0,0]) if len(bad) else None
    individual = []
    for j in bad_joints:
        k = int(np.argmax(ratio[:,j]))
        individual.append(dict(joint=a["names"][j], first_exceed_s=float(a["t"][np.flatnonzero(ratio[:,j] > 1+1e-7)[0]]),
                   peak_ratio=float(ratio[k,j]), peak_time_s=float(a["t"][k]),
                   peak_actual_qdot_radps=float(a["v"][k,j]),
                   frozen_source_limit_radps=float(a["lim"][j])))
    individual.sort(key=lambda x:-x["peak_ratio"])
    return dict(first_true_qdot_exceed_s=float(a["t"][first]) if first is not None else None,
                first_exceed_before_ball_contact=bool(first is not None and a["t"][first] < a["contact"]),
                max_ratio_all_original29=float(np.max(ratio)), all_exceeding_joints=individual,
                count_original_joints_over_limit=int(len(bad_joints)),
                no_violation_imputed_after_termination=True)

def paired_intervention(base, cmp):
    if base["names"] != cmp["names"]:
        raise ValueError("SOURCE_G1_JOINT_ORDER_MISMATCH")
    n = min(len(base["t"]),len(cmp["t"]))
    if n < 100 or not np.array_equal(base["t"][:n],cmp["t"][:n]):
        raise ValueError("PHYSICAL_200HZ_TIMES_NOT_ALIGNED")
    t = base["t"][:n]
    delta = cmp["v"][:n] - base["v"][:n]
    measure = np.max(np.abs(delta),axis=1)
    nz = np.flatnonzero(measure > EPS)
    onset = float(t[nz[0]]) if len(nz) else None
    va, vb = violations(base),violations(cmp)
    firsts = [x for x in (va["first_true_qdot_exceed_s"],
                          vb["first_true_qdot_exceed_s"]) if x is not None]
    hazard = min(firsts) if firsts else None
    pre_contact = t < min(base["contact"],cmp["contact"])
    return dict(actual_shared_200hz_steps=n,
                first_observed_whole29_qdot_difference_s=onset,
                difference_lead_to_first_contact_s=(float(min(base["contact"],cmp["contact"]) - onset)
                           if onset is not None else None),
                first_excess_qdot_any_controller_s=hazard,
                excess_happened_before_first_measured_controller_difference=bool(
                    hazard is not None and (onset is None or hazard < onset-0.000001)),
                before_impact_qdot_delta_rms_radps=float(np.sqrt(np.mean(delta[pre_contact]**2))) if pre_contact.any() else None,
                group_first_affected_s={
                   g:(float(t[np.flatnonzero(np.max(np.abs(delta[:,inds]),axis=1)>EPS)[0]])
                       if np.any(np.max(np.abs(delta[:,inds]),axis=1)>EPS) else None)
                    for g,inds in base["groups"].items()},
                both_have_real_14s=bool(base["t"][-1]>=13.995 and cmp["t"][-1]>=13.995),
                native_contact_actor_collider_manifold_certified=False,
                pre_existing_safety_violation_not_retroactively_fixable=True)

def audit():
    master=ROOT/"reports/v116_original_g1_batch.audit.json"
    batch=json.loads(master.read_text())
    if batch.get("version") != "v0.1.116":
        raise ValueError("NOT_THIS_FROZEN_BATCH")
    entries=[]; seen={}
    for row in batch["runs"]:
        if not row["certified_original_G1_cuda_200Hz"]:
            continue
        file=ROOT/"reports"/("v116_"+row["name"]+".json")
        if sha256(file) != row["raw_sha256"]:
            raise ValueError("CERTIFIED_RAW_G1_SHA256_DRIFT:"+row["name"])
        arr=physical_arrays(json.loads(file.read_text()))
        item=dict(name=row["name"], scene=row["scene"], control=row["control"],
                  repeat_observer=row["observer"], original_raw_sha256=row["raw_sha256"],
                  true_200hz_frames=len(arr["t"]), terminal_not_a_real_physics_step=arr["terminal_markers"],
                  observed_time_end_s=float(arr["t"][-1]),
                  observed_14s_recovery=bool(arr["t"][-1]>=13.995),
                  post_1p25s_observed=bool(arr["t"][-1]>=arr["contact"]+1.25-1e-6),
                  all_strict_hard_gates_pass=row["hard_gates"]["hard_gate_pass"],
                  failed_gates=row["hard_gates"]["failed_gates"], qdot=violations(arr),
                  cannot_claim_human_grf_cop=True)
        entries.append(item)
        if not row["observer"]:
            seen[(row["scene"],row["control"])]=arr
    pairs=[]
    for scene in ("AG","AH","AI"):
        if (scene,"v32") not in seen: continue
        for control in ("cmu02","cmu05"):
            if (scene,control) in seen:
                pairs.append(dict(scene=scene, baseline="v32", comparator=control,
                             measurements=paired_intervention(seen[(scene,"v32")],seen[(scene,control)])))
    return dict(version="v0.1.116", source_batch_sha256=sha256(master),
            original_safety_risk_precedes_instrumented_human_reference_count=sum(
                x["measurements"]["excess_happened_before_first_measured_controller_difference"] for x in pairs),
            pairs_with_observed_original_qdot_violation=sum(
                x["measurements"]["first_excess_qdot_any_controller_s"] is not None for x in pairs),
            measured_independent_original_shots=sum(not x["repeat_observer"] for x in entries),
            actual_new_physx_rows=sum(x["true_200hz_frames"] for x in entries),
            native_manifold_certified=False, qualified_human_force_truth=False,
            learned_policy_actuation=False, per_episode=entries,
            independent_matched_controller_pairs=pairs)

def main():
    dest=ROOT/"reports/v116_first_qdot_causality.audit.json"
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("WRITE_ONCE_QDOT_AUDIT")
    obj=audit()
    with dest.open("x") as f: json.dump(obj,f,indent=2); f.write("\n")
    print("V116_TRUE_QDOT_CAUSALITY", json.dumps({k:v for k,v in obj.items()
                if k not in ("per_episode","independent_matched_controller_pairs")}), flush=True)

if __name__=="__main__":
    main()
