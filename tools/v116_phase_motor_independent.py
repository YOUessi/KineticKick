#!/usr/bin/env python3
"""Independent SHA-pinned true G1 200Hz phase mechanics; no simulated future rows."""
import json
import numpy as np
from pathlib import Path
from v113_phase_biomechanics import load_physical_arrays, phase_metrics, paired_controller_effect, sha256

ROOT=Path(__file__).resolve().parents[1]

def audit(root=ROOT):
    master=root/"reports/v116_original_g1_batch.audit.json"
    batch=json.loads(master.read_text())
    if batch.get("version")!="v0.1.116":raise ValueError("WRONG_SOURCE_BATCH")
    records=[]; controls={}
    for row in batch["runs"]:
        if not row.get("certified_original_G1_cuda_200Hz"):continue
        path=root/"reports"/("v116_"+row["name"]+".json")
        if sha256(path)!=row["raw_sha256"]:raise ValueError("PHYSICS_SHA_DRIFT")
        raw=json.loads(path.read_text())
        raw["joint_velocity_limits"]=np.asarray(raw["joint_velocity_limits"],float).reshape(-1).tolist()
        measured=load_physical_arrays(raw)
        summary=phase_metrics(measured)
        records.append(dict(name=row["name"],scene=row["scene"],control=row["control"],
             observer_repeat=row["observer"],sha256=row["raw_sha256"],
             hard_gate_pass=row["hard_gates"]["hard_gate_pass"],
             failed_gates=row["hard_gates"]["failed_gates"],phases=summary))
        if not row["observer"]:controls[(row["scene"],row["control"])]=measured
    contrasts=[]
    for scene in ("AG","AH","AI"):
        if (scene,"v32") not in controls:continue
        for ctl in ("cmu02","cmu05"):
            if (scene,ctl) in controls:
                contrasts.append(dict(scene=scene,comparator=ctl,
                    original_common_200hz_physics=paired_controller_effect(
                        controls[(scene,"v32")],controls[(scene,ctl)])))
    return dict(version="v0.1.116",source_master_sha256=sha256(master),
        new_scored_original_29dof_g1_runs=len(records),
        independent_conditions=len(controls),
        strict_pass_independent=sum(r["hard_gate_pass"] for r in records if not r["observer_repeat"]),
        actual_real_200hz_steps=sum(r["phases"]["physics_rows"] for r in records),
        full_14s_observed=sum(r["phases"]["actual_14s_recovery_coverage"] for r in records),
        true_actor_collider_contact_manifold_certified=False,
        human_grf_cop_impact_impulse_certified=False,
        human_trained_motor_policy_actuated=False,
        runs=records,paired=contrasts)

if __name__=="__main__":
    obj=audit()
    path=ROOT/"reports/v116_phase_motor_independent.audit.json"
    with path.open("x") as f:json.dump(obj,f,indent=2);f.write("\n")
    print("V116_TRUE_PHASE_MOTOR",json.dumps({k:v for k,v in obj.items() if k not in ("runs","paired")}))
