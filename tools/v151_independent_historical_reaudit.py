#!/usr/bin/env python3
"""V151 independent read-only re-audit of original v53 12 and v54 3 physics runs.

Input digests pinned by original 2026-11-11 v54 experiment protocol.
Recomputes strict conjunction from every recorded v53 gate, never runs PhysX.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from v151_native_campaign import SOURCE_ROOT,frozen,source_inventory
from v142_native_pair_ledger import historical_recheck,safe_write_once
from v146_real_200hz_coupled_audit import verified_sources,split_real_5ms_rows


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def independently_reaudit():
    study=frozen()
    verified=source_inventory()
    protocol=json.loads((SOURCE_ROOT/"configs/cmu_named_nonfoot_diagnostics_v054.json").read_text())
    audit_path=SOURCE_ROOT/"reports/v053_cmu_frozen_matrix_audit.json"
    expected_sha=protocol["immutable_sha256"]["reports/v053_cmu_frozen_matrix_audit.json"]
    if sha(audit_path)!=expected_sha:
        raise ValueError("V151_V53_TWELVE_RUN_AUDIT_SHA_CHANGED")
    original=json.loads(audit_path.read_text())
    if original.get("n_original_PhySX_runs")!=12 or original.get("n_ball_conditions")!=4:
        raise ValueError("V151_PRIOR_REAL_PHYSICS_MATRIX_INCOMPLETE")
    failures={name:Counter() for name in study["policies"]}
    gate_names={}
    success={}
    for name in study["policies"]:
        recorded=original["summary"][name]
        trials=[c["policies"][name] for c in original["comparisons"]]
        if len(trials)!=4:
            raise ValueError("V151_HISTORICAL_CASE_COUNT_DRIFT")
        gate_names[name]=sorted(trials[0]["gates"])
        for trial in trials:
            gates=trial["gates"]
            if set(gates)!=set(gate_names[name]) or any(type(v) is not bool for v in gates.values()):
                raise ValueError("V151_HISTORICAL_GATES_UNREGISTERED")
            if all(gates.values()) is not trial["success"]:
                raise ValueError("V151_HISTORICAL_SUCCESS_GATE_DISAGREES")
            failures[name].update(k for k, passed in gates.items() if not passed)
        count=sum(int(x["success"]) for x in trials)
        if count!=recorded["success_count"]:
            raise ValueError("V151_HISTORICAL_STRICT_SUCCESS_DRIFT")
        success[name]={"success":count,"trials":4,
                       "strict_success_fraction":count/4,
                       "failed_gate_counts":dict(failures[name]),
                       "real_g1_physics_historical_not_new":True}
    recorded=historical_recheck()
    sources=verified_sources()
    states=0
    terminal_notes=0
    source_details=[]
    for row in verified:
        name=row["policy"]
        path=sources[name]
        data=json.loads(path.read_text())
        phys,terminal=split_real_5ms_rows(
            data["traces"][0],study["history_expected_rows"][name])
        if data["seed"]!=study["seed"]:
            raise ValueError("V151_HISTORICAL_PHYSICS_SEED_CHANGED")
        terminal_notes+=len(terminal)
        states+=len(phys)
        source_details.append({
            "policy":name,"historical_original_steps":len(phys),
            "source_sha256":sha(path),
            "terminal_non_5ms_note_count":len(terminal),
            "14s_completed":len(phys)==2800,
            "true_left_wrist_hip_native_manifold":"UNKNOWN"})
    if states!=recorded["historical_physics_5ms_frames"] or states!=6628 or terminal_notes!=1:
        raise ValueError("V151_HISTORICAL_V54_TIMELINE_CHANGED")
    return dict(
        version="v0.1.151",
        source_level_reanalysis_of_real_physx=True,
        historical_v53_physx_episodes=12,
        historical_v54_physx_episodes=3,
        historical_v54_physics_200hz_rows=states,
        excluded_terminal_duplicate_notes=terminal_notes,
        historical_v53_strict_by_policy=success,
        historical_v54_rows=source_details,
        new_gpu_physx_rollouts=0,
        strict_new_policy_success=None,
        no_native_manifold_left_wrist_hip=True,
        true_contact_collider_pair="UNKNOWN",
        simultaneous_human_grf_true_instep_supervision=False,
        notes="Existing v53/v54 PhysX verified inputs only; v151 sidecars pending actual safe GPU runtime.")


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(argv)
    out=independently_reaudit()
    digest=safe_write_once(a.output,out)
    print(json.dumps({"output":str(a.output),"sha256":digest,
         "v53_runs":12,"v54_runs":3,"v54_states":out["historical_v54_physics_200hz_rows"],
         "strict_success":{k:v["success"] for k,v in out["historical_v53_strict_by_policy"].items()},
         "new_physx":0}))


if __name__=="__main__":
    main()
