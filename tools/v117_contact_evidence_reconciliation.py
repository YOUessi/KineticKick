#!/usr/bin/env python3
"""Read-only provenance reconciliation of isolated PhysX positive controls and G1 Tensor.

A missing native manifold cannot be inferred safe from a zero filtered Tensor
contact. Never mutate previous v114 raw evidence, controller or model.
"""
from __future__ import annotations
import json
from pathlib import Path
from v117_causal_safety_statistics import ROOT,sha

V114=Path("/home/you/projects/kinetickick-wt-v114")


def inspect_previous_positive(root=V114):
    audit=root/"reports/v114_tensor_positive_batch.audit.json"
    b=json.loads(audit.read_text())
    if b.get("version")!="v0.1.114" or b.get("actual_certified_physx_runs",0)<3:
        raise ValueError("UNQUALIFIED_PREVIOUS_ISOLATED_CONTROLS")
    controls=[]
    for row in b["cases"]:
        name=row["name"]
        raw=root/"reports"/("v114_tensor_"+name+".json")
        if not raw.is_file():
            # The exact frozen runner stored the stem, not a fabricated alias.
            raw=root/"reports"/("v114_"+name+".json")
        if not raw.is_file():
            raise FileNotFoundError("PRIOR_CONTACT_POSITIVE_RAW_NOT_FOUND:"+name)
        if sha(raw)!=row["output_sha"]:
            raise ValueError("PREVIOUS_SOURCE_HASH_MISMATCH:"+name)
        controls.append(dict(name=name,source_sha256=row["output_sha"],
            true_physics_steps=row["actual_200hz_steps"],
            tensor_observed_contact_rows=row["tensor_contacts"],
            tensor_peak_force_N=row["positive_tensor_force"],
            native_colliders_and_manifold_certified=row["native_pair_certified"]))
    return dict(sha256=sha(audit),
        certified_isolated_physx_total=int(b["actual_certified_physx_runs"]),
        genuine_tensor_contact_events=int(b["positive_tensor_report_count"]),
        every_pair_native_manifold_certified=bool(
            controls and all(x["native_colliders_and_manifold_certified"] for x in controls)),
        controls=controls)


def inspect_g1_observer(root=ROOT):
    master=json.loads((root/"reports/v117_original_g1_batch.audit.json").read_text())
    checks=[]
    for row in master["runs"]:
        if not row.get("observer"):continue
        if not row.get("certified_original_G1_cuda_200Hz"):
            continue
        p=root/"reports"/("v117_"+row["name"]+".json")
        if sha(p)!=row["raw_sha256"]:
            raise ValueError("G1_WITNESS_SOURCE_SHA_DRIFT")
        raw=json.loads(p.read_text())
        observer=raw["g1_tensor_pair_v098"]
        paths=observer["pairs"]
        for label,data in paths.items():
            checks.append(dict(
                episode=row["name"],pair=label,
                samples=data["samples"],detected_tensor_force_N=data["peak_force_N"],
                detected_contact_points=data["observed_data_count"],
                both_source_filter_paths_present=(
                    len(data["sensor_paths_found"])==1 and
                    len(data["filter_paths_found"])==1),
                status=data["status"],
                native_identity_and_geometry_certified=False,
                zero_filtered_force_implies_no_collision=False))
    return checks


def audit():
    prior=inspect_previous_positive()
    g1=inspect_g1_observer()
    return dict(version="v0.1.117",
        previous_isolated_positive_controls=prior,
        independently_sha_verified_G1_tensor_observers=g1,
        previous_isolated_GPU_tensor_witness_calibrated=(
            prior["genuine_tensor_contact_events"]>0),
        native_actor0_actor1_collider0_collider1_contact_geometry_certified=False,
        native_wrist_hip_pair_direct_positive_witness=False,
        absence_of_self_collision_certified=False,
        reason="Tensor calibrated only for isolated ball-box pair, not for "
        "a native collider-level G1 wrist-hip pair; previous native "
        "reports lacked positive collider manifold; zero uncalibrated G1 "
        "filtered force cannot prove no G1 self collision.",
        learned_motor_action_cleared_by_contact_evidence=False)


if __name__=="__main__":
    obj=audit()
    out=ROOT/"reports/v117_contact_evidence_reconciliation.audit.json"
    with out.open("x") as f:json.dump(obj,f,indent=2);f.write("\n")
    print("V117_CONTACT_RECONCILIATION",json.dumps({
        k:v for k,v in obj.items()
        if k not in ("previous_isolated_positive_controls",
                      "independently_sha_verified_G1_tensor_observers")
    }),flush=True)
