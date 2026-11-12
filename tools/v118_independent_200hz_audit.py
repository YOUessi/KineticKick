#!/usr/bin/env python3
"""Second independent 29DoF original G1 raw-physics validator for v118.

This uses a different 200Hz decoder from the phase-power verifier, checks raw
SHA before analysis, never extrapolates censored physics or human GRF/CoP,
and never relaxes the original strict hard-gate scoring.
"""
from __future__ import annotations
from pathlib import Path
import json, numpy as np
from v116_200hz_coupled_audit import analyze,sha
from v113_phase_biomechanics import load_physical_arrays
ROOT=Path(__file__).resolve().parents[1]

def validate_record(root, row):
    rawpath=root/"reports"/("v118_"+row["name"]+".json")
    if sha(rawpath)!=row["raw_sha256"]:
        raise ValueError("SECOND_PARSER_RAW_SHA_MISMATCH")
    raw=json.loads(rawpath.read_text())
    if not row["certified_original_G1_cuda_200Hz"]:
        raise ValueError("PHYSICS_UNCERTIFIED")
    independent=analyze(raw,certified=True)
    flat=dict(raw,joint_velocity_limits=np.asarray(raw["joint_velocity_limits"],dtype=float).reshape(-1).tolist())
    other=load_physical_arrays(flat)
    t=other["t"];ratio=np.abs(other["vel"])/other["lim"][None,:]
    if independent["count_real_200hz"]!=row["actual_traced_200hz_rows"]:
        raise ValueError("REAL_200HZ_COUNT_DISAGREES")
    if abs(float(ratio.max())-independent["qdot_peak_all_original29_ratio"])>1e-9:
        raise ValueError("TWO_INDEPENDENT_QDOT_DECODERS_DISAGREE")
    gate=row["hard_gates"]
    if abs(float(ratio.max())-float(gate["original_29_qdot_max_ratio"]))>1e-5:
        raise ValueError("ORIGINAL_GATES_JOINT_RATIO_DISAGREES")
    return dict(name=row["name"],control=row["control"],scene=row["scene"],
        observer=row["observer"],raw_sha256=row["raw_sha256"],
        true_200hz_steps=independent["count_real_200hz"],
        strict_original_gate_pass=gate["hard_gate_pass"],
        original_gate_failures=gate["failed_gates"],
        true_contact_time_s=independent["contact_s"],
        actual_29_qdot_peak_to_limit_ratio=independent["qdot_peak_all_original29_ratio"],
        original_goal_entered=gate["original_goal_entered"],
        original_instep_impulse_share=gate["measured_instep_fraction"],
        original_ball_speed_proxy_mps=gate["ball_speed_peak_mps_post_5_to_110ms_proxy"],
        real_full_pre400=independent["has_full_pre_0p4_s"],
        real_full_post1250=independent["has_full_post_1p25_s"],
        real_full14=independent["full_14s"],
        first_nonfoot_ge20N_s=independent["first_nonfoot_ge20N_s"],
        true_pre400_summary=independent["pre_contact"],
        true_post1250_summary=independent["post_contact_observed_only"],
        native_collider_identity_certified=False,
        original_instep_and_recovery_scored_independently=True)

def audit(root=ROOT):
    master=root/"reports/v118_original_g1_batch.audit.json"
    b=json.loads(master.read_text())
    if b["version"]!="v0.1.118":raise ValueError("WRONG_BATCH")
    records=[]
    for r in b["runs"]:
        if r["certified_original_G1_cuda_200Hz"]:
            records.append(validate_record(root,r))
    independent=[r for r in records if not r["observer"]]
    return dict(version="v0.1.118",source_master_sha256=sha(master),
        total_certified_g1_runs=len(records),
        observed_200hz_frames=sum(r["true_200hz_steps"] for r in records),
        independent_scene_controller_cells=len(independent),
        independent_all_hard_gates_pass=sum(r["strict_original_gate_pass"] for r in independent),
        off_on_repeated_runs=len(records)-len(independent),
        true_complete_pre400=sum(r["real_full_pre400"] for r in records),
        true_complete_post1250=sum(r["real_full_post1250"] for r in records),
        true_complete_14s=sum(r["real_full14"] for r in records),
        native_actor_collider_manifold_certified=False,
        human_GRFCOP_fully_qualified=False,episodes=records)

if __name__=="__main__":
    result=audit()
    dest=ROOT/"reports/v118_independent_200hz.audit.json"
    with dest.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V118_INDEPENDENT_200HZ",json.dumps({k:v for k,v in result.items() if k!="episodes"}),flush=True)
