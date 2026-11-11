#!/usr/bin/env python3
"""Qualify PhysX raw-contact pathways WITHOUT promoting toy contacts to G1 truth.

Read-only, post-hoc audit of pre-existing independent 200-Hz toy probes
on GPU/CPU plus three frozen original G1 rollouts. Never edits old evidence,
never assumes zero callback implies no collision. Output is creation-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBES = {
    "gpu_collide": "reports/v056_isolate2_collide.json",
    "gpu_separated": "reports/v056_isolate2_separated.json",
    "cpu_collide": "reports/v056_isolate_cpu_collide.json",
    "cpu_separated": "reports/v056_isolate_cpu_separated.json",
}
G1 = {
    "frozen_v32": "reports/v056_center_native_frozen_v32.json",
    "cmu_10_02": "reports/v056_center_native_cmu_10_02.json",
    "cmu_10_05": "reports/v056_center_native_cmu_10_05.json",
}
DEFAULT_OUT = ROOT / "reports/v056_native_backend_qualification.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identify_probe(d: dict) -> dict:
    if d["schema"] != "v056 isolation and signed negative" or not d["not_G1_evidence"]:
        raise ValueError("Probe evidence provenance mismatch")
    if d["sim_dt_s"] != .005 or d["sensor_body_count"] != 1 or d["sensor_filter_count"] != 1:
        raise ValueError("Not measured by the expected 200Hz one-to-one filtered view")
    if d["errors"] or d["contact_tensor_errors"]:
        raise ValueError("Probe had errors")
    events = d["raw_events"]
    qualified = [
        e for e in events
        if e["count"] > 0
        and any("/World/Probe/A" in p for p in e["paths"][:2])
        and any("/World/Probe/B" in p for p in e["paths"][:2])
        and any("/World/Probe/A/geometry/mesh" in p for p in e["paths"][2:])
        and any("/World/Probe/B/geometry/mesh" in p for p in e["paths"][2:])
        and any("position" in q and "normal" in q and "impulse" in q
                for q in e["contacts"])
    ]
    return {
        "filtered_pair_peak_N": d["filtered_pair_peak_N"],
        "callback_calls": d["callbacks"],
        "polled_headers": d["polled_headers"],
        "total_events": d["total_probe_events"],
        "actor_and_collider_paths_plus_manifold_present": bool(qualified),
        "native_pair_proven_in_toy_only": bool(d["native_pair_proven"] and qualified),
    }


def identify_g1(d: dict) -> dict:
    if d.get("root_fixed") is not False or len(d.get("joint_names", [])) != 29:
        raise ValueError("Not original free root G1")
    r = d.get("physx_native_report_v056")
    if not isinstance(r, dict) or r["errors"] or r["dropped_events"]:
        raise ValueError("Invalid original G1 reporter metadata")
    return {
        "callback_calls": r["callback_calls"],
        "headers": r["native_headers"],
        "wrist_hip_path_pairs": r["wrist_hip_direct_path_events"],
        "report_api_present_on_observed_links": all(r["report_api_present_on_original_bodies"].values()),
        "pair_manifold_proven": bool(r["rigid_pair_certified"]),
        # DO NOT trust report_capture_complete if callback_calls==0.
        "contact_reporting_operational_observed": bool(r["callback_calls"] and r["native_headers"]),
        "absence_of_events_proves_no_self_collision": False,
    }


def qualify(probes: dict, g1: dict) -> dict:
    if set(probes) != set(PROBES) or set(g1) != set(G1):
        raise ValueError("Need all CPU/GPU positive/negative and all 3 G1 records")
    for key in ("gpu_collide", "cpu_collide"):
        if probes[key]["filtered_pair_peak_N"] <= 20:
            raise ValueError("Positive control did not actually collide")
    for key in ("gpu_separated", "cpu_separated"):
        if probes[key]["filtered_pair_peak_N"] > 1e-6:
            raise ValueError("Negative control produced contact")
    return {
        "gpu_reporter_false_negative_for_verified_toy_contact": (
            probes["gpu_collide"]["callback_calls"] == 0
            and probes["gpu_collide"]["polled_headers"] == 0
            and not probes["gpu_collide"]["native_pair_proven_in_toy_only"]),
        "cpu_toy_pair_paths_manifold_positive": probes["cpu_collide"]["native_pair_proven_in_toy_only"],
        "cpu_negative_control_clear": (
            probes["cpu_separated"]["total_events"] == 0
            and probes["cpu_separated"]["filtered_pair_peak_N"] == 0),
        "original_gpu_G1_pair_proven": any(x["pair_manifold_proven"] for x in g1.values()),
        "original_gpu_G1_native_events_seen": any(x["contact_reporting_operational_observed"] for x in g1.values()),
        "G1_collision_pair_inference": "UNDETERMINED_NOT_A_NEGATIVE",
        "CPU_toy_truth_does_not_transfer_to_GPU_G1": True,
        "recommendation": (
            "Do not enable unverified whole-body collision guard. "
            "Only use a stage-read-only method with certified per-pair actor/collider paths "
            "or design separately pre-registered CPU-backend G1 parity experiments.")
    }


def run(out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    probes, g1, source_hashes = {}, {}, {}
    for name, rel in PROBES.items():
        src = ROOT / rel
        source_hashes[rel] = sha(src)
        probes[name] = identify_probe(json.loads(src.read_text()))
    for name, rel in G1.items():
        src = ROOT / rel
        source_hashes[rel] = sha(src)
        g1[name] = identify_g1(json.loads(src.read_text()))
    verdict = qualify(probes, g1)
    result = {"version": "0.1.56",
              "method": "read-only post-hoc evidence qualification; not a fitted control policy",
              "sources_sha256": source_hashes, "toy_probes": probes,
              "original_G1_physx_reporter": g1, "verdict": verdict}
    with out.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("V056_CONTACT_BACKEND_QUALIFICATION", json.dumps(verdict), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    run(p.parse_args().output)
