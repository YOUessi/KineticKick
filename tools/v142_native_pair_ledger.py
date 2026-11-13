#!/usr/bin/env python3
"""v142: fail-closed review of native PhysX collider pair provenance.

This is NOT a simulation and CPU packets cannot attest physical contact.
Real admission needs independently verified live runtime, calibrated event
clock, real USD owner map, isolated positive+negative controls and exact
OFF/ON physical trajectories. Never change a controller or USD stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from collections import Counter

import v134_native_contact_vector_gate as native

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v142_frozen_native_admission_and_human_eligibility.json"
FROZEN_SHA = "e9cc197bd1e26615c9596f2456caa22312595366"
TARGET = frozenset(("left_wrist_yaw_link", "left_hip_roll_link"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen():
    actual = CONFIG.read_bytes()
    original = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", FROZEN_SHA + ":" + str(CONFIG.relative_to(ROOT))]
    )
    if actual != original:
        raise ValueError("V142_FROZEN_PROTOCOL_DRIFT")
    cfg = json.loads(original)
    if (cfg["version"] != "v0.1.142"
            or cfg["parent_sha"] != "4edf26fd332b3df03e4e85007d64f5056796ca38"
            or cfg["physics_dt_s"] != .005 or cfg["original_g1_dof"] != 29
            or cfg["original_cases"] != {"frozen_v32": 2800, "cmu_10_02": 2800, "cmu_10_05": 1028}
            or not cfg["immutable_no_overwrite"]
            or cfg["allow_sim_when_foreign_gpu_process"]
            or cfg["source_authorization_from_cpu_fixture"]
            or cfg["strict_kick_gates"]["instep_impulse_fraction_min"] != .95):
        raise ValueError("V142_FROZEN_ASSUMPTIONS_DRIFT")
    return cfg


def safe_write_once(path, value):
    path = Path(path)
    if path.is_symlink() or not path.parent.is_dir():
        raise ValueError("UNSAFE_EVIDENCE_PATH")
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")
    return sha256(path)


def verify_original_state(state):
    if not isinstance(state, dict) or set(state) != set(frozen()["physical_fields"]):
        raise ValueError("INCOMPLETE_REAL_200HZ_STATE")
    native.validate_original_g1_physics_row(state)
    return state["step"]


def collider_short_name(owner_path):
    if not isinstance(owner_path, str):
        return None
    native.canonical_usd_path(owner_path)
    return owner_path.rsplit("/", 1)[-1]


def packet_review(packet, state, owner_map, *, expected_stage_id):
    """Review full copied manifold and same-step 29DoF data.

    Matching a packet to an input state proves only consistency OF INPUTS,
    not that any input came from an actual PhysX runtime.
    """
    verify_original_state(state)
    if not isinstance(packet, dict) or not isinstance(packet.get("headers"), list):
        raise ValueError("MALFORMED_NATIVE_PACKET")
    if packet.get("source_claim") != "CPU_OR_LIVE_CALLBACK_UNATTESTED":
        raise ValueError("SPOOFED_NATIVE_SOURCE_CLAIM")
    if packet.get("motor_execution_authorized") is not False or packet.get("native_pair_certified") is not False:
        raise ValueError("SELF_ATTESTED_NATIVE_CONTACT_REJECTED")
    if packet.get("all_contact_data_consumed") is not True:
        raise ValueError("UNCOVERED_NATIVE_CONTACT_DATA")
    if packet.get("stage_id") != expected_stage_id:
        raise ValueError("DIFFERENT_NATIVE_USD_STAGE")
    if packet.get("step_hint") != state["step"]:
        raise ValueError("CALLBACK_AND_PHYSICS_STEP_OFF_BY_ONE")
    if packet.get("time_hint_s") != round(.005 * state["step"], 9):
        raise ValueError("WRONG_REAL_200HZ_CLOCK")
    if not isinstance(owner_map, dict) or not owner_map:
        raise ValueError("RUNTIME_STAGE_OWNER_MAP_MISSING")
    if packet.get("total_headers") != len(packet["headers"]) or packet.get("total_points") != sum(
        len(h.get("manifold_points", [])) for h in packet["headers"]
    ):
        raise ValueError("NATIVE_POINT_COUNTS_INCONSISTENT")
    reviewers = native.review_colliders(packet, read_only_usd_owner_map=owner_map)
    candidates = []
    rejected = Counter()
    for header, check in zip(packet["headers"], reviewers):
        if header["stage_id"] != expected_stage_id:
            raise ValueError("HEADER_WRONG_STAGE")
        if check["review_status"] != "UNATTESTED_MATCH":
            rejected[check["review_status"]] += 1
            continue
        owners = check["owner_paths"]
        if frozenset(collider_short_name(x) for x in owners) != TARGET:
            continue
        if header["event_type"] not in ("CONTACT_FOUND", "CONTACT_PERSISTS"):
            rejected["NOT_ACTIVE_CONTACT_EVENT"] += 1
            continue
        if not header["manifold_points"]:
            rejected["NO_MANIFOLD_POINTS"] += 1
            continue
        norm = 0.0
        for p in header["manifold_points"]:
            v = p["impulse_world_vector_Ns"]
            if len(v) != 3 or not all(math.isfinite(float(a)) for a in v):
                raise ValueError("MALFORMED_PHYSX_VECTOR_IMPULSE")
            norm += p["impulse_vector_magnitude_Ns"]
        if norm <= 0:
            rejected["ZERO_VECTOR_IMPULSE"] += 1
            continue
        candidates.append({
            "step": state["step"],
            "collider_paths": list(header["collider_paths"]),
            "owner_paths": list(owners),
            "contact_points": len(header["manifold_points"]),
            "total_vector_magnitude_Ns": norm,
            "provenance": "INPUT_PACKET_ONLY_NO_NATIVE_ATTESTATION",
            "pair_physx_truth_certified": False,
        })
    return {
        "step": state["step"],
        "native_callback_headers": packet["total_headers"],
        "all_manifest_points": packet["total_points"],
        "left_wrist_hip_colliding_pair_candidates": candidates,
        "rejected_headers_by_reason": dict(rejected),
        "measured_original_29dof_input_dimensions": 29,
        "provenance_calibration_missing": True,
        "runtime_native_pair_certified": False,
        "motor_execution_authorized": False,
        "strict_kick_pass": False,
    }


def review_timeline(packets, physics_rows, stage_owner_map, *, stage_id):
    """One state per step, arbitrary number of callbacks per step; never
    equate number of callback invocations with number of 200Hz physics steps.
    """
    rows = list(physics_rows)
    times = [verify_original_state(x) for x in rows]
    if not times or times != list(range(1, len(rows) + 1)):
        raise ValueError("MISSING_OR_DUPLICATE_200HZ_PHYSICS_STEP")
    grouped = {}
    for pkt in packets:
        s = pkt.get("step_hint")
        if not isinstance(s, int) or s not in times:
            raise ValueError("CALLBACK_NOT_ATTACHED_TO_PHYSICAL_STEP")
        grouped.setdefault(s, []).append(pkt)
    reviews = []
    for step, data in grouped.items():
        for item in data:
            reviews.append(packet_review(item, rows[step - 1], stage_owner_map,
                                         expected_stage_id=stage_id))
    return {
        "actual_input_physics_step_count": len(rows),
        "callback_packet_count": len(reviews),
        "callback_is_not_physics_tick": True,
        "target_native_contact_candidates": sum(len(x["left_wrist_hip_colliding_pair_candidates"]) for x in reviews),
        "no_callback_does_not_prove_no_contact": True,
        "post_contact_recovery_if_early_terminated": "UNKNOWN",
        "native_pair_certified": False,
        "all_strict_kick_gates_pass": False,
        "per_callback_reviews": reviews,
    }


def paired_exact_cpu_review(off_rows, on_rows):
    """Equality of real INPUT rows; source authenticity NOT attested."""
    return native.exact_off_on_physics_equality(
        off_rows, on_rows, required_fields=frozen()["physical_fields"])


def historical_recheck():
    result = native.review_historical_original_v54()
    expected = frozen()["original_cases"]
    if result["cases"] != expected or result["historical_real_200hz_frames_sha_verified"] != 6628:
        raise ValueError("SHA_LOCKED_HISTORICAL_PHYSX_DRIFT")
    return {
        "historical_physics_episodes": 3,
        "historical_physics_5ms_frames": 6628,
        "cases": result["cases"],
        "cmu_10_05_last_actual_q_s": result["failed_last_real_q_time_s"],
        "cmu_10_05_sensor_event_s": result["failed_named_contact_sensor_time_s"],
        "cmu_10_05_nonfoot_peak_s": result["failed_real_nonfoot_peak_time_s"],
        "native_physx_pair_manifold_truth": "UNKNOWN",
        "new_physx_episodes": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    cfg = frozen()
    result = {"version": cfg["version"], "frozen_protocol_sha": FROZEN_SHA,
              "historical": historical_recheck(),
              "runtime_native_provenance_calibrated": False,
              "new_isaaclab_runs": 0,
              "strict_physical_success": None}
    s = safe_write_once(a.output, result)
    print(json.dumps({"output": str(a.output), "sha256": s, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
