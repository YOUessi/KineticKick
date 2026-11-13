#!/usr/bin/env python3
"""Independent schema/clock/denominator audit of v149 CPU fixtures.

Intentionally does NOT import or call the capture engine or matrix generator.
A pass certifies only internal consistency of the synthetic event matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from v142_native_pair_ledger import safe_write_once

POLICIES = ("frozen_v32", "cmu_10_02", "cmu_10_05")
PHASES = (
    "BEFORE_FIRST_SCENE_UPDATE", "INSIDE_SCENE_UPDATE",
    "BETWEEN_SCENE_UPDATES", "AFTER_TERMINAL_SCENE_UPDATE",
)
KINDS = ("VECTOR_CONTACT", "INVALID_SCALAR_IMPULSE")


def independent_review(document):
    if document.get("version") != "v0.1.149":
        raise ValueError("V149_AUDIT_SOURCE_VERSION_WRONG")
    if (document.get("new_real_physx_episodes") != 0
            or document.get("runtime_physx_wrist_hip_pair_truth") != "UNKNOWN"
            or document.get("runtime_off_on_parity") != "NOT_RUN"):
        raise ValueError("V149_SYNTHETIC_RUNTIME_ATTESTATION_SPOOF")
    hist = document.get("historical_real_physx_not_rerun")
    if (not isinstance(hist, dict)
            or hist.get("historical_physics_5ms_frames") != 6628
            or hist.get("cases") != {
                "frozen_v32": 2800, "cmu_10_02": 2800, "cmu_10_05": 1028
            }):
        raise ValueError("V149_HISTORICAL_REAL_TRACE_REFERENCE_DRIFT")
    cells = document.get("synthetic_cells")
    if not isinstance(cells, list) or len(cells) != 48:
        raise ValueError("V149_SYNTHETIC_DENOMINATOR_CHANGED")
    expected = set(itertools.product(POLICIES, PHASES, KINDS, (False, True)))
    seen = set()
    copied_valid, invalid_rejected = 0, 0
    total_callback = 0
    for x in cells:
        key = (
            x["policy_name_is_only_a_fixture_label"], x["phase"],
            x["header_kind"], x["existing_report_api_fixture"]
        )
        if key not in expected or key in seen:
            raise ValueError("V149_DUPLICATE_OR_UNDECLARED_MATRIX_CASE")
        seen.add(key)
        wanted = 1 if key[2] == "VECTOR_CONTACT" else 0
        if x["valid_packets_preserved"] != wanted:
            raise ValueError("V149_EVIDENCE_COPIER_MISSED_VALID_EVENT")
        if x["callback_invocations"] != 1:
            raise ValueError("V149_CALLBACK_COUNTER_NOT_ONE_PER_FIXTURE")
        if x["arrival_phase_counts"] != {key[1]: 1}:
            raise ValueError("V149_CALLBACK_PHASE_ACCOUNTING_WRONG")
        if x["schema_eligible_in_fixture"] is not key[3]:
            raise ValueError("V149_SCHEMA_ELIGIBILITY_NOT_SOURCE_PRESERVING")
        if x["invalid_callback_flagged"] is not (wanted == 0):
            raise ValueError("V149_INVALID_NATIVE_IMPULSE_SILENTLY_ACCEPTED")
        if (x["source_is_runtime"] is not False or
                x["same_native_physics_tick_calibrated"] is not False or
                x["native_wrist_hip_contact_certified"] is not False or
                x["strict_kick_success"] is not None):
            raise ValueError("V149_MOCK_PROMOTED_TO_REAL_WORLD_PROOF")
        if not isinstance(x["simulated_rows_count"], int) or not 1 <= x["simulated_rows_count"] <= 2:
            raise ValueError("V149_INVALID_SYNTHETIC_STATE_LENGTH")
        if wanted:
            c = x["step_candidates_unattested"]
            if not isinstance(c, list) or len(c) not in (1,2):
                raise ValueError("V149_LOST_ADJACENT_STEP_CANDIDATES")
            if not all(type(s) is int and 1 <= s <= x["simulated_rows_count"]+1 for s in c):
                raise ValueError("V149_FAKE_OR_INVALID_STEP_INDEX")
            if x["valid_event_path_pair_candidates_unattested"] != 1:
                raise ValueError("V149_VALID_VECTOR_AND_OWNER_PATHS_MISSING")
        elif x["step_candidates_unattested"] or x["valid_event_path_pair_candidates_unattested"]:
            raise ValueError("V149_BAD_IMPULSE_PROMOTED_TO_RAW_PACKET")
        copied_valid += wanted
        invalid_rejected += 1 - wanted
        total_callback += x["callback_invocations"]
    if seen != expected or copied_valid != 24 or invalid_rejected != 24:
        raise ValueError("V149_EXHAUSTIVE_MATRIX_DENOMINATOR_MISMATCH")
    if (document.get("synthetic_case_count") != 48
            or document.get("synthetic_expected_valid_packets") != 24
            or document.get("synthetic_observed_valid_packets") != copied_valid
            or document.get("invalid_vector_payloads_rejected") != invalid_rejected):
        raise ValueError("V149_SUMMARY_COUNTER_FALSE")
    source = document.get("runner_source_checks")
    if (not isinstance(source, dict)
            or not source.get("legacy_only_scene_update_callback_bracket")
            or source.get("runtime_src_and_positive_negative_calibration") is not False
            or any(not isinstance(source.get(k), str) or len(source[k]) != 64
                   for k in ("v146_runner_sha256", "v149_runner_sha256",
                             "original_v080_runner_sha256"))):
        raise ValueError("V149_RUNNER_SOURCE_AUDIT_INVALID")
    return {
        "synthetic_cells_independently_checked": 48,
        "valid_vec3_copies_preserved": copied_valid,
        "malformed_scalar_rejections": invalid_rejected,
        "callback_invocations": total_callback,
        "historical_sha_verified_200hz_rows_referenced": 6628,
        "true_new_physx_episodes": 0,
        "native_pair_manifold_certified": False,
        "source_only_48_cell_gate": True,
        "runtime_positive_negative_controls_still_needed": True,
        "independent_audit_is_not_physical_attestation": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    result = independent_review(data)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    digest = safe_write_once(args.output, result)
    print(json.dumps({"audit_sha256":digest, **result}))


if __name__ == "__main__":
    main()
