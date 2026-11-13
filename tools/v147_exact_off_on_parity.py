#!/usr/bin/env python3
"""v147: SOURCE-LEVEL exact original29 OFF/ON sidecar parity, never live attestation.

Reject missing, duplicated and misordered 200Hz rows; compare *all* original
physical fields recursively with zero numerical tolerance, reporting the first
field divergence. CPU JSON files can never certify real runtime provenance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from v142_native_pair_ledger import safe_write_once, verify_original_state
from v146_native_original_bridge import PHYSICS_FIELDS, json_sha

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA = "3ecf094137d9f1d82e94a9dba3b9bf7d215c97c8"
CONFIG_PATH = "configs/v147_frozen_physics_parity_and_coupled_dynamics.json"


def frozen():
    expected = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                        FROZEN_SHA + ":" + CONFIG_PATH])
    if (ROOT / CONFIG_PATH).read_bytes() != expected:
        raise ValueError("V147_FROZEN_PROTOCOL_DRIFT")
    cfg = json.loads(expected)
    if (cfg["version"] != "v0.1.147" or cfg["num_joints"] != 29
            or cfg["dt_seconds"] != .005 or cfg["parity_tolerance"] != 0
            or not cfg["cpu_callback_fixture_cannot_authenticate_physx"]
            or not cfg["no_missing_5ms_rows_may_be_invented"]
            or set(cfg["required_parity_fields"]) != set(PHYSICS_FIELDS)-{"step"}):
        raise ValueError("V147_FROZEN_ASSUMPTIONS_DRIFT")
    return cfg


def _walk_diff(a, b, path):
    """First differing value with an unambiguous nested field/index address."""
    if type(a) is not type(b):
        # int/float are both legitimate numeric JSON representations.
        if (not isinstance(a, bool) and not isinstance(b, bool)
                and isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isfinite(a) and math.isfinite(b) and a == b):
            return None
        return path + ".type"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return path + ".keys"
        for k in sorted(a):
            diff = _walk_diff(a[k], b[k], path + "." + k)
            if diff:
                return diff
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return path + ".length"
        for i, (x, y) in enumerate(zip(a, b)):
            diff = _walk_diff(x, y, path + "[" + str(i) + "]")
            if diff:
                return diff
        return None
    if isinstance(a, float) and not math.isfinite(a):
        return path + ".nonfinite"
    return path if a != b else None


def validate_sidecar(data, expected_mode, *, max_rows=2800):
    frozen()
    if not isinstance(data, dict):
        raise ValueError("V147_SIDECAR_INVALID")
    s = data.get("summary")
    if not isinstance(s, dict) or s.get("mode") != expected_mode:
        raise ValueError("V147_WRONG_OR_MISSING_OBSERVER_MODE")
    for forbidden in ("left_wrist_hip_native_physx_truth",):
        if s.get(forbidden) != "UNKNOWN":
            raise ValueError("V147_UNCALIBRATED_NATIVE_TRUTH_CLAIM")
    if (s.get("same_native_physics_tick_calibrated") is not False
            or s.get("off_on_physical_equality_certified") is not False
            or s.get("motor_execution_authorized") is not False):
        raise ValueError("V147_FAKE_RUNTIME_CALIBRATION")
    rows, packets = data.get("original29_5ms_state_rows"), data.get("raw_callback_copies")
    if not isinstance(rows, list) or not 1 <= len(rows) <= max_rows:
        raise ValueError("V147_SIDECAR_REAL_ROWS_MISSING")
    if not isinstance(packets, list):
        raise ValueError("V147_RAW_PACKETS_MISSING")
    if (s.get("original_29dof_completed_5ms_rows") != len(rows)
            or s.get("copied_native_callback_packets") != len(packets)):
        raise ValueError("V147_SIDECAR_COUNTER_DRIFT")
    if expected_mode == "off" and packets:
        raise ValueError("V147_OFF_HAS_NATIVE_CALLBACKS")
    if s.get("original_full_state_checksum") != json_sha(rows):
        raise ValueError("V147_STATE_HASH_MISMATCH")
    if set(PHYSICS_FIELDS) != set(rows[0]):
        raise ValueError("V147_MISSING_PHYSICAL_FIELDS")
    for i, row in enumerate(rows, 1):
        verify_original_state(row)
        if isinstance(row["step"], bool) or row["step"] != i:
            raise ValueError("V147_ORPHAN_OR_DUPLICATE_200HZ_STATE")
    for packet in packets:
        if not isinstance(packet, dict) or not isinstance(packet.get("step_hint"), int):
            raise ValueError("V147_INVALID_CALLBACK_STEP_HINT")
        if (isinstance(packet["step_hint"], bool)
                or not 1 <= packet["step_hint"] <= len(rows)):
            raise ValueError("V147_CALLBACK_STEP_OUT_OF_RANGE")
    return rows


def audit(off, on, *, expected_steps=None):
    cfg = frozen()
    if expected_steps is not None and expected_steps not in (1028, 2800):
        raise ValueError("V147_EXPECTED_STEPS_NOT_FROZEN")
    a = validate_sidecar(off, "off")
    b = validate_sidecar(on, "on")
    if expected_steps is not None and (len(a) != expected_steps or len(b) != expected_steps):
        return {
            "off_steps": len(a), "on_steps": len(b),
            "first_difference": "episode_length_or_early_termination",
            "physics_core_equal": False, "native_pair_truth": "UNKNOWN",
            "runtime_source_attested": False, "live_noninterference_certified": False,
            "strict_physical_success": None,
        }
    first = None
    if len(a) != len(b):
        first = "episode_length_or_early_termination"
    else:
        for i, (left, right) in enumerate(zip(a, b), 1):
            for field in ("step", *cfg["required_parity_fields"]):
                first = _walk_diff(left[field], right[field], "step_" + str(i) + "." + field)
                if first:
                    break
            if first:
                break
    return {
        "off_steps": len(a), "on_steps": len(b),
        "first_difference": first,
        "physics_core_equal": first is None,
        "off_readout_complete": not off["summary"].get("readout_incomplete", True),
        "on_readout_complete": not on["summary"].get("readout_incomplete", True),
        "native_pair_truth": "UNKNOWN",
        "runtime_source_attested": False,
        "live_noninterference_certified": False,
        "strict_physical_success": None,
        "source_level_only_not_physx_attestation": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--off", required=True, type=Path)
    p.add_argument("--on", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expected-steps", type=int)
    args = p.parse_args()
    result = audit(json.loads(args.off.read_text()), json.loads(args.on.read_text()),
                   expected_steps=args.expected_steps)
    result["frozen_protocol_sha"] = FROZEN_SHA
    result["input_off_path"] = str(args.off)
    result["input_on_path"] = str(args.on)
    checksum = safe_write_once(args.output, result)
    print(json.dumps({"output_sha256": checksum, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
