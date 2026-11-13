#!/usr/bin/env python3
"""v134: NVIDIA-ABI-compatible, pure-copy native PhysX contact data gate.

No stage mutation, dynamics execution, contact/success attestation or actuator API.
This module fixes v133's scalar-impulse CPU mock: live PhysX impulse is vec3.
CPU fixtures NEVER prove that a contact happened on Tang.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import numbers
from pathlib import Path
import subprocess
from typing import Callable

from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/v134_frozen_native_vector_pair_and_provenance.json"
FROZEN_SHA = "dbe76c45527dc4595dd90ebaa3754f81f051ffba"
PHYSICAL_DT = 0.005
ACTOR_KEYS = ("actor0", "actor1", "collider0", "collider1")
REQUIRED_COMMON_PHYSICS_FIELDS = (
    "step", "q", "dq", "base_pose", "base_twist", "com_world",
    "left_foot_net_force", "right_foot_net_force", "nonfoot_net_force",
    "ball_pose", "ball_twist", "motor_torque",
)


def frozen_protocol() -> dict:
    original = subprocess.check_output([
        "git", "-C", str(ROOT), "show",
        FROZEN_SHA + ":configs/v134_frozen_native_vector_pair_and_provenance.json",
    ])
    if original != PROTOCOL.read_bytes():
        raise ValueError("FROZEN_V134_PROTOCOL_DRIFT")
    p = json.loads(original)
    if (p["parent_sha"] != "521861d6edb807c71968fc640f975614ea5a10e3"
            or p["physical_dt_s"] != PHYSICAL_DT or p["original_dof"] != 29
            or p["historical_original_g1_steps"] != {
                "frozen_v32": 2800, "cmu_10_02": 2800, "cmu_10_05": 1028
            } or p["native_test_rollouts_while_foreign_gpu_busy"] != 0
            or p["strict_kick_gates"]["instep_impulse_fraction_min"] != 0.95
            or p["strict_kick_gates"]["ball_speed_mps_min"] != 3.5
            or not p["never_certify_native_pair_from_fake_callback"]):
        raise ValueError("FROZEN_V134_ASSUMPTIONS_DRIFT")
    return p


def member(obj, *names):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise ValueError("MISSING_NATIVE_FIELD:" + "|".join(names))


def optional_member(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def finite_number(v):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ValueError("NONNUMERIC_NATIVE_NUMBER")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError("NONFINITE_NATIVE_NUMBER")
    return x


def vec3(v, name="VECTOR"):
    if isinstance(v, (str, bytes)):
        raise ValueError("INVALID_" + name + "_VEC3")
    try:
        s = list(v)
    except (TypeError, ValueError) as exc:
        raise ValueError("INVALID_" + name + "_VEC3") from exc
    if len(s) != 3:
        raise ValueError("INVALID_" + name + "_VEC3")
    return [finite_number(x) for x in s]


def integer(v, name, *, allow_zero=True):
    if (isinstance(v, (list, tuple)) and len(v) == 1):
        v = v[0]
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise ValueError("INVALID_" + name)
    i = int(v)
    if i < (0 if allow_zero else 1):
        raise ValueError("INVALID_" + name)
    return i


def canonical_usd_path(path):
    if not isinstance(path, str) or not path.startswith("/") or path == "/":
        raise ValueError("BAD_NATIVE_USD_PATH")
    if (path.endswith("/") or "//" in path
            or any(segment in ("", ".", "..") for segment in path.split("/")[1:])
            or any(ch.isspace() for ch in path)):
        raise ValueError("BAD_NATIVE_USD_PATH")
    return path


def native_path_decoder(encoded):
    """Only use inside initialized Isaac Sim, with its actual path token ABI."""
    from pxr import PhysicsSchemaTools
    return str(PhysicsSchemaTools.intToSdfPath(encoded))


def event_type_name(event):
    candidate = getattr(event, "name", None)
    text = str(candidate if candidate is not None else event).upper()
    for val in ("CONTACT_FOUND", "CONTACT_PERSISTS", "CONTACT_LOST"):
        if val in text:
            return val
    return "UNKNOWN_EVENT_TYPE"


def _optional_index(cp, a, b):
    value = optional_member(cp, a, b)
    return None if value is None else integer(value, "FACE_INDEX")


def copy_contact_vector_frame(
    headers, contact_data, *, decode_path: Callable,
    step_hint: int, expected_stage_id: int, dt_s: float = PHYSICAL_DT,
):
    """Copy borrowed arrays BEFORE callback returns. No source authentication.

    For a lost event a zero-point header is valid, but is NOT evidence that
    the target pair never made contact. Other zero-point reports are retained
    and flagged for review. No scalar fallback for impulse is permitted.
    """
    step_hint = integer(step_hint, "STEP_HINT", allow_zero=False)
    stage = integer(expected_stage_id, "EXPECTED_STAGE_ID")
    if finite_number(dt_s) != PHYSICAL_DT:
        raise ValueError("NOT_200HZ")
    if not callable(decode_path):
        raise ValueError("NATIVE_PATH_DECODER_REQUIRED")
    source_headers = tuple(headers)
    points = tuple(contact_data)
    rows = []
    used = set()
    for h in source_headers:
        actual_stage = integer(member(h, "stage_id", "stageId"), "STAGE_ID")
        if actual_stage != stage:
            raise ValueError("NATIVE_CONTACT_STAGE_MISMATCH")
        offset = integer(member(h, "contact_data_offset", "contactDataOffset"),
                         "CONTACT_OFFSET")
        count = integer(member(h, "num_contact_data", "contactDataCount",
                               "numContactData"), "CONTACT_COUNT")
        if offset + count > len(points):
            raise ValueError("NATIVE_CONTACT_SLICE_OUT_OF_BOUNDS")
        indices = set(range(offset, offset + count))
        if used.intersection(indices):
            raise ValueError("NATIVE_CONTACT_SLICES_OVERLAP")
        used.update(indices)
        paths = []
        for k in ACTOR_KEYS:
            path = canonical_usd_path(str(decode_path(member(h, k))))
            paths.append(path)
        if paths[0] == paths[1] or paths[2] == paths[3]:
            raise ValueError("NATIVE_PAIR_HAS_DUPLICATE_PATH")
        event = event_type_name(member(h, "type"))
        if event == "CONTACT_LOST" and count:
            raise ValueError("LOST_EVENT_HAS_UNEXPECTED_MANIFOLD")
        copied_points = []
        for i in range(offset, offset + count):
            cp = points[i]
            normal = vec3(member(cp, "normal"), "NORMAL")
            length = math.sqrt(sum(x * x for x in normal))
            if length <= 1e-12:
                raise ValueError("ZERO_CONTACT_NORMAL")
            impulse = vec3(member(cp, "impulse"), "IMPULSE")
            # NVIDIA callback contains 3D total vector impulse; do not
            # rename its magnitude as measured *normal* impulse.
            magnitude = math.sqrt(sum(x * x for x in impulse))
            projection = sum(a * b for a, b in zip(impulse, normal)) / length
            copied_points.append({
                "position_world": vec3(member(cp, "position"), "POSITION"),
                "normal_reported": normal,
                "impulse_world_vector_Ns": impulse,
                "impulse_vector_magnitude_Ns": magnitude,
                "signed_impulse_projection_along_normal_Ns": projection,
                "separation_m": finite_number(member(cp, "separation")),
                "face_index0": _optional_index(cp, "face_index0", "faceIndex0"),
                "face_index1": _optional_index(cp, "face_index1", "faceIndex1"),
            })
        rows.append({
            "stage_id": actual_stage,
            "event_type": event,
            "raw_event_type": str(member(h, "type")),
            "actor_paths": paths[:2],
            "collider_paths": paths[2:],
            "contact_offset": offset,
            "contact_count": count,
            "manifold_points": copied_points,
            "zero_point_is_no_collision_proof": False,
        })
    if used != set(range(len(points))):
        raise ValueError("UNREFERENCED_NATIVE_CONTACT_BUFFER_DATA")
    return {
        "source_claim": "CPU_OR_LIVE_CALLBACK_UNATTESTED",
        "step_hint": step_hint,
        "time_hint_s": round(step_hint * dt_s, 9),
        "stage_id": stage,
        "headers": rows,
        "total_points": len(points),
        "total_headers": len(rows),
        "all_contact_data_consumed": True,
        "unknown_event_types": sum(x["event_type"] == "UNKNOWN_EVENT_TYPE" for x in rows),
        "original_29_state_same_step_certified": False,
        "runtime_positive_negative_controls_certified": False,
        "read_only_off_on_full_physics_invariance_certified": False,
        "native_pair_certified": False,
        "motor_execution_authorized": False,
        "strict_kick_pass": False,
    }


def review_colliders(packet, *, read_only_usd_owner_map):
    """Exact path lookups, never suffix/name matching or fabricated contact proof."""
    if not isinstance(read_only_usd_owner_map, dict):
        raise ValueError("NO_READONLY_USD_OWNER_MAP")
    result = []
    for h in packet["headers"]:
        pairs = h["collider_paths"]
        owners = [read_only_usd_owner_map.get(path) for path in pairs]
        valid = all(isinstance(x, str) and x.startswith("/")
                    for x in owners)
        if valid:
            owners = [canonical_usd_path(x) for x in owners]
        if not valid or owners[0] == owners[1]:
            result.append({
                "colliders": list(pairs),
                "review_status": "REJECT_UNKNOWN_OR_SAME_OWNER",
                "owner_paths": owners,
                "native_pair_certified": False,
            })
            continue
        actors_match_owner = sorted(h["actor_paths"]) == sorted(owners)
        result.append({
            "colliders": list(pairs),
            "owner_paths": owners,
            "actor_paths_match_owner_map": actors_match_owner,
            "review_status": ("UNATTESTED_MATCH" if actors_match_owner
                              else "REJECT_ACTOR_OWNER_MISMATCH"),
            "has_contact_points": bool(h["manifold_points"]),
            "lost_event": h["event_type"] == "CONTACT_LOST",
            "native_pair_certified": False,
        })
    return result


def time_align_packet(packet, *, step, q, dq, joint_names, source_step_authoritative):
    """Only a deterministic review; even all-true inputs cannot attest a GPU run."""
    from numpy import asarray, isfinite
    s = integer(step, "POST_STEP", allow_zero=False)
    qv = asarray(q, dtype=float)
    dv = asarray(dq, dtype=float)
    if qv.shape != (29,) or dv.shape != (29,) or not (
            isfinite(qv).all() and isfinite(dv).all()):
        raise ValueError("INVALID_ORIGINAL29_POST_STEP_STATE")
    reasons = []
    if tuple(joint_names) != tuple(G1_JOINT_ORDER):
        reasons.append("ORIGINAL_G1_JOINT_ORDER_MISMATCH")
    if s != packet["step_hint"]:
        reasons.append("CONTACT_AND_STATE_DIFFERENT_PHYSICS_STEP")
    if not source_step_authoritative:
        reasons.append("BACKEND_STEP_CLOCK_NOT_CALIBRATED")
    if packet["unknown_event_types"]:
        reasons.append("UNKNOWN_PHYSX_EVENT_TYPE")
    if not packet["total_headers"]:
        reasons.append("EMPTY_CALLBACK_CANNOT_ATTEST_ABSENCE_OF_CONTACT")
    return {
        "step": s,
        "source_claim": packet["source_claim"],
        "review_ready": not reasons,
        "reasons": reasons,
        "29dof_position_checksum": hashlib.sha256(qv.tobytes()).hexdigest(),
        "29dof_velocity_checksum": hashlib.sha256(dv.tobytes()).hexdigest(),
        "native_pair_certified": False,
        "motor_execution_authorized": False,
    }


@dataclass
class CopyOnlyContactRecorder:
    """An unprivileged listener, install into an existing PhysX world only.

    The caller supplies the true backend step counter; never increment on
    callback count, which need not equal physics steps. Ownership remains with
    the caller and must remain alive for the duration of the physical run.
    """
    callback_step: Callable
    stage_id: int
    decode_path: Callable = native_path_decoder
    frames: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    subscription: object = None

    def on_contact(self, headers, data):
        try:
            p = copy_contact_vector_frame(
                headers, data, decode_path=self.decode_path,
                step_hint=self.callback_step(), expected_stage_id=self.stage_id)
            self.frames.append(p)
        except Exception as e:
            self.errors.append(type(e).__name__ + ":" + str(e))

    def attach_read_only(self):
        if self.subscription is not None:
            raise ValueError("DUPLICATE_NATIVE_SUBSCRIPTION")
        from omni.physx import get_physx_simulation_interface
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self.on_contact)
        return self.subscription

    def review(self):
        seen = [x["step_hint"] for x in self.frames]
        if any(b < a for a, b in zip(seen, seen[1:])):
            raise ValueError("NONMONOTONIC_NATIVE_CALLBACK_STEP_HINT")
        return {
            "total_copied_callback_frames": len(self.frames),
            "total_copied_headers": sum(f["total_headers"] for f in self.frames),
            "total_copied_manifold_points": sum(f["total_points"] for f in self.frames),
            "errors": list(self.errors),
            "callback_step_clock_source_unattested": True,
            "native_pair_certified": False,
            "motor_execution_authorized": False,
        }


def validate_original_g1_physics_row(row):
    """Validate minimum 29DoF and body state dimensions BEFORE equality."""
    if integer(row["step"], "PHYSICS_STEP", allow_zero=False) < 1:
        raise ValueError("INVALID_PHYSICS_STEP")
    for key in ("q", "dq", "motor_torque"):
        try:
            seq = list(row[key])
        except (TypeError, ValueError) as exc:
            raise ValueError("INVALID_ORIGINAL29_PHYSICAL_FIELD:" + key) from exc
        if len(seq) != 29:
            raise ValueError("INVALID_ORIGINAL29_PHYSICAL_FIELD:" + key)
        for value in seq:
            finite_number(value)
    for key in ("com_world", "left_foot_net_force",
                "right_foot_net_force", "nonfoot_net_force",
                "ball_pose", "ball_twist"):
        vec3(row[key], key.upper())
    for key,subfields in (("base_pose", ("position", "quaternion")),
                          ("base_twist", ("linear", "angular"))):
        if not isinstance(row[key], dict) or set(row[key]) != set(subfields):
            raise ValueError("INCOMPLETE_ROOT_STATE:" + key)
        for sub in subfields:
            seq = list(row[key][sub])
            if len(seq) != (4 if sub == "quaternion" else 3):
                raise ValueError("INVALID_ROOT_STATE_VECTOR:" + key)
            for value in seq:
                finite_number(value)


def exact_off_on_physics_equality(off_rows, on_rows, *, required_fields):
    """Compare every explicitly supplied physical state field, not event logs.

    This is a CPU comparator; identical synthetic records do NOT attest
    observer noninterference. Records must come from separate real rollouts.
    """
    if not required_fields or not isinstance(required_fields, (tuple, list)):
        raise ValueError("MISSING_PHYSICS_STATE_FIELD_SET")
    if len(set(required_fields)) != len(required_fields):
        raise ValueError("DUPLICATED_REQUIRED_FIELDS")
    if set(required_fields) != set(REQUIRED_COMMON_PHYSICS_FIELDS):
        raise ValueError("INCOMPLETE_FULL_COMMON_PHYSICS_FIELDS")
    off_rows, on_rows = list(off_rows), list(on_rows)
    if not off_rows or len(off_rows) != len(on_rows):
        return {
            "exact_common_physics_state_equality": False,
            "reason": "EPISODE_LENGTH_OR_EMPTY_MISMATCH",
            "real_source_provenance_certified": False,
        }
    mismatches = []
    for i, (off, on) in enumerate(zip(off_rows, on_rows)):
        if not isinstance(off, dict) or not isinstance(on, dict):
            raise ValueError("INVALID_PHYSICS_FRAME")
        if set(off) != set(on) or set(off) != set(required_fields):
            raise ValueError("PHYSICAL_FIELD_SET_INCOMPLETE_OR_CHANGED")
        validate_original_g1_physics_row(off)
        validate_original_g1_physics_row(on)
        # JSON floats deliberately preserve signed zeros as explicit values,
        # refuse nonfinite input and compare actual nested scalar/array values.
        def normalize(x):
            if isinstance(x, (list, tuple)):
                return [normalize(v) for v in x]
            if isinstance(x, dict):
                return {k: normalize(v) for k, v in sorted(x.items())}
            if isinstance(x, bool) or x is None or isinstance(x, str):
                return x
            return finite_number(x)
        a = normalize(off)
        b = normalize(on)
        if a != b:
            mismatches.append(i)
    return {
        "checked_frames": len(off_rows),
        "checked_physical_fields_each_frame": len(required_fields),
        "exact_common_physics_state_equality": not mismatches,
        "mismatch_frame_indices": mismatches[:20],
        "mismatch_frame_count": len(mismatches),
        "real_source_provenance_certified": False,
        "native_pair_certified": False,
        "motor_execution_authorized": False,
    }


def review_historical_original_v54():
    """Runs existing SHA-locked 3-episode true-physics provenance pipeline."""
    import v133_physical_event_ledger as old
    source = old.real_event_ledger()
    cfg = frozen_protocol()
    counts = {k: x["real_200hz_steps"] for k, x in source["cases"].items()}
    if counts != cfg["historical_original_g1_steps"]:
        raise ValueError("V54_ORIGINAL_200HZ_STEP_COUNT_DRIFT")
    bad = source["cases"]["cmu_10_05"]
    if (bad["pre_event_nonfoot_peak_t_s"] != 5.135
            or bad["observed_last_time_s"] != 5.14
            or bad["event_5p145_has_real_29dof_step"]):
        raise ValueError("V54_FAILURE_TIME_EVIDENCE_DRIFT")
    return {
        "version": "v0.1.134",
        "historical_physx_episodes_sha_verified": 3,
        "historical_real_200hz_frames_sha_verified": sum(counts.values()),
        "cases": counts,
        "failed_real_nonfoot_peak_time_s": 5.135,
        "failed_last_real_q_time_s": 5.140,
        "failed_named_contact_sensor_time_s": 5.145,
        "failed_named_event_same_step_q_state_exists": False,
        "new_physx_episodes": 0,
        "native_vector_manifold_truth_captured": False,
        "this_is_not_new_policy_success_rate": True,
        "motor_execution_authorized": False,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    r = review_historical_original_v54()
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
