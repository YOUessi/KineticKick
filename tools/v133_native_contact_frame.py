#!/usr/bin/env python3
"""v133 copy-only PhysX callback adapter and strict per-step evidence ledger.

CPU unit tests supply fake contact objects; they NEVER attest physical contacts.
The live adapter must be calibrated on a native positive & negative PhysX pair
and OFF/ON physical invariance before any runtime collision truth is asserted.
No USD prim is created or mutated, no motor control occurs in this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA = "6cd4e597cc7ab4683fc097558d0df2fad9cb0d74"
FILE = ROOT / "configs/v133_frozen_native_contact_frame_and_provenance.json"


def frozen():
    expected = subprocess.check_output(
        ["git", "-C", str(ROOT), "show",
         FROZEN_SHA + ":configs/v133_frozen_native_contact_frame_and_provenance.json"])
    if FILE.read_bytes() != expected:
        raise ValueError("FROZEN_PROTOCOL_CHANGED")
    cfg = json.loads(expected)
    if (cfg["version"] != "v0.1.133" or cfg["physical_dt_s"] != .005
            or cfg["original_g1_dof"] != 29
            or cfg["original_v54_cases"] != {
                "frozen_v32": 2800, "cmu_10_02": 2800, "cmu_10_05": 1028}
            or not cfg["unknown_pair_blocks_motor"]
            or not cfg["no_gpu_parallel_with_existing_process"]):
        raise ValueError("FROZEN_PROTOCOL_ASSUMPTIONS_CHANGED")
    return cfg


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite_float(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("NONNUMERIC_OR_BOOL")
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("NONFINITE_NUMBER")
    return v


def xyz(v):
    try:
        v = list(v)
    except TypeError as exc:
        raise ValueError("INVALID_VECTOR") from exc
    if len(v) != 3:
        raise ValueError("INVALID_3_VECTOR")
    return [finite_float(x) for x in v]


def canonical_path(s):
    if (not isinstance(s, str) or not s.startswith("/")
            or s.endswith("/") or "//" in s or ".." in s
            or any(x.isspace() for x in s)):
        raise ValueError("UNQUALIFIED_ABSOLUTE_USD_PATH")
    return s


def attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise ValueError("PHYSX_FIELD_NOT_AVAILABLE:" + "/".join(names))


def positive_int(v):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError("INVALID_PHYSX_ARRAY_INDEX")
    return v


def copy_native_callback(
    headers, contacts, *, decode_path, step, dt_s=.005, stage_id=None,
    backend_clock_calibrated=False,
):
    """Copy borrowed callback buffers immediately; never retain references.

    step is only a *callback clock hint* until backend-specific timing was
    calibrated against an independently captured post-simulation G1 state.
    Native origin can be forged in tests; schema validity is NOT source proof.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("INVALID_CALLBACK_STEP_CLOCK")
    if finite_float(dt_s) != .005:
        raise ValueError("NONFROZEN_200HZ")
    if not callable(decode_path):
        raise ValueError("MISSING_NATIVE_PATH_DECODER")
    headers = tuple(headers)
    contacts = tuple(contacts)
    rows = []
    used = set()
    for header in headers:
        start = positive_int(attr(header, "contact_data_offset", "contactDataOffset"))
        count = positive_int(attr(header, "num_contact_data", "contactDataCount"))
        if start + count > len(contacts):
            raise ValueError("PHYSX_CONTACT_SLICE_OUT_OF_BOUNDS")
        indices = set(range(start, start+count))
        if indices & used:
            raise ValueError("OVERLAPPING_CONTACT_HEADER_SLICES")
        used |= indices
        stage = attr(header, "stage_id", "stageId")
        if stage_id is not None and stage != stage_id:
            raise ValueError("CONTACT_STAGE_ID_MISMATCH")
        paths = [canonical_path(str(decode_path(attr(header, name))))
                 for name in ("actor0", "actor1", "collider0", "collider1")]
        if paths[0] == paths[1] or paths[2] == paths[3]:
            raise ValueError("DUPLICATE_ACTOR_OR_COLLIDER")
        points = []
        for i in range(start, start+count):
            cp = contacts[i]
            pos = xyz(attr(cp, "position"))
            normal = xyz(attr(cp, "normal"))
            if sum(x*x for x in normal) < 1e-12:
                raise ValueError("ZERO_PHYSX_CONTACT_NORMAL")
            impulse = finite_float(attr(cp, "impulse"))
            if impulse < 0:
                raise ValueError("NEGATIVE_PHYSX_NORMAL_IMPULSE")
            separation = finite_float(attr(cp, "separation"))
            points.append(dict(position=pos, normal=normal,
                               impulse=impulse, separation=separation))
        rows.append(dict(stage_id=stage, actors=paths[:2],
                         colliders=paths[2:], contact_data_offset=start,
                         native_contact_count=count, points=points,
                         header_type=str(attr(header, "type"))))
    return dict(
        source_label="COPIED_CALLBACK_UNATTESTED",
        callback_step_hint=step, callback_time_hint_s=round(step*dt_s, 9),
        backend_clock_calibrated=bool(backend_clock_calibrated),
        native_header_count=len(rows), native_contact_total=len(contacts),
        all_native_contacts_covered=(used == set(range(len(contacts)))),
        headers=rows, same_step_original29_state_certified=False,
        independent_runtime_positive_and_negative_controls=False,
        native_actor_collider_pair_certified=False,
        strict_g1_kick_success_certified=False, motor_execution_authorized=False)


def candidate_links(packet, *, allowed_owners):
    """Report possible link pairs only when explicit collider->link owners exist.

    Never use string suffix/substr matches; actual USD stage traversal is needed
    to construct an independently verified mapping on the runtime.
    """
    if not isinstance(allowed_owners, dict):
        raise ValueError("OWNER_MAP_REQUIRED")
    proposals=[]
    for item in packet["headers"]:
        owners=[]
        for collider in item["colliders"]:
            link = allowed_owners.get(collider)
            owners.append(link)
        if all(isinstance(x, str) for x in owners) and owners[0] != owners[1]:
            proposals.append(dict(
                pair=sorted(owners), callback_step_hint=packet["callback_step_hint"],
                points=len(item["points"]), suspected_only=True,
                native_pair_certified=False))
    return proposals


def state_snapshot(step, q, dq, *, dt_s=.005, expected_order=None):
    """Independent post-step state, never automatically fused with callbacks."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("BAD_STATE_STEP")
    q, dq = list(q), list(dq)
    if len(q) != 29 or len(dq) != 29:
        raise ValueError("NOT_ORIGINAL_29DOF")
    if expected_order is not None:
        from wholebody_safety_interface_v055 import G1_JOINT_ORDER
        if tuple(expected_order) != tuple(G1_JOINT_ORDER):
            raise ValueError("ORIGINAL_29_JOINT_ORDER_NOT_VERIFIED")
    return dict(physical_step=step, physical_time_s=round(step*dt_s, 9),
                q=[finite_float(x) for x in q],
                dq=[finite_float(x) for x in dq],
                q_order=list(expected_order) if expected_order else None,
                native_pair_certified=False, motor_execution_authorized=False)


def reconcile_callback_and_state(packet, post_state, *, calibrations):
    """Conservative evidence admission; no authorization even if all keys match.

    Proof of physical events additionally requires trusted source attestation,
    working positive and negative controls, and OFF/ON state equality.
    """
    reasons=[]
    if packet["callback_step_hint"] != post_state["physical_step"]:
        reasons.append("CALLBACK_STATE_STEP_MISMATCH")
    from wholebody_safety_interface_v055 import G1_JOINT_ORDER
    if post_state.get("q_order") != list(G1_JOINT_ORDER):
        reasons.append("ORIGINAL_29_JOINT_ORDER_UNVERIFIED")
    if abs(packet["callback_time_hint_s"]-post_state["physical_time_s"])>1e-9:
        reasons.append("CALLBACK_STATE_TIME_MISMATCH")
    for name in ("native_positive", "native_negative", "callback_timing",
                 "off_on_state_invariance", "owner_map_verified"):
        if calibrations.get(name) is not True:
            reasons.append("UNCALIBRATED_" + name.upper())
    if not packet["all_native_contacts_covered"]:
        reasons.append("CONTACT_HEADER_DATA_UNCOVERED")
    if not packet["backend_clock_calibrated"]:
        reasons.append("BACKEND_CALLBACK_CLOCK_UNCALIBRATED")
    if not packet["headers"]:
        reasons.append("NO_NATIVE_CONTACT_HEADERS")
    return dict(
        checks_passed=not reasons, reasons=sorted(set(reasons)),
        callback_headers=packet["native_header_count"],
        step=post_state["physical_step"],
        original29_state_dimension=29,
        callback_buffers_copied=True,
        source_attestation_required=True,
        native_actor_collider_pair_certified=False,
        motor_execution_authorized=False)


class LiveCallbackCopier:
    """Attach to an ALREADY configured physics world; never changes USD state.

    The caller must control step timing and own the subscription lifetime. A
    callback that fires without an unambiguous backend clock is not aligned.
    """
    def __init__(self, *, clock, stage_id, decode_path):
        self.clock = clock
        self.stage_id = stage_id
        self.decode_path = decode_path
        self.packets = []
        self.errors = []

    def __call__(self, headers, contacts):
        try:
            step = self.clock()
            payload = copy_native_callback(
                headers, contacts, decode_path=self.decode_path,
                step=step, stage_id=self.stage_id)
            self.packets.append(payload)
        except Exception as exc:
            self.errors.append(type(exc).__name__ + ":" + str(exc))

    def subscribe(self):
        """Executed inside Isaac Sim Python runtime, not on CPU-only test host."""
        from omni.physx import get_physx_simulation_interface
        return get_physx_simulation_interface().subscribe_contact_report_events(self)
