#!/usr/bin/env python3
"""v149 pre-registered 48-case synthetic callback-timing falsification matrix.

No physical control, no IsaacLab import and no new PhysX episode. Runtime
source qualification is explicitly false, including every positive fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

from v149_native_callback_capture import (
    V149CopyOnlyNativeCapture, existing_schema_census, frozen,
)
from v142_native_pair_ledger import historical_recheck, safe_write_once
from v146_native_original_bridge import PHYSICS_FIELDS

ROOT = Path(__file__).resolve().parents[1]
ACTORS = {
    1: "/World/Robot/left_wrist_yaw_link",
    2: "/World/Robot/left_hip_roll_link",
    3: "/World/Robot/left_wrist_yaw_link/collision",
    4: "/World/Robot/left_hip_roll_link/collision",
}
OWNERS = {ACTORS[3]: ACTORS[1], ACTORS[4]: ACTORS[2]}
POLICIES = ("frozen_v32", "cmu_10_02", "cmu_10_05")
HEADER_KINDS = ("VECTOR_CONTACT", "INVALID_SCALAR_IMPULSE")
PHASES = (
    "BEFORE_FIRST_SCENE_UPDATE", "INSIDE_SCENE_UPDATE",
    "BETWEEN_SCENE_UPDATES", "AFTER_TERMINAL_SCENE_UPDATE",
)


def fake_state(step):
    """One complete artificial 29DoF row; never imported as a PhysX result."""
    return {
        "step": step, "q": [0.]*29, "dq": [0.]*29,
        "motor_torque": [0.]*29,
        "base_pose": {"position":[0.,0.,1.],"quaternion":[1.,0.,0.,0.]},
        "base_twist": {"linear":[0.,0.,0.],"angular":[0.,0.,0.]},
        "com_world": [0.,0.,.5],
        "left_foot_net_force":[0.,0.,100.],
        "right_foot_net_force":[0.,0.,100.],
        "nonfoot_net_force":[0.,0.,0.],
        "ball_pose":[0.,0.,0.], "ball_twist":[0.,0.,0.],
    }


class FakeSampler:
    def __init__(self):
        self.samples = []
    def sample(self, step):
        self.samples.append(step)
        return fake_state(step)


def native_borrowed_fixture(kind):
    header = [{
        "stage_id": 42, "actor0": 1, "actor1": 2,
        "collider0": 3, "collider1": 4,
        "type": "CONTACT_FOUND",
        "contact_data_offset": 0, "num_contact_data": 1,
    }]
    point = [{
        "position": [0.,0.,0.], "normal": [1.,0.,0.],
        "impulse": ([1.25,0.,0.] if kind == "VECTOR_CONTACT" else 1.25),
        "separation": -.001,
    }]
    return header, point


def fake_stage_report(schema_present):
    census = existing_schema_census(
        OWNERS,
        has_owner_report=lambda path: bool(schema_present),
        has_reporting_articulation_ancestor=lambda path: False,
    )
    return (42, OWNERS, lambda x: ACTORS[x], {
        "source": "SYNTHETIC_USD_STAGE_FIXTURE_NO_RUNTIME_ATTESTATION",
        "v149_existing_report_schema_census": census,
    })


def run_case(policy, phase, header_kind, schema_present):
    cfg = frozen()
    if (policy not in POLICIES or phase not in PHASES or
            header_kind not in HEADER_KINDS or type(schema_present) is not bool):
        raise ValueError("V149_MATRIX_CELL_NOT_PREREGISTERED")
    callback = {}
    sampler = FakeSampler()
    def subscribe(fn):
        callback["fn"] = fn
        return object()
    s = V149CopyOnlyNativeCapture(
        mode="on", sampler=sampler, native_subscribe=subscribe,
        stage_reader=lambda: fake_stage_report(schema_present),
    )
    s.attach()
    assert not s.failed
    assert callback
    header, point = native_borrowed_fixture(header_kind)

    def inject():
        callback["fn"](header, point)
        if header_kind == "VECTOR_CONTACT":
            # A real callback borrows data, so mutate it after returning.
            point[0]["impulse"][0] = 99.

    if phase == "BEFORE_FIRST_SCENE_UPDATE":
        inject()
        s.before_native_update(.005)
        s.after_native_update(.005)
    elif phase == "INSIDE_SCENE_UPDATE":
        s.before_native_update(.005)
        inject()
        s.after_native_update(.005)
    elif phase == "BETWEEN_SCENE_UPDATES":
        s.before_native_update(.005)
        s.after_native_update(.005)
        inject()
        s.before_native_update(.005)
        s.after_native_update(.005)
    else:
        s.before_native_update(.005)
        s.after_native_update(.005)
        s.mark_terminal()
        inject()
    summary = s.review()
    expected_packet = int(header_kind == "VECTOR_CONTACT")
    if summary["copied_native_callback_packets"] != expected_packet:
        raise AssertionError("V149_MATRIX_EVENT_LOSS_OR_FALSE_CAPTURE")
    if summary["left_wrist_hip_native_physx_truth"] != "UNKNOWN":
        raise AssertionError("V149_SYNTHETIC_PAIR_PROMOTED_TO_PHYSX")
    if expected_packet:
        pkt = s.packets[0]
        if pkt["v149_arrival_phase"] != phase:
            raise AssertionError("V149_CALLBACK_PHASE_MISCLASSIFIED")
        if pkt["headers"][0]["manifold_points"][0]["impulse_world_vector_Ns"][0] != 1.25:
            raise AssertionError("V149_BORROWED_BUFFER_NOT_DEEPCOPIED")
    return {
        "policy_name_is_only_a_fixture_label": policy,
        "phase": phase,
        "header_kind": header_kind,
        "existing_report_api_fixture": schema_present,
        "simulated_rows_count": len(s.rows),
        "valid_packets_preserved": len(s.packets),
        "callback_invocations": summary["callback_invocations_not_step_count"],
        "arrival_phase_counts": summary["arrival_phase_counts"],
        "step_candidates_unattested": (
            s.packets[0]["v149_adjacent_step_candidates_unattested"]
            if s.packets else []
        ),
        "valid_event_path_pair_candidates_unattested": summary["wrist_hip_candidate_total"],
        "schema_eligible_in_fixture": summary["existing_report_api_census"]["pair_report_schema_eligible"],
        "invalid_callback_flagged": any("V149_CALLBACK_COPY_INVALID" in e for e in s.errors),
        "source_is_runtime": False,
        "same_native_physics_tick_calibrated": False,
        "native_wrist_hip_contact_certified": False,
        "strict_kick_success": None,
    }


def frozen_historical():
    real = historical_recheck()
    if real["historical_physics_5ms_frames"] != 6628:
        raise ValueError("V149_HISTORICAL_REAL_STATE_COUNT_DRIFT")
    return real


def run_all():
    cfg = frozen()
    grid = [
        run_case(p, phase, h, present)
        for p, phase, h, present in product(POLICIES, PHASES, HEADER_KINDS, (False, True))
    ]
    if len(grid) != cfg["synthetic_event_matrix"]["total"]:
        raise ValueError("V149_PREREGISTERED_DENOMINATOR_DRIFT")
    origin = ROOT / "tools/v146_run_original_with_readonly.py"
    runner = ROOT / "tools/v149_run_original_anyphase_readonly.py"
    originals = ROOT / "tools/v080_run_poll_strike.py"
    base = {
        "v146_runner_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        "v149_runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "original_v080_runner_sha256": hashlib.sha256(originals.read_bytes()).hexdigest(),
        "legacy_only_scene_update_callback_bracket":
            ("before_native_update(dt)" in origin.read_text()
             and "native_update(dt)" in origin.read_text()
             and "after_native_update(dt)" in origin.read_text()),
        "runtime_src_and_positive_negative_calibration": False,
    }
    return {
        "version": "v0.1.149",
        "parent_sha": cfg["parent_sha"],
        "frozen_protocol_sha": "8f6670c5942c330cc91679e798e1911e41e0606e",
        "historical_real_physx_not_rerun": frozen_historical(),
        "new_real_physx_episodes": 0,
        "synthetic_case_count": len(grid),
        "synthetic_expected_valid_packets": 24,
        "synthetic_observed_valid_packets": sum(x["valid_packets_preserved"] for x in grid),
        "invalid_vector_payloads_rejected": sum(x["invalid_callback_flagged"] for x in grid),
        "runtime_physx_wrist_hip_pair_truth": "UNKNOWN",
        "all_provenance_stays_unattested": True,
        "runtime_off_on_parity": "NOT_RUN",
        "human_professional_instep_contact_and_grf": "NOT_AVAILABLE",
        "runner_source_checks": base,
        "synthetic_cells": grid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = run_all()
    sha = safe_write_once(args.output, output)
    print(json.dumps({
        "file": str(args.output), "sha256": sha,
        "synthetic_cases": output["synthetic_case_count"],
        "valid_vector_callbacks_preserved": output["synthetic_observed_valid_packets"],
        "invalid_callbacks_detected": output["invalid_vector_payloads_rejected"],
        "historical_real_200hz_states_checked":
            output["historical_real_physx_not_rerun"]["historical_physics_5ms_frames"],
        "new_real_physx_episodes": 0,
        "native_contact_truth": "UNKNOWN",
    }))


if __name__ == "__main__":
    main()
