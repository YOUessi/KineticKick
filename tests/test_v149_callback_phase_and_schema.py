"""Falsification of v146 bracket-only callback loss; CPU fixtures cannot certify PhysX."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import v149_native_callback_capture as cap
from v149_phase_matrix import (
    ACTORS, OWNERS, FakeSampler, fake_stage_report, native_borrowed_fixture,
    run_case, PHASES, POLICIES, HEADER_KINDS,
)


def make(schema=True, mode="on", max_packets=None):
    callbacks = {}
    def subscribe(fn):
        callbacks["cb"] = fn
        return object()
    s = cap.V149CopyOnlyNativeCapture(
        mode=mode, sampler=FakeSampler(), native_subscribe=subscribe,
        stage_reader=lambda: fake_stage_report(schema),
        max_packets=max_packets,
    )
    s.attach()
    return s, callbacks


def step(s):
    s.before_native_update(.005)
    s.after_native_update(.005)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("kind", HEADER_KINDS)
@pytest.mark.parametrize("schema_present", (False, True))
def test_48_registered_synthetic_source_cases(policy, phase, kind, schema_present):
    r = run_case(policy, phase, kind, schema_present)
    assert r["source_is_runtime"] is False
    assert r["same_native_physics_tick_calibrated"] is False
    assert r["native_wrist_hip_contact_certified"] is False
    assert r["strict_kick_success"] is None
    assert r["valid_packets_preserved"] == (kind == "VECTOR_CONTACT")


def test_native_reporting_schema_readonly_presence_and_missing():
    yes = cap.existing_schema_census(
        OWNERS, has_owner_report=lambda p:p == ACTORS[1],
        has_reporting_articulation_ancestor=lambda p:False,
    )
    no = cap.existing_schema_census(
        OWNERS, has_owner_report=lambda p:False,
        has_reporting_articulation_ancestor=lambda p:False,
    )
    assert yes["pair_report_schema_eligible"]
    assert yes["status"] == "EXISTING_REPORTING_ACTOR_FOUND"
    assert not no["pair_report_schema_eligible"]
    assert no["status"] == "NO_EXISTING_REPORTING_ACTOR_FOUND"
    assert not no["no_events_means_no_contact"]
    assert not yes["runtime_positive_negative_controls_passed"]


def test_articulation_root_existing_schema_is_relevant_but_not_certification():
    r = cap.existing_schema_census(
        OWNERS, has_owner_report=lambda p: False,
        has_reporting_articulation_ancestor=lambda p: p == ACTORS[2],
    )
    assert r["pair_report_schema_eligible"]
    assert not r["runtime_positive_negative_controls_passed"]


def test_child_collision_prim_schema_only_does_not_authorize_owner():
    checked = []
    def checker(path):
        checked.append(path)
        return path.endswith("/collision")
    r = cap.existing_schema_census(
        OWNERS, has_owner_report=checker,
        has_reporting_articulation_ancestor=lambda p:False,
    )
    assert not r["pair_report_schema_eligible"]
    assert all("/collision" not in p for p in checked)


def test_duplicate_target_owner_paths_not_selectively_resolved():
    owners = dict(OWNERS)
    owners["/Other/left_hip_roll_link/collision"] = "/Other/left_hip_roll_link"
    r = cap.existing_schema_census(
        owners, has_owner_report=lambda p:True,
        has_reporting_articulation_ancestor=lambda p:True,
    )
    assert r["status"] == "UNKNOWN_TARGET_ACTOR_OWNERS"
    assert not r["pair_report_schema_eligible"]


@pytest.mark.parametrize("answer", (1, "yes", None, 0.0))
def test_fake_schema_flags_refused(answer):
    with pytest.raises(ValueError, match="UNTRUSTED_SCHEMA_BOOL"):
        cap.existing_schema_census(
            OWNERS, has_owner_report=lambda p:answer,
            has_reporting_articulation_ancestor=lambda p:False,
        )


def test_absent_observable_schema_always_blocks_inference_not_state_sampling():
    s,callbacks=make(schema=False)
    step(s)
    assert len(s.rows)==1
    assert s.review()["readout_incomplete"]
    assert not s.review()["negative_event_absence_certified"]
    assert s.review()["left_wrist_hip_native_physx_truth"]=="UNKNOWN"


def test_off_only_copies_state_and_does_not_subscribe():
    s,callbacks = make(schema=False,mode="off")
    for _ in range(3):step(s)
    assert callbacks=={}
    assert len(s.rows)==3
    assert not s.review()["readout_incomplete"]
    assert s.review()["copied_native_callback_packets"]==0


def test_invalid_vec3_is_contained_without_suppressing_actual_states():
    s,callbacks=make()
    h,p=native_borrowed_fixture("INVALID_SCALAR_IMPULSE")
    callbacks["cb"](h,p)
    for _ in range(3):step(s)
    assert len(s.rows)==3
    assert not s.packets
    assert any("V149_CALLBACK_COPY_INVALID" in e for e in s.errors)


def test_callback_before_update_now_preserved_unlike_v146():
    s,callbacks=make()
    h,p=native_borrowed_fixture("VECTOR_CONTACT")
    callbacks["cb"](h,p)
    step(s)
    assert len(s.packets)==1
    assert s.packets[0]["v149_arrival_phase"]=="BEFORE_FIRST_SCENE_UPDATE"
    assert s.packets[0]["v149_adjacent_step_candidates_unattested"]==[1]
    assert s.review()["wrist_hip_candidate_total"]==1
    assert s.review()["same_native_physics_tick_calibrated"] is False


def test_two_callbacks_one_state_not_two_steps():
    s,callbacks=make()
    s.before_native_update(.005)
    for _ in range(2):
        callbacks["cb"](*native_borrowed_fixture("VECTOR_CONTACT"))
    s.after_native_update(.005)
    assert s.review()["copied_native_callback_packets"]==2
    assert s.review()["callback_invocations_not_step_count"]==2
    assert s.review()["original_29dof_completed_5ms_rows"]==1


def test_source_arrays_are_not_retained_after_callback():
    s,callbacks=make()
    h,p=native_borrowed_fixture("VECTOR_CONTACT")
    s.before_native_update(.005)
    callbacks["cb"](h,p)
    p[0]["impulse"][0]=77.
    s.after_native_update(.005)
    assert s.packets[0]["headers"][0]["manifold_points"][0]["impulse_world_vector_Ns"][0]==1.25


def test_memory_cap_explicitly_reports_truncation_and_keeps_state():
    s,callbacks=make(max_packets=1)
    for _ in range(3):callbacks["cb"](*native_borrowed_fixture("VECTOR_CONTACT"))
    step(s)
    assert len(s.packets)==1
    assert s.review()["callback_invocations_not_step_count"]==3
    assert "V149_NATIVE_EVENT_MEMORY_CAP_EXHAUSTED" in s.errors
    assert len(s.rows)==1


def test_no_physics_state_is_orphan_not_interpolated():
    s,callbacks=make()
    callbacks["cb"](*native_borrowed_fixture("VECTOR_CONTACT"))
    r=s.review()
    assert r["orphaned_callback_packets"]==1
    assert r["readout_incomplete"]
    assert r["original_29dof_completed_5ms_rows"]==0


def test_nested_update_invalid_without_physical_sample_fabrication():
    s,callbacks=make()
    s.before_native_update(.005)
    s.before_native_update(.005)
    s.after_native_update(.005)
    assert len(s.rows)==1
    assert "V149_NESTED_SCENE_UPDATE" in s.errors


def test_no_runtime_physx_claim_even_with_fixture_pair_and_schema():
    s,callbacks=make(schema=True)
    s.before_native_update(.005)
    callbacks["cb"](*native_borrowed_fixture("VECTOR_CONTACT"))
    s.after_native_update(.005)
    r=s.review()
    assert r["wrist_hip_candidate_total"]==1
    assert r["left_wrist_hip_native_physx_truth"]=="UNKNOWN"
    assert r["runtime_positive_negative_controls_passed"] is False
    assert r["motor_execution_authorized"] is False


def test_frozen_contract_immutable_sha():
    cfg=cap.frozen()
    assert cfg["parent_sha"]=="4505e095ec5e13e9226d35adcfb9ba368e053156"
    assert cfg["native_callback_may_run_outside_scene_update"]
