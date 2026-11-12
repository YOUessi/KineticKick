"""v124 fail-closed true actor/collider evidence, no fabricated PhysX claims."""
import copy
import sys
from pathlib import Path

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v124_native_contact_truth_gate as m

A="/World/G1/left_wrist_yaw_link"
B="/World/G1/left_hip_roll_link"
C="/World/G1/left_wrist_yaw_link/collision"
D="/World/G1/left_hip_roll_link/collision"
OWN={C:A,D:B}
PAIR=(A,B)
SOURCE="a"*64

def synthetic_packet():
    return dict(packet_origin="native_callback",step=100,dt_s=.005,
                timestamp_s=.5,actors=[A,B],colliders=[C,D],
                native_header_count=1,native_contact_count=1,
                points=[{"position":[0,0,.9],"normal":[1,0,0],
                         "impulse":[.1,0,0]}])

def screen(event,owners=OWN,source=SOURCE):
    return m.qualified_native_packet(event,PAIR,owners,source)

def test_synthetic_valid_schema_is_not_native_physics_proof():
    row=screen(synthetic_packet())
    assert row["packet_schema_valid"] is True
    assert row["native_pair_qualified"] is False
    assert row["requires_independent_physx_runtime_and_source_attestation"] is True
    assert row["conclusion"]=="SCHEMA_VALID_BUT_UNATTESTED"

@pytest.mark.parametrize("change,why",[
    ({"packet_origin":"tensor"}, "TENSOR_OR_SYNTHETIC"),
    ({"step":0},"NO_REAL_STEP"),
    ({"timestamp_s":.2},"STEP_TIME_INCONSISTENT"),
    ({"actors":[A,A]},"DUPLICATE_ACTOR_OR_COLLIDER"),
    ({"actors":[A,"/World/G1/left_knee_link"]},"WRONG_BODY_PAIR"),
    ({"colliders":[C,C]},"DUPLICATE_ACTOR_OR_COLLIDER"),
    ({"native_header_count":0},"NO_NATIVE_HEADERS"),
    ({"native_contact_count":0},"NO_NATIVE_HEADERS"),
    ({"points":[]},"NO_NATIVE_MANIFOLD"),
])
def test_each_missing_condition_fails_schema(change,why):
    event=synthetic_packet()
    event.update(change)
    row=screen(event)
    assert not row["packet_schema_valid"]
    assert not row["native_pair_qualified"]
    assert any(why in s for s in row["reasons"])

def test_actor_collider_ordering_is_invariant_when_pair_swapped():
    event=synthetic_packet()
    event["actors"].reverse();event["colliders"].reverse()
    assert screen(event)["packet_schema_valid"]

def test_actor_collider_ownership_needs_real_stage_map():
    event=synthetic_packet()
    assert "UNPROVEN_COLLIDER_TO_ACTOR_OWNERSHIP" in screen(event,{})["reasons"]
    assert not screen(event,{C:B,D:A})["packet_schema_valid"]

@pytest.mark.parametrize("normal,impulse",[
    ([0,0,0],[.3,0,0]),([1,0,0],[0,0,0]),
    ([float("nan"),0,0],[1,0,0]),([100,0,0],[.1,0,0])
])
def test_point_normal_or_impulse_invalid(normal,impulse):
    e=synthetic_packet()
    e["points"][0]["normal"]=normal
    e["points"][0]["impulse"]=impulse
    assert not screen(e)["packet_schema_valid"]

def test_fake_source_hash_or_string_substring_never_sufficient():
    event=synthetic_packet()
    assert "SOURCE_SHA_INVALID" in screen(event,source="abc")["reasons"]
    event["actors"][0]=A+"/fake"
    assert "WRONG_BODY_PAIR" in screen(event)["reasons"]

def test_bad_step_duration_or_absent_timestamp_fails():
    for invalid in (None,-.005,0,float("nan")):
        event=synthetic_packet()
        event["dt_s"]=invalid
        assert not screen(event)["packet_schema_valid"]
    event=synthetic_packet();event.pop("timestamp_s")
    assert not screen(event)["packet_schema_valid"]

def test_count_mismatch_rejected():
    event=synthetic_packet()
    event["native_contact_count"]=2
    assert "CONTACT_POINT_COUNT_MISMATCH" in screen(event)["reasons"]

def test_existing_authentic_v119_six_runs_are_negative_evidence_only():
    source=Path("/home/you/projects/kinetickick-wt-v119")
    if not (source/"reports/v119_native_backend_master.audit.json").exists():
        pytest.skip("Ignored real old source absent")
    result=m.audit_v119(source)
    assert result["actual_isolated_physx_runs"]==6
    assert result["actual_physics_steps"]==1080
    assert result["native_pair_positive_certifications"]==0
    assert result["original_G1_wrist_hip_pair_unknown"] is True
    assert result["original_G1_new_physx_episodes"]==0
    assert all(x["real_200hz_steps"]==180 for x in result["per_case"])

def test_malformed_source_no_silent_fallback(tmp_path):
    with pytest.raises(Exception):m.audit_v119(tmp_path)
