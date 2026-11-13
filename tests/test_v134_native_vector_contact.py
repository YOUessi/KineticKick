"""v134 native vector PhysX ABI CPU contract tests; no real collision attestation."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as N

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v134_native_contact_vector_gate as v134
from wholebody_safety_interface_v055 import G1_JOINT_ORDER


def fixtures(event="CONTACT_FOUND", impulse=(0.6, -0.8, 0.0)):
    h = N(actor0="a", actor1="b", collider0="c", collider1="d",
          contact_data_offset=0, num_contact_data=1,
          stage_id=[23], type=event)
    cp = N(position=[1, 2, 3], normal=[0, -1, 0],
           impulse=list(impulse), separation=-0.0002,
           face_index0=13, face_index1=4294967295)
    d = {"a": "/World/G1/left_wrist_yaw_link",
         "b": "/World/G1/left_hip_roll_link",
         "c": "/World/G1/left_wrist_yaw_link/collisions/cylinder",
         "d": "/World/G1/left_hip_roll_link/collisions/cylinder"}
    return h, cp, d


def packet(event="CONTACT_FOUND", impulse=(0.6, -0.8, 0)):
    h, cp, d = fixtures(event, impulse)
    return v134.copy_contact_vector_frame([h], [cp], decode_path=d.__getitem__,
                                          step_hint=1027, expected_stage_id=23)


def test_frozen_sha_and_semantics():
    cfg = v134.frozen_protocol()
    assert cfg["version"] == "v0.1.134"
    assert cfg["nvidia_contact_doc_date"] == "2026-07-23"
    assert cfg["strict_kick_gates"]["nonfoot_force_N_max"] == 20
    assert cfg["native_test_rollouts_while_foreign_gpu_busy"] == 0


def test_correct_native_3d_impulse_and_projection():
    p = packet()
    hit = p["headers"][0]["manifold_points"][0]
    assert hit["impulse_world_vector_Ns"] == [.6, -.8, 0.0]
    assert hit["impulse_vector_magnitude_Ns"] == pytest.approx(1.)
    assert hit["signed_impulse_projection_along_normal_Ns"] == pytest.approx(.8)
    assert hit["face_index0"] == 13
    assert hit["face_index1"] == 4294967295
    assert p["time_hint_s"] == 5.135
    assert not p["native_pair_certified"]
    assert not p["motor_execution_authorized"]


def test_borrowed_contact_buffer_deep_copy():
    h, cp, d = fixtures()
    p = v134.copy_contact_vector_frame([h], [cp], decode_path=d.__getitem__,
                                        step_hint=3, expected_stage_id=23)
    cp.impulse[0] = -100
    cp.position[:] = [0, 0, 0]
    h.collider0 = "b"
    assert p["headers"][0]["manifold_points"][0]["impulse_world_vector_Ns"][0] == .6
    assert p["headers"][0]["manifold_points"][0]["position_world"] == [1., 2., 3.]
    assert p["headers"][0]["collider_paths"][0].endswith("cylinder")


@pytest.mark.parametrize("bad", [1., -1., float("nan"), None, [1., 2.], [1., 2., 3., 4.]])
def test_real_physx_requires_vector_impulse_not_scalar(bad):
    h, cp, d = fixtures()
    cp.impulse = bad
    with pytest.raises(ValueError, match="IMPULSE|NONFINITE"):
        v134.copy_contact_vector_frame([h], [cp], decode_path=d.__getitem__,
                                        step_hint=3, expected_stage_id=23)


@pytest.mark.parametrize("bad", [[float("inf"), 0, 0], [0, float("nan"), 0],
                                  [None, 1, 2], [True, 2, 3]])
def test_malformed_vector_impulse_rejected(bad):
    h, cp, d = fixtures()
    cp.impulse = bad
    with pytest.raises(ValueError):
        v134.copy_contact_vector_frame([h], [cp], decode_path=d.__getitem__,
                                        step_hint=3, expected_stage_id=23)


def test_negative_impulse_component_valid_real_vector():
    p = packet(impulse=(-2, 0, 1))
    assert p["headers"][0]["manifold_points"][0]["impulse_vector_magnitude_Ns"] == pytest.approx(5**0.5)


def test_lost_header_zero_points_is_not_negative_contact_proof():
    h, cp, d = fixtures("ContactEventType.CONTACT_LOST")
    h.num_contact_data = 0
    p = v134.copy_contact_vector_frame([h], [], decode_path=d.__getitem__,
                                        step_hint=1029, expected_stage_id=23)
    assert p["headers"][0]["event_type"] == "CONTACT_LOST"
    assert p["headers"][0]["contact_count"] == 0
    assert p["headers"][0]["zero_point_is_no_collision_proof"] is False
    assert p["native_pair_certified"] is False


def test_lost_header_with_points_rejected():
    h, cp, d = fixtures("CONTACT_LOST")
    with pytest.raises(ValueError, match="LOST_EVENT"):
        v134.copy_contact_vector_frame([h], [cp], decode_path=d.__getitem__,
                                        step_hint=3, expected_stage_id=23)


@pytest.mark.parametrize("field,bad", [("contact_data_offset",-1), ("num_contact_data",-1),
                                       ("contact_data_offset",2), ("num_contact_data",2)])
def test_header_slice_bounds(field,bad):
    h, cp, d = fixtures()
    setattr(h, field, bad)
    with pytest.raises(ValueError, match="OFFSET|COUNT|BOUNDS"):
        v134.copy_contact_vector_frame([h], [cp], decode_path=d.__getitem__,
                                        step_hint=3, expected_stage_id=23)


def test_nonoverlap_and_all_contact_points_consumed():
    h, cp, d = fixtures()
    two = copy.deepcopy(h)
    with pytest.raises(ValueError, match="OVERLAP"):
        v134.copy_contact_vector_frame([h, two], [cp], decode_path=d.__getitem__,
                                        step_hint=3, expected_stage_id=23)
    with pytest.raises(ValueError, match="UNREFERENCED"):
        v134.copy_contact_vector_frame([h], [cp, copy.deepcopy(cp)],
                                        decode_path=d.__getitem__,
                                        step_hint=3, expected_stage_id=23)


@pytest.mark.parametrize("bad", [True, -1, 3.5, None, [1,2]])
def test_bad_stage_ids(bad):
    h, cp, d = fixtures()
    h.stage_id = bad
    with pytest.raises(ValueError, match="STAGE"):
        v134.copy_contact_vector_frame([h], [cp], decode_path=d.__getitem__,
                                        step_hint=3, expected_stage_id=23)


@pytest.mark.parametrize("bad", [0, -1, True, 2.4])
def test_bad_clock(bad):
    h, cp, d = fixtures()
    with pytest.raises(ValueError, match="STEP"):
        v134.copy_contact_vector_frame([h], [cp], decode_path=d.__getitem__,
                                        step_hint=bad, expected_stage_id=23)


def test_no_accidental_float_clock_or_non200hz():
    h, cp, d = fixtures()
    with pytest.raises(ValueError, match="NOT_200HZ"):
        v134.copy_contact_vector_frame([h], [cp], decode_path=d.__getitem__,
                                        step_hint=3, expected_stage_id=23, dt_s=.01)


@pytest.mark.parametrize("path", ["relative", "", "/x//y", "/x/../y",
                                  "/x/./y", "/x/", "/World/foo bar"])
def test_invalid_usd_path(path):
    with pytest.raises(ValueError, match="USD_PATH"):
        v134.canonical_usd_path(path)


def test_path_decoder_matches_runtime_only_lazy_contract():
    assert callable(v134.native_path_decoder)
    # Must not import unavailable pxr until actual callback runtime is live.
    assert "pxr" not in v134.__dict__


def test_exact_owner_map_review_and_unattested_even_when_matching():
    p = packet()
    h = p["headers"][0]
    m = dict(zip(h["collider_paths"], h["actor_paths"]))
    r = v134.review_colliders(p, read_only_usd_owner_map=m)[0]
    assert r["review_status"] == "UNATTESTED_MATCH"
    assert r["actor_paths_match_owner_map"]
    assert not r["native_pair_certified"]


def test_unknown_collider_owner_is_not_absence_of_contact():
    p = packet()
    r = v134.review_colliders(p, read_only_usd_owner_map={})[0]
    assert r["review_status"] == "REJECT_UNKNOWN_OR_SAME_OWNER"


def test_incorrect_owner_cannot_be_accepted():
    p = packet()
    h = p["headers"][0]
    m = {h["collider_paths"][0]: "/World/other",
         h["collider_paths"][1]: h["actor_paths"][1]}
    r = v134.review_colliders(p, read_only_usd_owner_map=m)[0]
    assert r["review_status"] == "REJECT_ACTOR_OWNER_MISMATCH"


def test_wrong_29dof_step_clocks_refused():
    r = v134.time_align_packet(packet(), step=1028, q=np.zeros(29),
                               dq=np.zeros(29), joint_names=G1_JOINT_ORDER,
                               source_step_authoritative=True)
    assert not r["review_ready"]
    assert "CONTACT_AND_STATE_DIFFERENT_PHYSICS_STEP" in r["reasons"]
    assert not r["native_pair_certified"]


def test_even_matching_synthetic_state_never_attests_physics():
    r = v134.time_align_packet(packet(), step=1027, q=np.zeros(29),
                               dq=np.zeros(29), joint_names=G1_JOINT_ORDER,
                               source_step_authoritative=True)
    assert r["review_ready"] and not r["native_pair_certified"]
    assert not r["motor_execution_authorized"]


def test_untrusted_clock_and_joint_order_block():
    r = v134.time_align_packet(packet(), step=1027, q=np.zeros(29),
                               dq=np.zeros(29), joint_names=G1_JOINT_ORDER[::-1],
                               source_step_authoritative=False)
    assert "ORIGINAL_G1_JOINT_ORDER_MISMATCH" in r["reasons"]
    assert "BACKEND_STEP_CLOCK_NOT_CALIBRATED" in r["reasons"]


def test_empty_callback_cannot_attest_no_contact():
    p = v134.copy_contact_vector_frame([], [], decode_path=lambda x: str(x),
                                        step_hint=5, expected_stage_id=23)
    assert not v134.time_align_packet(
        p, step=5, q=np.zeros(29), dq=np.zeros(29),
        joint_names=G1_JOINT_ORDER, source_step_authoritative=True)["review_ready"]


def test_copy_only_sink_with_no_isaac_runtime():
    h, cp, d = fixtures()
    sink = v134.CopyOnlyContactRecorder(callback_step=lambda: 5,
                                         stage_id=23, decode_path=d.__getitem__)
    sink.on_contact([h], [cp])
    sink.on_contact([], [])
    assert sink.review()["total_copied_manifold_points"] == 1
    assert sink.review()["total_copied_callback_frames"] == 2
    assert not sink.review()["native_pair_certified"]
    assert sink.subscription is None


def test_sink_reports_malformed_data_not_false_pass():
    h, cp, d = fixtures()
    cp.impulse = -2.
    sink = v134.CopyOnlyContactRecorder(callback_step=lambda: 1,
                                         stage_id=23, decode_path=d.__getitem__)
    sink.on_contact([h], [cp])
    assert not sink.frames
    assert "INVALID_IMPULSE_VEC3" in sink.review()["errors"][0]


def test_sink_nonmonotonic_hint_rejected():
    steps = iter([3, 2])
    sink = v134.CopyOnlyContactRecorder(callback_step=lambda: next(steps),
                                         stage_id=23, decode_path=lambda x: str(x))
    sink.on_contact([], [])
    sink.on_contact([], [])
    with pytest.raises(ValueError, match="NONMONOTONIC"):
        sink.review()


def test_historical_authentic_physics_sha_verification():
    r = v134.review_historical_original_v54()
    assert r["historical_physx_episodes_sha_verified"] == 3
    assert r["historical_real_200hz_frames_sha_verified"] == 6628
    assert r["failed_last_real_q_time_s"] == 5.140
    assert r["failed_named_contact_sensor_time_s"] == 5.145
    assert r["new_physx_episodes"] == 0
    assert not r["native_vector_manifold_truth_captured"]
