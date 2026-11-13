"""Copy-only native sidecar mocks: no test can establish real PhysX truth."""
import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from v142_runtime_bridge import NativeReadOnlyEpisodeSidecar
from test_v142_native_contact_and_human_qualification import state, packet, owner_map


def sidecar():
    ids={1:"/World/Robot/left_wrist_yaw_link",2:"/World/Robot/left_hip_roll_link",
         3:"/World/Robot/left_wrist_yaw_link/collision",
         4:"/World/Robot/left_hip_roll_link/collision"}
    return NativeReadOnlyEpisodeSidecar(stage_id=42,owner_map=owner_map(),decode_path=lambda k:ids[k])


def real_shape_header():
    return [{"stage_id":42,"actor0":1,"actor1":2,
             "collider0":3,"collider1":4,
             "type":"CONTACT_FOUND","contact_data_offset":0,"num_contact_data":1}]


def real_shape_point():
    return [{"position":[0.,0.,0.],"normal":[1.,0.,0.],
             "impulse":[1.25,0.,0.],"separation":-0.001}]


def test_two_physics_steps_one_contact_and_copy_only():
    s=sidecar()
    headers=real_shape_header()
    points=real_shape_point()
    s.before_physics_step(1)
    s.on_native_contact(headers,points)
    points[0]["impulse"][0]=10.
    s.after_physics_step(state(1))
    s.before_physics_step(2)
    s.after_physics_step(state(2))
    result=s.review()
    assert result["original_physics_steps_input"]==2
    assert result["native_packets_copied"]==1
    assert result["native_manifold_points_copied"]==1
    assert result["g1_wrist_hip_candidate_events"][0]["total_vector_magnitude_Ns"]==1.25
    assert not result["native_pair_manifold_certified"]
    assert not result["read_only_off_on_invariance_certified"]
    assert result["motor_execution_authorized"] is False


def test_callback_before_clock_rejected_as_error_not_state_mutation():
    s=sidecar()
    s.on_native_contact(real_shape_header(),real_shape_point())
    s.before_physics_step(1)
    s.after_physics_step(state(1))
    x=s.review()
    assert x["callback_errors"]==["CONTACT_OUTSIDE_OR_UNCALIBRATED_STEP"]
    assert x["native_packets_copied"]==0


@pytest.mark.parametrize("wrong",[
    None, 0, 2, -1, True
])
def test_before_step_requires_consecutive_physics(wrong):
    s=sidecar()
    if wrong is None:
        s.before_physics_step(1)
        wrong=1
    with pytest.raises(ValueError,match="PRE_STEP_MISSING_OR_REORDERED"):
        s.before_physics_step(wrong)


def test_physics_step_not_callback_counter():
    s=sidecar()
    s.before_physics_step(1)
    for _ in range(5):
        s.on_native_contact(real_shape_header(),real_shape_point())
    s.after_physics_step(state(1))
    assert s.review()["original_physics_steps_input"]==1
    assert s.review()["native_packets_copied"]==5


def test_out_of_order_state_rejected():
    s=sidecar()
    s.before_physics_step(1)
    with pytest.raises(ValueError,match="POST_STEP_CLOCK_MISMATCH"):
        s.after_physics_step(state(2))


def test_no_state_no_review():
    s=sidecar()
    with pytest.raises(ValueError,match="INCOMPLETE_REAL"):
        s.review()


def test_mid_step_not_reported_as_complete():
    s=sidecar()
    s.before_physics_step(1)
    with pytest.raises(ValueError,match="INCOMPLETE_REAL"):
        s.review()


def test_uncalibrated_duplicate_subscription_rejected():
    s=sidecar()
    handle=object()
    assert s.attach_existing_physx_read_only(subscribe=lambda cb:handle) is handle
    with pytest.raises(ValueError,match="DUPLICATE"):
        s.attach_existing_physx_read_only(subscribe=lambda cb:object())


def test_failed_subscription_disclosed():
    s=sidecar()
    with pytest.raises(ValueError,match="NO_HANDLE"):
        s.attach_existing_physx_read_only(subscribe=lambda cb:None)


def test_bad_native_contact_buffer_recorded_without_killing_sim():
    s=sidecar()
    s.before_physics_step(1)
    wrong=real_shape_point()
    wrong[0]["impulse"]=3.5
    s.on_native_contact(real_shape_header(),wrong)
    s.after_physics_step(state(1))
    x=s.review()
    assert "INVALID_IMPULSE_VEC3" in x["callback_errors"][0]
    assert not x["native_pair_manifold_certified"]


def test_write_once_and_preserved_state_rows(tmp_path):
    s=sidecar()
    s.before_physics_step(1)
    original=state(1)
    s.after_physics_step(original)
    original["q"][0]=1e4
    p=tmp_path/"physical_capture.json"
    result=s.persist_write_once(p)
    assert result["review"]["original_physics_steps_input"]==1
    assert '"q": [' in p.read_text()
    import json
    assert json.loads(p.read_text())["full_original_physics_rows"][0]["q"][0]==0.
    with pytest.raises(FileExistsError):
        s.persist_write_once(p)


def test_fully_empty_callback_not_positively_certified():
    s=sidecar()
    s.before_physics_step(1)
    s.after_physics_step(state(1))
    x=s.review()
    assert x["native_packets_copied"]==0
    assert x["no_native_callbacks_is_no_contact_proof"] is False
    assert x["physical_kick_success"] is None


def test_stage_owner_map_is_copy_not_mutable_reference():
    owners=owner_map()
    s=NativeReadOnlyEpisodeSidecar(
        stage_id=42,owner_map=owners,decode_path=lambda x:"/World")
    owners.clear()
    assert s.owner_map


@pytest.mark.parametrize("invalid",[-1, "42", None])
def test_stage_id_requires_integer(invalid):
    with pytest.raises(ValueError,match="UNVERIFIED_STAGE_ID"):
        NativeReadOnlyEpisodeSidecar(stage_id=invalid,owner_map=owner_map(),
                                     decode_path=lambda x:"/World")


def test_no_write_to_source_and_no_sim_backend_import_until_attached():
    s=sidecar()
    assert s.subscription is None
    assert not hasattr(s,"step")
    assert not hasattr(s,"apply_action")
    assert not hasattr(s,"apply_schema")
