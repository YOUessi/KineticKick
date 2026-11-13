"""v134 full field OFF/ON comparator: CPU behavior, not physics source proof."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v134_native_contact_vector_gate as v

FIELDS = ("step", "q", "dq", "base_pose", "base_twist", "com_world",
          "left_foot_net_force", "right_foot_net_force",
          "nonfoot_net_force", "ball_pose", "ball_twist", "motor_torque")


def frame(step=1):
    return {
        "step": step,
        "q": [0.01 * i for i in range(29)],
        "dq": [0.002 * i for i in range(29)],
        "base_pose": {"position": [1., 2., 3.], "quaternion": [1.,0.,0.,0.]},
        "base_twist": {"linear": [1.7,0.,0.], "angular":[0.,0.,0.]},
        "com_world": [0.,0.,.7],
        "left_foot_net_force": [0.,0.,300.],
        "right_foot_net_force": [0.,0.,0.],
        "nonfoot_net_force": [0.,0.,0.],
        "ball_pose": [3.,0.,.12],
        "ball_twist": [0.,0.,0.],
        "motor_torque": [0.] * 29,
    }


def test_frozen_field_equality_records_no_physics_proof():
    a = [frame(1), frame(2)]
    b = copy.deepcopy(a)
    x = v.exact_off_on_physics_equality(a, b, required_fields=FIELDS)
    assert x["exact_common_physics_state_equality"]
    assert x["checked_frames"] == 2
    assert x["checked_physical_fields_each_frame"] == len(FIELDS)
    assert not x["real_source_provenance_certified"]
    assert not x["native_pair_certified"]


@pytest.mark.parametrize("field,value", [
    ("q", [1.] * 29),
    ("dq", [1.] * 29),
    ("com_world", [1., 0., .7]),
    ("ball_twist", [1., 2., 3.]),
    ("left_foot_net_force", [0., 0., 301.]),
    ("right_foot_net_force", [0., 0., 1.]),
    ("nonfoot_net_force", [0., 0., 0.1]),
    ("motor_torque", [5.] * 29),
    ("base_twist", {"linear":[1.7,.01,0.], "angular":[0.,0.,0.]}),
])
def test_any_common_physics_state_mutation_detected(field, value):
    a = [frame(1), frame(2)]
    b = copy.deepcopy(a)
    b[1][field] = value
    r = v.exact_off_on_physics_equality(a, b, required_fields=FIELDS)
    assert not r["exact_common_physics_state_equality"]
    assert r["mismatch_frame_indices"] == [1]
    assert r["mismatch_frame_count"] == 1


def test_different_traces_length_is_failure():
    assert not v.exact_off_on_physics_equality(
        [frame()], [frame(), frame(2)], required_fields=FIELDS
    )["exact_common_physics_state_equality"]


def test_empty_traces_do_not_pass():
    assert not v.exact_off_on_physics_equality(
        [], [], required_fields=FIELDS
    )["exact_common_physics_state_equality"]


def test_required_physx_fields_must_be_complete():
    with pytest.raises(ValueError, match="INCOMPLETE_FULL_COMMON"):
        v.exact_off_on_physics_equality([frame()], [frame()],
                                         required_fields=("q", "dq"))
    with pytest.raises(ValueError, match="DUPLICATED"):
        v.exact_off_on_physics_equality([frame()], [frame()],
                                         required_fields=("q", "q"))


def test_nan_is_never_equal_to_nan_for_provenance():
    a = [frame()]
    b = copy.deepcopy(a)
    a[0]["q"][4] = float("nan")
    b[0]["q"][4] = float("nan")
    with pytest.raises(ValueError, match="NONFINITE"):
        v.exact_off_on_physics_equality(a, b, required_fields=FIELDS)


def test_no_live_physics_authorization_api_exposed():
    assert not hasattr(v, "apply_motor_commands")
    assert not hasattr(v, "activate_actuators")
    assert not hasattr(v, "train_human_policy")


def test_equal_but_missing_root_or_ball_state_never_suffices():
    shallow = [{"step": 1, "q": [0.] * 29, "dq": [0.] * 29}]
    with pytest.raises(ValueError, match="INCOMPLETE_FULL_COMMON"):
        v.exact_off_on_physics_equality(
            shallow, copy.deepcopy(shallow), required_fields=("step","q","dq"))


@pytest.mark.parametrize("field,length", [("q", 28), ("dq", 30), ("motor_torque", 1)])
def test_equal_invalid_original29_joint_fields_rejected(field, length):
    bad = frame()
    bad[field] = [0.] * length
    with pytest.raises(ValueError, match="ORIGINAL29"):
        v.exact_off_on_physics_equality(
            [bad], [copy.deepcopy(bad)], required_fields=FIELDS)


def test_equal_but_bad_root_quaternion_is_rejected():
    bad=frame()
    bad["base_pose"]["quaternion"]=[1.,0.,0.]
    with pytest.raises(ValueError,match="ROOT_STATE"):
        v.exact_off_on_physics_equality(
            [bad],[copy.deepcopy(bad)],required_fields=FIELDS)
