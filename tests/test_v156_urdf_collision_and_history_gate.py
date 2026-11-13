"""Original URDF collision envelope & original v54 same-step evidence rejections."""
from __future__ import annotations
import copy
import numpy as np
import pytest
from v156_urdf_collision_footprint_audit import (
    footprint_audit, foot_collision_bbox, rpy_rot,
)
from v156_real_urdf_pin_model import frozen, urdf_path
from v156_independent_runtime_and_history_gate import (
    validate_state_admission, independent_real_history_audit,
)

@pytest.fixture(scope="module")
def facts():
    return footprint_audit()

@pytest.mark.parametrize("foot",["left_ankle_roll_link","right_ankle_roll_link"])
def test_original_ankle_collision_primitives_not_v155_fabricated_patch(facts,foot):
    info=facts["feet"][foot]
    assert info["collider_count"]==7
    assert info["v155_unmeasured_corner_count_outside_urdf_collision_outer_bbox"]==4
    assert info["enclosing_min_xyz_m"][0]==pytest.approx(-.054,abs=1e-6)
    assert info["enclosing_max_xyz_m"][0]==pytest.approx(.132,abs=1e-6)
    assert info["enclosing_min_xyz_m"][1]==pytest.approx(-.036,abs=1e-9)
    assert info["enclosing_max_xyz_m"][1]==pytest.approx(.036,abs=1e-9)
    assert info["enclosing_min_xyz_m"][2]<-.03
    assert info["enclosing_max_xyz_m"][2]<0
    assert info["outer_bbox_not_true_support_polygon"]
    assert info["no_physx_ground_calibration"]

@pytest.mark.parametrize("foot",["left_ankle_roll_link","right_ankle_roll_link"])
def test_bounding_is_enclosing_for_every_original_cylinder(facts,foot):
    info=facts["feet"][foot]
    lo=info["enclosing_min_xyz_m"];hi=info["enclosing_max_xyz_m"]
    for c in info["sources"]:
        center=c["center_xyz_m"];half=c["bbox_halfwidth_m"]
        for i in range(3):
            assert center[i]-half[i]>=lo[i]-1e-10
            assert center[i]+half[i]<=hi[i]+1e-10
        assert np.linalg.norm(c["axis_link"])==pytest.approx(1,abs=1e-10)

def test_original_urdf_same_two_foot_geometry_and_not_contact_certificate(facts):
    left=facts["feet"]["left_ankle_roll_link"]
    right=facts["feet"]["right_ankle_roll_link"]
    assert left["enclosing_min_xyz_m"]==right["enclosing_min_xyz_m"]
    assert left["enclosing_max_xyz_m"]==right["enclosing_max_xyz_m"]
    assert facts["collision_outer_bounds_are_not_physical_ground_support_truth"]
    assert facts["foot_contact_sensor_is_not_ground_unique_grf"]
    assert facts["real_physx_episode_count"]==0

def test_schema_rejects_missing_actual_base_pose():
    sample={"joint_pos":[0.]*29,"joint_vel":[0.]*29}
    with pytest.raises(ValueError,match="MISSING_REAL_6D_ROOT:root_quat"):
        validate_state_admission(sample)

@pytest.mark.parametrize("edit,match",[
    ("wrong_quaternion","BAD_RUNTIME_QUATERNION"),
    ("not_normalized","BAD_RUNTIME_QUATERNION"),
    ("bad_base_velocity","BAD_WORLD_BASE_VELOCITY"),
    ("bad_shakey_sha","INVALID_SAME_STEP_SOURCE_SHA"),
    ("complete_but_unverified","SCHEMA_ONLY_NO_NATIVE_RUNTIME_VERIFICATION"),
    ("missing_original29","MISSING_ORIGINAL_29"),
])
def test_adversarial_false_real_state_injection_always_rejected(edit,match):
    row={
        "joint_pos":[0.]*29,"joint_vel":[0.]*29,
        "root_quat_wxyz":[1.,0.,0.,0.],
        "root_lin_vel_world":[0.,0.,0.],
        "root_ang_vel_world":[0.,0.,0.],
        "same_step_source_sha256":"0"*64,
    }
    if edit=="wrong_quaternion":row["root_quat_wxyz"]=[0]*3
    if edit=="not_normalized":row["root_quat_wxyz"]=[1.,2.,3.,4.]
    if edit=="bad_base_velocity":row["root_ang_vel_world"]=[0,0,np.nan]
    if edit=="bad_shakey_sha":row["same_step_source_sha256"]="x"
    if edit=="missing_original29":del row["joint_vel"]
    with pytest.raises(ValueError,match=match):
        validate_state_admission(row)

def test_history_all_actual_200hz_and_unknown_manifold():
    result=independent_real_history_audit()
    assert result["actual_historical_physx_episodes"]==3
    assert result["actual_original_200hz_states"]==6628
    assert result["original_29dof_event_window_states"]==993
    assert result["early_terminal_real_steps"]==81
    assert result["valid_35d_historical_pin_reconstruction_steps"]==0
    assert result["real_foot_ball_contact_manifold"]=="UNKNOWN"
    assert result["new_physx_episodes"]==0
    for result_row in result["runs"].values():
        assert result_row["admissible_full_35d_observed_runtime_steps"]==0
        assert result_row["admission_rejection_counts"]["V156_MISSING_REAL_6D_ROOT:root_quat_wxyz"]==result_row["real_200hz_rows"]

@pytest.mark.parametrize("angle",[-np.pi/2,0,np.pi/2])
def test_rpy_rotation_is_orthonormal(angle):
    R=rpy_rot((.3,angle,.7))
    assert np.max(np.abs(R.T@R-np.eye(3)))<1e-12
    assert np.linalg.det(R)==pytest.approx(1,abs=1e-12)

def test_changed_urdf_foot_link_or_shape_cannot_pass(tmp_path):
    path=tmp_path/"fragment.xml"
    path.write_text('<robot name="g1"><link name="left_ankle_roll_link"><collision><origin xyz="0 0 0"/><geometry><box size=".1 .1 .1"/></geometry></collision></link></robot>')
    with pytest.raises(ValueError,match="UNACCOUNTED_NONCYLINDER"):
        foot_collision_bbox(path,"left_ankle_roll_link")
    with pytest.raises(ValueError,match="MISSING_ANKLE_COLLISION"):
        foot_collision_bbox(path,"right_ankle_roll_link")

def test_frozen_version_is_consistent_and_original_source_exists():
    cfg=frozen()
    assert cfg["required_pinocchio_version"]=="2.7.0"
    assert cfg["source_urdf_sha256"]==footprint_audit()["source_urdf_sha256"]
    assert urdf_path().is_file()
