"""Actual immutable historical v54 200Hz state and cross-clock falsification."""
import copy
from functools import lru_cache
import math
import pytest
from v153_historical_collision_clock_audit import (
    audit_all, assert_recorded_pair, timeline_relation,
)
from v153_independent_graph_audit import independent_source_check

@lru_cache(maxsize=1)
def original():
    return audit_all()

def valid_two():
    return [
        dict(body="left_wrist_yaw_link",force_w_N=[2.,3.,4.],time_s=5.145,position_w=[0.,0.,0.]),
        dict(body="left_hip_roll_link",force_w_N=[-2.,-3.,-4.],time_s=5.145,position_w=[.1,0.,0.]),
    ]

def test_original_source_and_real_state_count():
    x=original()
    assert x["original_real_physics_200hz_rows"]==6628
    assert [r["original_real_physx_200hz_rows"] for r in x["historical_cases"]]==[2800,2800,1028]
    assert sum(r["excluded_nonstep_terminal_annotations"] for r in x["historical_cases"])==1
    assert x["new_physx_runs"]==0
    assert x["actual_left_wrist_hip_physx_contact_pair"]=="UNKNOWN"

def test_real_net_body_event_is_after_final_real_state():
    x=original()
    t=x["last_400ms_censored_trace_timeline"]
    assert t["named_body_sensor_diag_s"]>t["last_original_recorded_5ms_state_s"]
    assert t["nonfoot_net_spike_at_real_trace_s"]<t["last_original_recorded_5ms_state_s"]
    assert 9.99 < t["gap_named_to_nonfoot_spike_ms"] < 10.01
    assert 4.99 < t["gap_named_beyond_last_real_row_ms"] < 5.01
    assert not t["named_diag_has_same_step_full_original29_state"]

def test_real_observed_force_norm_and_anti_symmetry():
    x=original()
    f=x["observed_wrist_net_force_N"]
    g=x["observed_hip_net_force_N"]
    assert math.isclose(math.sqrt(sum(v*v for v in f)),25.89015328949694,abs_tol=1e-8)
    assert all(abs(a+b)<1e-10 for a,b in zip(f,g))
    assert 0.10 < x["wrist_hip_link_origin_distance_m"] < 0.13
    assert x["not_narrow_phase_contact_position"]

def test_real_full29_and_support_censored():
    x=original()
    for row in x["historical_cases"]:
        assert row["max_original29_velocity_ratio"]>=0
        assert row["max_sim_estimated_torque_ratio"]>=0
        assert row["max_com_velocity_norm_mps"]>=0
        assert row["support_force_is_aggregate_contact_not_isolated_grf"]
    t=x["last_400ms_censored_trace_timeline"]
    assert t["last_400ms_actual_step_count"]==81
    assert t["last_400ms_peak_29dof_joint_speed"]>=0
    assert t["root_pose_does_not_prove_fall"]

def test_independent_original_file_checksum_and_censoring():
    o=independent_source_check(original())
    assert o["real_5ms_steps"]==1028
    assert o["excluded_terminal_notes"]==1
    assert o["named_net_sensor_is_later_than_final_real_state"]
    assert not o["native_contact_manifold_certification"]

@pytest.mark.parametrize("bad_field",("body","force","clock","geometry"))
def test_named_body_bad_controls(bad_field):
    two=valid_two()
    if bad_field=="body":two[0]["body"]="left_ankle_roll_link"
    if bad_field=="force":two[0]["force_w_N"][0]=99.
    if bad_field=="clock":two[1]["time_s"]=5.12
    if bad_field=="geometry":two[0]["position_w"]=[math.nan,0,0]
    with pytest.raises(ValueError):
        assert_recorded_pair(two)

def test_named_same_clock_does_not_prove_collider_path():
    x=assert_recorded_pair(valid_two())
    assert x["named_pair_verification"]=="NET_FORCE_ONLY"
    assert x["possible_net_contact_paths_are_not_identified"]

def test_same_step_clock_fabrication_rejected():
    with pytest.raises(ValueError,match="NOT_CENSORED"):
        timeline_relation(5.135,5.14,5.14,.005)

def test_insufficient_clock_gap_rejected():
    with pytest.raises(ValueError,match="CROSS_STEP"):
        timeline_relation(5.1395,5.14,5.14001,.005)

def test_nonfinite_clock_rejected():
    with pytest.raises(ValueError,match="NONFINITE"):
        timeline_relation(5.135,5.14,math.inf,.005)

def test_source_data_falsification_rejected():
    x=copy.deepcopy(original())
    x["historical_failed_trial_sha256"]="0"*64
    with pytest.raises(ValueError,match="SOURCE_CHANGED"):
        independent_source_check(x)

def test_source_count_falsification_rejected():
    x=copy.deepcopy(original())
    x["original_real_physics_200hz_rows"]=6629
    with pytest.raises(ValueError,match="HISTORY_INCOMPLETE"):
        independent_source_check(x)
