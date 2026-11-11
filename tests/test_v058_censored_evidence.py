"""Posthoc CPU original G1 contact-truth and 200 Hz fail-closed tests."""
import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from summarize_v058_censored_g1 import pair_points,summarize

def valid_event():
    base="/World/envs/env_0/Robot/"
    actors=[base+"pelvis",base+"right_hip_roll_link"]
    return {"time_s":1.66,"actor_paths":actors,"collider_paths":[a+"/collisions" for a in actors],
            "point_sample":[{"point_w":[0.1,0.2,0.3],"normal_w":[1.,0.,0.],"separation_m":-0.00001}]}

def valid_native():
    return {"errors":[],"dropped_events":0,"report_capture_complete":True,
            "saved_interesting_events":[valid_event()]}

def fake_raw():
    ts=[round(.005*i,6) for i in range(1,701)]
    frames=[{"time_s":t,"joint_vel":[.2]*29,
             "support_foot_force_N":{"left":100.,"right":100.},
             "nonfoot_contact_N":0.} for t in ts]
    dy=[{"time_s":t,"com_w":[0.,0.,.75],
         "left_wrist_hip_link_origin_distance_m":.3} for t in ts]
    return {"v057_requested_physics_backend":"cpu","root_fixed":False,
            "joint_names":[f"joint_{i}" for i in range(29)],
            "joint_velocity_limits":[[1.]*29],"traces":[frames],
            "physx_pair_dynamics_v055":{"frames":dy},
            "records":[{"contact_time_s":1.6,"requested_surface_impulse_fraction":.98,"goal":True}],
            "physx_native_report_v056":valid_native(),
            "termination_reasons":[[]],"stride_shot_success":True}

def test_real_collider_paths_with_native_point():
    assert len(pair_points(valid_native()))==1

@pytest.mark.parametrize("field,value",[
    ("errors",["tampered"]),("dropped_events",1),("report_capture_complete",False)])
def test_native_incomplete_fails_closed(field,value):
    a=valid_native();a[field]=value
    assert not pair_points(a)

def test_not_other_world_or_mismatched_shape_path():
    a=valid_native()
    a["saved_interesting_events"][0]["collider_paths"][0]="/World/Other/collisions"
    assert not pair_points(a)

def test_not_zero_normal_or_missing_manifold():
    a=valid_native();a["saved_interesting_events"][0]["point_sample"][0]["normal_w"]=[0.,0.,0.]
    assert not pair_points(a)
    a=valid_native();a["saved_interesting_events"][0]["point_sample"]=[]
    assert not pair_points(a)

def test_not_other_g1_pair():
    assert not pair_points(valid_native(),{"left_wrist_yaw_link","left_hip_roll_link"})

def test_full_200hz_29dof_and_actual_recovery():
    r=summarize(fake_raw())
    assert r["post_1p25_complete"] and r["pre_0p4_complete"]
    assert r["native_G1_pelvis_right_hip_certified"]
    assert not r["native_G1_left_wrist_hip_certified"]
    assert r["strict_success"]

def test_terminal_status_is_not_misreported_as_200hz():
    a=fake_raw()
    terminal=dict(a["traces"][0][-1],time_s=a["traces"][0][-1]["time_s"]+.000001,terminated=True)
    a["traces"][0].append(terminal)
    r=summarize(a)
    assert r["terminal_status_marker_not_200hz"] is True
    assert r["trace_frames"]==700
    assert r["trace_frames_including_terminal_marker"]==701

def test_early_stop_censors_without_fabricated_recovery():
    a=fake_raw()
    a["traces"][0]=[f for f in a["traces"][0] if f["time_s"]<=1.67]
    a["physx_pair_dynamics_v055"]["frames"]=[f for f in a["physx_pair_dynamics_v055"]["frames"] if f["time_s"]<=1.67]
    r=summarize(a)
    assert r["censored_by_early_stop"] and not r["post_1p25_complete"]
    assert r["post_observed_s"]<.1

def test_not_gpu_cpu_or_fixed_root():
    for field,value in (("v057_requested_physics_backend","cuda:0"),("root_fixed",True)):
        a=fake_raw();a[field]=value
        with pytest.raises(ValueError):summarize(a)

def test_invalid_or_missing_29dof_fails_closed():
    a=fake_raw();a["traces"][0][0]["joint_vel"]=[0.]*28
    with pytest.raises(ValueError):summarize(a)

def test_non_monotone_or_wrong_rate_fails_closed():
    a=fake_raw();a["traces"][0][10]["time_s"]=a["traces"][0][9]["time_s"]
    with pytest.raises(ValueError):summarize(a)

def test_original_joint_limit_must_be_positive():
    a=fake_raw();a["joint_velocity_limits"][0][4]=0.
    with pytest.raises(ValueError):summarize(a)
