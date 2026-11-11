"""Postregistered independent no-fake-native-witness and shot margin checks."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v085_independent_native_witness import exact_segment_match,matching_pair,positive_geometry,assess
from v085_cross_scene_margins import THRESHOLDS

def event(pair=("left_wrist_yaw_link","left_hip_roll_link")):
    return dict(actor_paths=[f"/env/Robot/{pair[0]}",f"/env/Robot/{pair[1]}"],
        collider_paths=[f"/env/Robot/{pair[0]}/shape",f"/env/Robot/{pair[1]}/shape"],
        contacts=[dict(point_w=[0,0,1],normal_w=[0,0,1],
                       impulse_w=[0,1,0],impulse_scalar=None)])

def test_exact_body_path_not_prefix_mismatch():
    assert exact_segment_match("/Robot/left_wrist_yaw_link", "left_wrist_yaw_link")
    assert not exact_segment_match("/Robot/left_wrist_yaw_link_extra", "left_wrist_yaw_link")
    x=event()
    assert matching_pair(x,("left_wrist_yaw_link","left_hip_roll_link"))
    x["actor_paths"][0]+="_extra"
    x["collider_paths"][0]=x["collider_paths"][0].replace("left_wrist_yaw_link","left_wrist_yaw_link_extra")
    assert not matching_pair(x,("left_wrist_yaw_link","left_hip_roll_link"))

def test_positive_impulse_requires_finite_point_normal():
    x=event()
    assert positive_geometry(x)
    x["contacts"][0]["impulse_w"]=[0,0,0]
    assert not positive_geometry(x)
    x["contacts"][0]["impulse_scalar"]=2
    assert positive_geometry(x)
    x["contacts"][0]["normal_w"]=[float("nan"),0,1]
    assert not positive_geometry(x)

def test_measured_ball_contact_without_native_football_is_uncalibrated():
    x={"records":[{"contact_time_s":1.59}],
       "traces":[[{"time_s":1.59,"contact_force_N":125.0}]],
       "physx_native_report_audit_v085":{
           "errors":[],"saved_interesting_events":[],"poll_calls":1100,
           "callback_calls":0,"callback_headers":0,"poll_headers":0}}
    result=assess(x)
    assert result["original_ball_contact_event"]
    assert not result["native_positive_control_calibrated"]
    assert result["conclusion"].startswith("UNQUALIFIED")
    assert not result["direct_native_wrist_hip_pair_certified"]

def test_observed_football_only_still_wrist_hip_unknown():
    x={"records":[{"contact_time_s":1.59}],
       "traces":[[{"time_s":1.59,"contact_force_N":145.0}]],
       "physx_native_report_audit_v085":{
           "errors":[],"saved_interesting_events":[event(("right_ankle_roll_link","SoccerBall"))],
           "poll_calls":1100,"callback_calls":1}}
    r=assess(x)
    assert r["native_positive_control_calibrated"]
    assert not r["direct_native_wrist_hip_pair_certified"]
    assert "STILL_UNKNOWN" in r["conclusion"]

def test_descriptive_thresholds_never_change_strict_gate():
    assert THRESHOLDS==(3.5,5.,10.,15.)
