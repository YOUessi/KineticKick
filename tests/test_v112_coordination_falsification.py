"""Independent safeguards for the *post-hoc* W/X/Y matched control comparisons."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v112_coordination_falsification as m

def trial(shift=0.0):
    times=np.arange(1,2801)*.005
    names=(["left_hip_pitch_joint","right_hip_pitch_joint","waist_yaw_joint",
      "left_knee_joint","right_knee_joint","left_ankle_pitch_joint","right_ankle_pitch_joint",
      "left_shoulder_pitch_joint","right_shoulder_pitch_joint","left_elbow_joint","right_elbow_joint",
      "left_wrist_roll_joint","right_wrist_roll_joint"] +
      ["left_hip_roll_joint","right_hip_roll_joint","left_hip_yaw_joint","right_hip_yaw_joint",
       "left_ankle_roll_joint","right_ankle_roll_joint","waist_roll_joint","waist_pitch_joint",
       "left_shoulder_roll_joint","right_shoulder_roll_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint",
       "left_wrist_pitch_joint","right_wrist_pitch_joint","left_wrist_yaw_joint","right_wrist_yaw_joint"])
    traces=[{"time_s":float(t),"joint_vel":[float(t+shift)]*29,
      "whole_body_com_xyz":[float(t),0.0,0.72],"support_foot_force_N":{"left":100.,"right":80.}} for t in times]
    return {"root_fixed":False,"joint_names":names,"traces":[traces]}
def test_200hz_measured_window_and_support_impulse():
    x=m.measured(trial(),1.6)
    assert x["full_14s"] and x["pre_0p4_s_rows"]==80
    assert x["post_1p25_s_rows"]==251
    assert x["pre_contact_force_proxy_impulse_Ns"]["left"]==pytest.approx(40)
    assert len(x["group_speed_peak_before_contact"])==5
def test_29_joint_pair_onset_and_no_false_claim():
    x=m.pairwise(trial(),trial(.1),1.6)
    assert x["first_actual_qdot_difference_gt_1e-7_s"]==pytest.approx(.005)
    assert x["pre_0p4_s_joint_speed_rms_delta_radps"]==pytest.approx(.1)
    assert x["pre_0p4_s_distinct_physical_frames"]==80
def test_equal_velocity_not_faked_as_difference():
    x=m.pairwise(trial(),trial(),1.6)
    assert x["first_actual_qdot_difference_gt_1e-7_s"] is None
    assert x["pre_0p4_s_distinct_physical_frames"]==0
def test_fail_closed_non_200hz():
    a=trial();a["traces"][0][15]["time_s"]+=.002
    with pytest.raises(ValueError,match="200HZ"):
        m.measured(a,1.6)
def test_fail_closed_different_joint_order():
    a,b=trial(),trial()
    b["joint_names"].reverse()
    with pytest.raises(ValueError,match="JOINT_ORDER"):
        m.pairwise(a,b,1.6)
def test_no_complete_post_contact_refused():
    a=trial();a["traces"][0]=a["traces"][0][:340]
    with pytest.raises(ValueError,match="COMPLETE_REAL_200HZ"):
        m.measured(a,1.6)
def test_preregistered_real_runs_verify_sha_and_parity():
    a=m.audit()
    assert a["n_original_physx_repeats"]==10
    assert a["independent_controller_scene_combinations"]==6
    assert a["strict_pass_independent"]==2
    assert len(a["preregistered_scene_pairs"])==3
    assert not a["native_wrist_hip_manifold_certified"]
