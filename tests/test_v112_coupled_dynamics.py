"""Independent 200Hz coupled dynamics must remain physical, complete, and censored."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v112_200hz_coupled_audit as a

NAMES=(
 "left_hip_pitch_joint","right_hip_pitch_joint","waist_yaw_joint","left_hip_roll_joint",
 "right_hip_roll_joint","waist_roll_joint","left_hip_yaw_joint","right_hip_yaw_joint",
 "waist_pitch_joint","left_knee_joint","right_knee_joint","left_shoulder_pitch_joint",
 "right_shoulder_pitch_joint","left_ankle_pitch_joint","right_ankle_pitch_joint",
 "left_shoulder_roll_joint","right_shoulder_roll_joint","left_ankle_roll_joint",
 "right_ankle_roll_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint",
 "left_elbow_joint","right_elbow_joint","left_wrist_roll_joint","right_wrist_roll_joint",
 "left_wrist_pitch_joint","right_wrist_pitch_joint","left_wrist_yaw_joint","right_wrist_yaw_joint")
def physical(rows=400):
    ts=np.arange(1,rows+1)*.005
    return {"root_fixed":False,"joint_names":list(NAMES),
      "joint_velocity_limits":[10.]*29,"records":[{"contact_time_s":.6}],
      "traces":[[{"time_s":float(t),"physics_dt_s":.005,"joint_pos":[0.]*29,
       "joint_vel":[float(t)]*29,"whole_body_com_xyz":[float(t),0.,.7],
       "root_xyz":[0.,0.,.75],"support_foot_force_N":{"left":110.,"right":80.},
       "nonfoot_contact_N":25. if abs(t-1.0)<.001 else 0.} for t in ts]]}
def test_groups_exact_cover():
    g=a.group_indices(NAMES)
    assert sum(len(x) for x in g.values())==29 and len(g)==5
def test_duplicate_joint_fails():
    with pytest.raises(ValueError,match="29DOF"):
        a.group_indices(NAMES[:-1]+(NAMES[0],))
def test_full_contact_window_and_measured_forces():
    d=a.analyze(physical(400))
    assert d["has_full_pre_0p4_s"] and d["has_full_post_1p25_s"]
    assert d["first_nonfoot_ge20N_s"]==pytest.approx(1.,abs=1e-5)
    assert d["pre_contact"]["support_left_N"]["peak"]==pytest.approx(110)
    assert d["post_contact_observed_only"]["joint_group_qdot_rms_radps"]["left_arm"]>0
    assert not d["native_actor_collider_manifold_certified"]
    assert not d["true_GRF_CoP_and_foot_slip_witness_certified"]
def test_early_termination_is_censored_not_completed():
    d=a.analyze(physical(240))
    assert not d["has_full_post_1p25_s"]
    assert d["right_censored"] and not d["full_14s"]
def test_missing_200hz_rejected():
    p=physical(400);p["traces"][0][99]["time_s"]+=.002
    with pytest.raises(ValueError,match="200HZ"):
        a.analyze(p)
def test_duplicate_terminal_is_not_new_physics():
    p=physical(400);p["traces"][0][-1]["time_s"]=p["traces"][0][-2]["time_s"]
    with pytest.raises(ValueError,match="200HZ"):
        a.analyze(p)
def test_unqualified_data_rejected():
    with pytest.raises(ValueError,match="NOT_CERTIFIED"):
        a.analyze(physical(),certified=False)
def test_bad_real_qdot_shape_rejected():
    p=physical();p["traces"][0][0]["joint_vel"]=[0.]*28
    with pytest.raises(ValueError,match="29DOF"):
        a.analyze(p)


def test_only_identical_last_terminated_marker_is_excluded():
    import copy
    p=physical(400)
    p["traces"][0][-1]["terminated"]=False
    sentinel=copy.deepcopy(p["traces"][0][-1])
    sentinel["time_s"]+=.000001
    sentinel["terminated"]=True
    p["traces"][0].append(sentinel)
    d=a.analyze(p)
    assert d["source_raw_trace_rows"]==401
    assert d["count_real_200hz"]==400
    assert d["duplicate_terminal_sentinel_markers"]==1

def test_terminal_marker_with_new_physics_rejected():
    import copy
    p=physical(400);p["traces"][0][-1]["terminated"]=False
    sentinel=copy.deepcopy(p["traces"][0][-1])
    sentinel["time_s"]+=.000001
    sentinel["terminated"]=True
    sentinel["joint_vel"][0]+=.1
    p["traces"][0].append(sentinel)
    with pytest.raises(ValueError,match="NEW_PHYSICAL_CONTENT"):
        a.analyze(p)
