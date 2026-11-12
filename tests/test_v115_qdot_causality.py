"""Frozen source joint-limit and controller-onset falsification."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v115_qdot_causality as q
from test_v115_coupled_dynamics import physical
def mk():
    p=physical(400)
    p["records"][0]["contact_time_s"]=1.4
    p["joint_velocity_limits"]=[[10.]*29]
    for row in p["traces"][0]:
        row["joint_vel"]=[0.]*29
    return p
def test_original_nested_limits_kept():
    a=q.physical_arrays(mk())
    assert a["lim"].shape==(29,)
    assert q.violations(a)["first_true_qdot_exceed_s"] is None
def test_first_hazard_precedes_late_control_difference():
    p=mk(); b=mk()
    p["traces"][0][199]["joint_vel"][9]=12.
    b["traces"][0][199]["joint_vel"][9]=12.
    for row in b["traces"][0][239:]:
        row["joint_vel"][0]=0.02
    a=q.physical_arrays(p); z=q.physical_arrays(b)
    v=q.violations(a)
    assert v["first_true_qdot_exceed_s"]==pytest.approx(1.0)
    assert v["all_exceeding_joints"][0]["joint"]=="left_knee_joint"
    o=q.paired_intervention(a,z)
    assert o["first_observed_whole29_qdot_difference_s"]==pytest.approx(1.2)
    assert o["excess_happened_before_first_measured_controller_difference"]
    assert o["group_first_affected_s"]["left_leg"]==pytest.approx(1.2)
def test_hazard_after_intervention_not_pre_existing():
    a=mk();b=mk()
    for row in b["traces"][0][119:]:
        row["joint_vel"][0]=0.03
    a["traces"][0][199]["joint_vel"][9]=11.
    o=q.paired_intervention(q.physical_arrays(a),q.physical_arrays(b))
    assert not o["excess_happened_before_first_measured_controller_difference"]
def test_source_joint_order_immutable():
    a=mk();b=mk();b["joint_names"][0],b["joint_names"][1]=b["joint_names"][1],b["joint_names"][0]
    with pytest.raises(ValueError,match="JOINT_ORDER"):
        q.paired_intervention(q.physical_arrays(a),q.physical_arrays(b))
def test_malformed_real_physics_rejected():
    a=mk();a["traces"][0][44]["time_s"]+=.002
    with pytest.raises(ValueError,match="200HZ"):q.physical_arrays(a)
def test_nan_and_bad_frozen_limit_rejected():
    a=mk();a["joint_velocity_limits"][0][2]=0.
    with pytest.raises(ValueError,match="SOURCE_QDOT_LIMIT"):q.physical_arrays(a)
def test_free_root_mandatory():
    a=mk();a["root_fixed"]=True
    with pytest.raises(ValueError,match="FREE_ROOT"):q.physical_arrays(a)
