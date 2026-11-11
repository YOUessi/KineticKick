"""Frozen true G1 trace contract: tests use constructed data, never call mocks real PhysX."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_v061_coupling import GROUPS, RIGHT_CHAIN, summarize
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

def fixture(end=3.0):
    tr=[]
    for t in np.arange(.005,end+.001,.005):
        tr.append(dict(time_s=float(t),joint_vel=[.1]*29,joint_pos=[0.]*29,
                       support_foot_force_N=dict(left=320.,right=3.),
                       nonfoot_contact_N=0.,whole_body_com_xyz=[0.,0.,.7],
                       root_xyz=[0.,0.,.8]))
    return {"joint_names":list(G1_JOINT_ORDER),"traces":[tr],
            "records":[{"contact_time_s":1.57}],
            "joint_velocity_limits":[[1.]*29]}
def test_complete_window_real_sample_contract():
    x=summarize(fixture(3.0))
    assert x["coverage"]["before_0p40_complete"]
    assert x["coverage"]["after_1p25_complete"]
    assert not x["coverage"]["recovery_to_14s_observed"]
    assert len(x["kinetic_chain_peaks"])==3
    assert x["windows"]["before_0p40"]["max_29dof_velocity_over_original_limit"]==pytest.approx(.1)
def test_early_nonfoot_censors_postimpact():
    x=summarize(fixture(1.65))
    assert x["coverage"]["before_0p40_complete"]
    assert not x["coverage"]["after_1p25_complete"]
    assert not x["windows"]["after_1p25"]["complete"]
def test_bad_g1_joint_order_fails_closed():
    raw=fixture();raw["joint_names"][0],raw["joint_names"][1]=raw["joint_names"][1],raw["joint_names"][0]
    with pytest.raises(ValueError,match="joint order"):summarize(raw)
def test_non_200hz_fails_closed():
    raw=fixture()
    raw["traces"][0].pop(4)
    with pytest.raises(ValueError,match="200Hz"):summarize(raw)
def test_all_29_joints_covered_by_groups():
    assert set().union(*[set(group) for group in GROUPS.values()])==set(G1_JOINT_ORDER)
    assert len(RIGHT_CHAIN)==3

