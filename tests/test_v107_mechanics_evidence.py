"""Regression tests for independent torque/support/foot-speed mechanics."""
import copy,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
sys.path.insert(0,str(ROOT/"tests"))
from test_v107_coupled_dynamics import physical
from v107_mechanics_evidence import mechanics
def sampled(rows=400):
    p=physical(rows)
    for r in p["traces"][0]:
        r["joint_torque_estimate"]=[2.]*29
        r["whole_body_com_velocity"]=[.3,0.,0.]
        r["foot_velocity_contributions"]={k:[float(v),0.,0.] for k,v in
                                          [("hip",.3),("knee",1.2),("ankle",-.2),("base",.4)]}
    return p
def test_original29_work_and_support_proxy():
    d=mechanics(sampled())
    assert d["pre_0p4s"]["observed_real_steps"]==80
    assert d["post_1p25s_full"] is True
    assert d["pre_0p4s"]["two_foot_normal_force_proxy_impulse_Ns"]["left"]==pytest.approx(44.)
    assert d["pre_0p4s"]["segment_estimated_motor_work"]["waist"]["estimated_positive_motor_work_J"]>0
    assert d["qualified_human_sports_kinetics"] is False
    assert not d["native_actor_collider_manifold_certified"]
def test_censored_post_never_claimed_full():
    d=mechanics(sampled(240))
    assert d["right_censored"] and not d["post_1p25s_full"]
    assert not d["full_14s"]
def test_no_fake_torque_data():
    p=sampled()
    del p["traces"][0][50]["joint_torque_estimate"]
    with pytest.raises(KeyError):
        mechanics(p)
def test_no_fake_human_forces():
    p=sampled()
    p["traces"][0][20]["whole_body_com_velocity"]=[float("nan"),0.,0.]
    with pytest.raises(ValueError,match="NONFINITE"):
        mechanics(p)
def test_missing_contribution_is_not_invented():
    p=sampled()
    del p["traces"][0][119]["foot_velocity_contributions"]
    with pytest.raises(ValueError,match="MISSING_SOURCE"):
        mechanics(p)
def test_joint_torque_29_shape_required():
    p=sampled()
    p["traces"][0][20]["joint_torque_estimate"]=[0.]*28
    with pytest.raises(ValueError):
        mechanics(p)
def test_200hz_duplicate_rejected_even_with_power():
    p=sampled()
    p["traces"][0][19]["time_s"]=p["traces"][0][18]["time_s"]
    with pytest.raises(ValueError,match="200HZ"):
        mechanics(p)
def test_no_time_reflection_from_future():
    p=sampled()
    a=mechanics(p)
    q=copy.deepcopy(p)
    for r in q["traces"][0]:
        if r["time_s"]>1.25:r["joint_torque_estimate"]=[120.]*29
    b=mechanics(q)
    assert a["pre_0p4s"]==b["pre_0p4s"]
