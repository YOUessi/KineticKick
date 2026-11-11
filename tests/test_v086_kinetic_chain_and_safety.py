"""Independent 200Hz evidence and fail-closed wholebody eligibility checks."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"tools"/(name+".py"))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
import sys
sys.path.insert(0,str(ROOT/"tools"))
from v086_wholebody_safety_gate import CollisionWitness,WholebodyState,check_prototype_readiness
a=load("v086_kinetic_chain_audit")
def obs(w=None,support=(120.,0.),qdot=None):
    return WholebodyState(q=[0.]*29,dq=qdot or [0.]*29,
                          supporting_foot_normal_forces_N=support,
                          signed_contact=w,max_joint_speed_rad_s=[3.]*29)
def good_witness():
    return CollisionWitness("/Robot/wrist","/Robot/hip",.2,[0.]*29,True,True,.001)
def test_safe_interface_never_authorizes_motor():
    r=check_prototype_readiness(obs(good_witness()))
    assert r["prototype_ready"] and not r["original_G1_motor_command_authorized"]
    assert not r["physical_safety_certified"]
def test_unknown_native_stream_fails_closed():
    from dataclasses import replace
    r=check_prototype_readiness(obs(replace(good_witness(),backend_contact_stream_positive_calibrated=False)))
    assert not r["prototype_ready"] and "UNCALIBRATED_NATIVE_CONTACT_BACKEND" in r["reject_reasons"]
def test_no_geometry_fails_closed():
    r=check_prototype_readiness(obs())
    assert not r["prototype_ready"] and "MISSING_SHAPE_PAIR_WITNESS" in r["reject_reasons"]
def test_collider_clearance_fails_closed():
    from dataclasses import replace
    r=check_prototype_readiness(obs(replace(good_witness(),signed_distance_m=.01)))
    assert "PREDICTED_WRIST_HIP_CLEARANCE_VIOLATION" in r["reject_reasons"]
def test_support_fails_closed():
    assert "NO_VERIFIED_SUPPORT" in check_prototype_readiness(obs(good_witness(),support=(0.,0.)))["reject_reasons"]
def test_qdot_fails_closed():
    assert "JOINT_QDOT_HARD_MARGIN" in check_prototype_readiness(obs(good_witness(),qdot=[4.]+[0.]*28))["reject_reasons"]
def make_row(t,terminated=False):
    return {"time_s":t,"terminated":terminated,
            "joint_pos":[0.]*29,"joint_vel":[0.]*29,"whole_body_com_xyz":[0.,0.,.8]}
def test_real_step_rejects_synthetic_gap():
    import pytest
    with pytest.raises(ValueError,match="NONREAL_TIME_STEP"):
        a.physical_steps([make_row(.005),make_row(.03)])
def test_only_precise_final_marker_is_removed():
    rows=[make_row(.005),make_row(.01),make_row(.010001,True)]
    assert len(a.physical_steps(rows))==2
def test_genuine_contact_step_not_removed():
    rows=[make_row(.005),make_row(.01),make_row(.015,True)]
    assert len(a.physical_steps(rows))==3
