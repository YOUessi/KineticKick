"""Tests enforce fail-closed 29DoF motor admission, never drive the robot."""
import dataclasses,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import pytest
from v107_motor_admission import Limits29,Witness,admit
def base():
    limit=Limits29([-2.]*29,[2.]*29,[5.]*29,[100.]*29,[30.]*29)
    win=Witness(age_seconds=.001,native_pair_certified=True,
        signed_collision_clearance_m=.04,support_certified=True,
        stance_foot_normal_force_N=100.,stance_foot_slip_mps=.01,
        body_geometry_certified=True,provenance_sha256="a"*64)
    return limit,win
def check(**change):
    lim,wit=base()
    return admit([0.]*29,[0.]*29,change.get("q",[0.]*29),
       change.get("dq",[0.]*29),change.get("tau",[0.]*29),
       lim,change.get("w",wit))
def test_synthetic_all_29_full_positive_has_no_actuation():
    r=check()
    assert r["admitted"] and r["original_joints_checked"]==29
    assert r["motor_command_issued"] is False
def test_position_violated_even_if_velocity_safe():
    p=[0.]*29;p[13]=2.1
    assert "JOINT_POSITION" in check(q=p)["refusal_reasons"]
def test_all_29_velocity_not_single_leg():
    dq=[0.]*29;dq[27]=9.
    assert "JOINT_VELOCITY" in check(dq=dq)["refusal_reasons"]
def test_acceleration_and_torque_fail():
    dq=[0.]*29;dq[2]=1.
    tau=[0.]*29;tau[16]=45.
    r=check(dq=dq,tau=tau)
    assert "JOINT_ACCELERATION" in r["refusal_reasons"]
    assert "JOINT_TORQUE" in r["refusal_reasons"]
@pytest.mark.parametrize("mods,why",[
 (dict(native_pair_certified=False),"UNCERTIFIED_ACTOR_COLLIDER_GEOMETRY"),
 (dict(body_geometry_certified=False),"UNCERTIFIED_ACTOR_COLLIDER_GEOMETRY"),
 (dict(signed_collision_clearance_m=None),"COLLISION_CLEARANCE"),
 (dict(signed_collision_clearance_m=.002),"COLLISION_CLEARANCE"),
 (dict(support_certified=False),"UNCERTIFIED_SUPPORT"),
 (dict(stance_foot_normal_force_N=3.),"INSUFFICIENT_STANCE_FORCE"),
 (dict(stance_foot_slip_mps=.08),"STANCE_SLIP"),
 (dict(age_seconds=.1),"STALE_SENSOR_WITNESS"),
 (dict(provenance_sha256=None),"MISSING_SOURCE_PROVENANCE"),
])
def test_missing_real_physical_evidence_refuses(mods,why):
    _,w=base()
    r=check(w=dataclasses.replace(w,**mods))
    assert not r["admitted"] and why in r["refusal_reasons"]
def test_bad_shape_refused():
    lim,w=base()
    with pytest.raises(ValueError,match="29"):
        admit([0.]*28,[0.]*29,[0.]*29,[0.]*29,[0.]*29,lim,w)
def test_bad_limit_refused():
    lim,w=base()
    lim=dataclasses.replace(lim,max_qdot=[0.]*29)
    with pytest.raises(ValueError,match="LIMITS"):
        admit([0.]*29,[0.]*29,[0.]*29,[0.]*29,[0.]*29,lim,w)
