"""v154 original 29DoF floating-base actuation-QP: 48 deterministic cases."""
import copy
import itertools
import numpy as np
import pytest
from v154_cpu_feasibility_campaign import synthetic_problem, PHASES, SUPPORT, SEEDS
from v154_floating_base_qp import solve,frozen,validate_problem
from v154_independent_wbc_audit import independent_check

@pytest.mark.parametrize("phase,stance,seed",list(itertools.product(PHASES,SUPPORT,SEEDS)))
def test_coupled_floatbase_29dof_fixture(phase,stance,seed):
    p=synthetic_problem(phase,stance,seed)
    r=solve(p)
    a=independent_check(p,r)
    assert a["pass"], (phase,stance,seed,r["status"],a)
    assert r["real_physics_episode"] is False
    assert r["legal_high_speed_instep_success"] is None
    assert not r["human_elite_technique_learned"]
    assert len(r["tau29"])==29 and len(r["a35"])==35
    assert all(a["per_group_joint_acceleration_abs_sum"][key]>0.05
               for key in ("legs","waist","arms"))
    assert a["max_joint_contact_clearance_violation"]<=2e-4
    assert a["max_dynamics_residual"]<=2e-4
    assert len(r["stance_forces3"])==len(p["contacts"])

def test_source_qualification_cannot_be_upgraded_to_real_by_label():
    p=synthetic_problem("swing","left",0)
    p["source_kind"]="runtime"
    with pytest.raises(ValueError,match="PROVENANCE"):
        validate_problem(p)

def test_real_provenance_needs_sha_digest():
    p=synthetic_problem("swing","left",0)
    p["source_kind"]="runtime"
    p["provenance"]={key:True for key in (
        "measured_signed_ground_normal","calibrated_mass_and_bias",
        "calibrated_stance_jacobians","real_collider_gap_gradients",
        "exact_native_step_id")}
    with pytest.raises(ValueError,match="SOURCE_HASHES"):
        validate_problem(p)

@pytest.mark.parametrize("tamper",("torque","stance","acceleration","impulse",
                                   "provenance","fake_success","kinematics"))
def test_independent_rejects_tampering(tamper):
    p=synthetic_problem("swing","double",1)
    r=solve(p)
    assert independent_check(p,r)["pass"]
    bad=copy.deepcopy(r)
    if tamper=="torque": bad["tau29"][0]+=20
    elif tamper=="stance":bad["a35"][0]+=10
    elif tamper=="acceleration":bad["a35"][9]+=20
    elif tamper=="impulse":bad["stance_forces3"][0][0]+=50
    elif tamper=="provenance":bad["real_physics_episode"]=True
    elif tamper=="fake_success":bad["legal_high_speed_instep_success"]=True
    elif tamper=="kinematics":bad["a35"]=bad["a35"][:-1]
    if tamper in ("provenance","fake_success","kinematics"):
        with pytest.raises(ValueError):
            independent_check(p,bad)
    else:
        assert not independent_check(p,bad)["pass"]

def test_frozen_matrix_contract():
    cfg=frozen()
    assert len(cfg["phases"])*len(cfg["fixture_support"])*len(cfg["fixture_seeds"])==48
    assert cfg["version"]=="v0.1.154"
    assert cfg["must_require"]["human_elite_motion_qualified_before_training"] is True
    assert cfg["contact_friction_coefficient_fixture"]>0

def test_wrong_29_joint_order_refused():
    p=synthetic_problem("plant","left",0)
    p["joint_names"][0],p["joint_names"][1]=p["joint_names"][1],p["joint_names"][0]
    with pytest.raises(ValueError,match="ORIGINAL_29"):
        solve(p)

@pytest.mark.parametrize("edit",("nonspd","nonfinite_ref","joint_limit","velocity_state","no_collision",
                                 "wrong_stance","no_ground_evidence","bad_normal","negative_mu"))
def test_invalid_physics_inputs_fail_closed(edit):
    p=synthetic_problem("plant","left",0)
    if edit=="nonspd":p["M"][1][1]=-1
    elif edit=="nonfinite_ref":p["a_ref_joint"][0]=float("inf")
    elif edit=="joint_limit":p["q_lo"][0]=2
    elif edit=="velocity_state":p["dq"][0]=100
    elif edit=="no_collision":p["guards"]=[]
    elif edit=="wrong_stance":p["contacts"][0]["body"]="unknown"
    elif edit=="no_ground_evidence":p["contacts"][0]["ground_only_signed_normal"]=False
    elif edit=="bad_normal":p["contacts"][0]["normal_axis"]="shoe"
    elif edit=="negative_mu":p["contacts"][0]["mu"]=-1
    with pytest.raises((ValueError,KeyError)):
        solve(p)
