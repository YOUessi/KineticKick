"""Frozen v155 CPU-only whole-body 6D 4-corner patch and independent safety tests."""
import copy
import itertools
import numpy as np
import pytest

from v155_patch_wrench_qp import fixture,solve,frozen,corner_mapping
from v155_independent_patch_audit import check
from v155_frozen_patch_campaign import corrupt
from v154_cpu_feasibility_campaign import synthetic_problem
from v154_floating_base_qp import solve as legacy_solve
from v154_independent_wbc_audit import independent_check as legacy_check

FROZEN = frozen()
CASES = list(itertools.product(FROZEN["phase_order"],FROZEN["supports"],FROZEN["seeds"]))
NEG = list(itertools.product(FROZEN["negative_kinds"],FROZEN["supports"],FROZEN["seeds"]))

@pytest.mark.parametrize("phase,stance,seed",CASES)
def test_all_six_axis_support_and_full29_phase_cases(phase,stance,seed):
    p=fixture(phase,stance,seed)
    r=solve(p)
    verdict=check(p,r)
    assert r["solved"] and verdict["pass"],(phase,stance,seed,r["status"],verdict)
    assert verdict["max_6d_stance_residual"]<1e-4
    assert verdict["max_dynamics_residual"]<1e-4
    assert len(r["tau29"])==29
    assert len(r["a35"])==35
    assert np.max(abs(np.asarray(r["a35"])[6:]))>0.01
    assert len(verdict["cop_xy_m"])==len(p["contacts"])
    assert all(cop is not None and abs(cop[0])<=.12+1e-7 and abs(cop[1])<=.045+1e-7
               for cop in verdict["cop_xy_m"])
    assert verdict["actual_physics_validated"] is False
    assert r["real_physics_episode"] is False
    assert r["legal_high_speed_instep_success"] is None

@pytest.mark.parametrize("kind,stance,seed",NEG)
def test_frozen_negative_cases_reject(kind,stance,seed):
    p=corrupt(fixture("swing",stance,seed),kind)
    try:
        out=solve(p)
        assert not check(p,out)["pass"],(kind,stance,seed)
    except (ValueError,KeyError):
        pass

@pytest.mark.parametrize("stance,seed",list(itertools.product(FROZEN["supports"],FROZEN["seeds"])))
def test_old_point_foot_misses_angular_support_counterexample(stance,seed):
    old=synthetic_problem("swing",stance,seed)
    old["h"][3]=2.0
    ro=legacy_solve(old)
    assert legacy_check(old,ro)["pass"]
    assert abs(ro["a35"][3])>1.0  # stance angular acceleration unrestrained!
    p=corrupt(fixture("swing",stance,seed),"out_of_patch_roll_moment")
    rn=solve(p)
    assert not check(p,rn)["pass"]

@pytest.mark.parametrize("fault",[
    "fake_ball_success","fake_manifold","fake_professional_learning",
    "base_ang_acc","corner_normal","corner_friction",
    "false_wrench","joint_torque","joint_position_limit","missing_corner",
    "wrong_stance_count","nonfinite_result"])
def test_independent_auditor_cannot_be_forged(fault):
    p=fixture("swing","double",0)
    out=solve(p)
    assert check(p,out)["pass"]
    bad=copy.deepcopy(out)
    if fault=="fake_ball_success":bad["legal_high_speed_instep_success"]=True
    elif fault=="fake_manifold":bad["native_collision_manifold_certified"]=True
    elif fault=="fake_professional_learning":bad["human_elite_technique_learned"]=True
    elif fault=="base_ang_acc":bad["a35"][3]+=1
    elif fault=="corner_normal":bad["corner_forces3"][0][0][2]=-20
    elif fault=="corner_friction":bad["corner_forces3"][0][0][0]+=50
    elif fault=="false_wrench":bad["wrenches6"][0][3]+=1
    elif fault=="joint_torque":bad["tau29"][0]+=100
    elif fault=="joint_position_limit":bad["a35"][6]+=1000000
    elif fault=="missing_corner":bad["corner_forces3"][0].pop()
    elif fault=="wrong_stance_count":bad["wrenches6"].pop()
    elif fault=="nonfinite_result":bad["a35"][0]=float("nan")
    try:
        assert not check(p,bad)["pass"]
    except (ValueError,KeyError):
        pass

def test_wrench_cross_product_geometry_and_positive_cop():
    p=fixture("plant","left",0)
    z=np.array([[1,2,3],[-2,0,4],[0,1,5],[0,-1,6]],dtype=float)
    xy=np.asarray(p["contacts"][0]["corners_xy"])
    exact=np.concatenate((z.sum(axis=0),np.cross(np.column_stack((xy,np.zeros(4))),z).sum(axis=0)))
    reconstructed=corner_mapping(xy)@z.reshape(12)
    assert np.allclose(exact,reconstructed)
    assert np.allclose([exact[3]/exact[2],-exact[4]/exact[2]],
                       [np.dot(xy[:,1],z[:,2])/z[:,2].sum(),
                        np.dot(xy[:,0],z[:,2])/z[:,2].sum()])

def test_human_qualification_and_true_physics_hard_gate():
    p=fixture("plant","left",0)
    assert p["provenance"]["human_training_qualified"] is False
    assert p["sole_calibrated_for_real"] is False
    p["source_kind"]="runtime"
    with pytest.raises(ValueError,match="RUNTIME_UNCERTIFIED"):
        solve(p)

def test_same_protocol_byte_lock():
    cfg=frozen()
    assert cfg["positive_cases"]==48
    assert len(cfg["negative_kinds"])*cfg["negative_seed_support_denominator"]==96
    assert cfg["gpu_allowed"] is False
    assert cfg["original_joint_count"]==29
