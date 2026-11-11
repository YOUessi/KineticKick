"""Independent v79 source-pinned 200Hz episode-to-termination tests."""
import numpy as np
from tools import v079_independent_200hz_audit as a
from tools.wholebody_safety_interface_v055 import G1_JOINT_ORDER

def fixture(complete):
    rows=[dict(time_s=i*0.005,joint_vel=[0.1]*29,
               joint_pos=[0.1]*29,nonfoot_contact_N=0,
               whole_body_com_xyz=[0,0,0.9],
               support_foot_force_N={"left":100,"right":0}) for i in range(300)]
    p=dict(root_fixed=False,seed=1201,joint_names=list(G1_JOINT_ORDER),joint_velocity_limits=[2]*29,
       traces=[rows],records=[{"contact_time_s":None,"goal":False,
                               "recovered":False}])
    side=dict(seed=1201,exit_code=0,watchdog=False,
              original_g1_full_physics_certified=complete,
              independent_hard_gates={"hard_gate_pass":complete,"failed_gates":[]},
              certification_reason="NOT_FULL_14S")
    return p,side

def test_no_cuda_markers_do_not_certify_partial():
    p,s=fixture(False)
    res=a.certified_rows(p,s,"No CUDA Isaac")
    assert res["status"]=="UNCERTIFIED_OR_NO_G1_PHYSICS"

def test_partial_real_G1_physx_censored_not_false_no_output():
    p,s=fixture(False)
    res=a.certified_rows(p,s,"Environment device    : cuda:0")
    assert res["status"]=="EARLY_TERMINATED_TRUE_GPU_PHYSICS"
    assert res["full_14s_versus_censored"] is False
    assert res["strict_hard_gate_pass"] is False
    assert res["missing_or_censored_contact_window"]
    assert not res["contact_plus_1p25_observed"]
    assert res["unique_200hz_integration_rows"]==300
    assert "EARLY_CENSORED_14S" in res["authoritative_failed_gates"]

def test_early_real_episode_never_imputes_future_recovery():
    p,s=fixture(False)
    p["records"][0]["contact_time_s"]=0.75
    result=a.certified_rows(p,s,"Environment device    : cuda:0")
    assert not result["contact_plus_1p25_observed"]
    assert result["windows"]=={}
    assert not result["original_record_recovered"]

def test_true_dynamics_requires_original_G1_order():
    p,s=fixture(False)
    p["joint_names"]=["j"]*28
    result=a.certified_rows(p,s,"Environment device    : cuda:0")
    assert result["status"]=="UNCERTIFIED_OR_NO_G1_PHYSICS"

