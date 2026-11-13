"""V147 pre-registered runtime parity guards and real SHA-locked coupled dynamics."""
import copy
import json
import math
import sys
from pathlib import Path

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v147_exact_off_on_parity as p
import v147_real_wholebody_dynamics as d
import v147_independent_dynamics_recheck as independent
from v146_native_original_bridge import json_sha
from test_v142_native_contact_and_human_qualification import state


def sidecar(mode="off", n=3):
    rows=[state(i) for i in range(1,n+1)]
    return {
        "summary":{
            "mode":mode, "original_29dof_completed_5ms_rows":n,
            "copied_native_callback_packets":0,
            "original_full_state_checksum":json_sha(rows),
            "left_wrist_hip_native_physx_truth":"UNKNOWN",
            "same_native_physics_tick_calibrated":False,
            "off_on_physical_equality_certified":False,
            "motor_execution_authorized":False,
            "errors":[], "readout_incomplete":False,
        },
        "original29_5ms_state_rows":rows, "raw_callback_copies":[],
        "collider_to_rigid_owner":{},
    }


def refresh(x):
    x["summary"]["original_full_state_checksum"]=json_sha(
        x["original29_5ms_state_rows"])
    x["summary"]["original_29dof_completed_5ms_rows"]=len(
        x["original29_5ms_state_rows"])
    x["summary"]["copied_native_callback_packets"]=len(
        x["raw_callback_copies"])


def test_v147_config_frozen_and_no_train_on_test():
    cfg=p.frozen()
    assert cfg["parent_sha"]=="2b0c0cab7f42c15564b0ddd945b5f86ed1d49109"
    assert cfg["no_gpu_with_foreign_process"]
    assert cfg["all_historical_policy_outputs_are_test_only_not_training"]
    assert len(cfg["expected_future_physx_matrix"])==6
    assert cfg["strict_gates"]["instep_fraction_min"]==.95
    assert cfg["strict_gates"]["complete_14s_recovery"]
    assert cfg["parity_tolerance"]==0.


def test_equal_input_off_on_does_not_certify_native():
    res=p.audit(sidecar("off"),sidecar("on"))
    assert res["physics_core_equal"]
    assert res["first_difference"] is None
    assert res["runtime_source_attested"] is False
    assert res["live_noninterference_certified"] is False
    assert res["native_pair_truth"]=="UNKNOWN"
    assert res["strict_physical_success"] is None


@pytest.mark.parametrize("key,index",[
    ("q",7),("dq",17),("motor_torque",28),
    ("left_foot_net_force",0),("right_foot_net_force",2),
    ("nonfoot_net_force",1),("ball_pose",0),("ball_twist",2),
    ("com_world",2),
])
def test_single_real_physical_state_difference_must_fail(key,index):
    a,b=sidecar("off"),sidecar("on")
    b["original29_5ms_state_rows"][1][key][index]+=.000001
    refresh(b)
    r=p.audit(a,b)
    assert not r["physics_core_equal"]
    assert "step_2."+key in r["first_difference"]


@pytest.mark.parametrize("key,nested",[
    ("base_pose","position"),("base_pose","quaternion"),
    ("base_twist","linear"),("base_twist","angular"),
])
def test_nested_floating_base_difference_detected(key,nested):
    a,b=sidecar("off"),sidecar("on")
    b["original29_5ms_state_rows"][0][key][nested][-1]+=.001
    refresh(b)
    r=p.audit(a,b)
    assert not r["physics_core_equal"]
    assert key+"."+nested in r["first_difference"]


def test_missing_rows_rejected_and_different_early_stop_not_success():
    a,b=sidecar("off",3),sidecar("on",2)
    assert not p.audit(a,b)["physics_core_equal"]
    assert p.audit(a,b)["first_difference"]=="episode_length_or_early_termination"
    assert not p.audit(sidecar("off",3),sidecar("on",3),
                       expected_steps=1028)["physics_core_equal"]


@pytest.mark.parametrize("corrupt",["duplicate","gap","negative","bool"])
def test_no_missing_duplicate_negative_or_bool_physics_steps(corrupt):
    x=sidecar("off")
    rows=x["original29_5ms_state_rows"]
    rows[1]["step"]={"duplicate":1,"gap":4,"negative":-1,"bool":True}[corrupt]
    refresh(x)
    with pytest.raises(ValueError,match="STEP|STATE|200HZ|PHYSICS"):
        p.validate_sidecar(x,"off")


@pytest.mark.parametrize("corrupt",["wrongmode","fakepair","fakeclock","fakeparity",
                                     "motorclaim","nosummary","mismatch_count"])
def test_fake_provenance_and_missing_summary_fail_closed(corrupt):
    x=sidecar("on")
    s=x["summary"]
    if corrupt=="wrongmode":s["mode"]="off"
    elif corrupt=="fakepair":s["left_wrist_hip_native_physx_truth"]="CONFIRMED"
    elif corrupt=="fakeclock":s["same_native_physics_tick_calibrated"]=True
    elif corrupt=="fakeparity":s["off_on_physical_equality_certified"]=True
    elif corrupt=="motorclaim":s["motor_execution_authorized"]=True
    elif corrupt=="nosummary":x.pop("summary")
    elif corrupt=="mismatch_count":s["original_29dof_completed_5ms_rows"]=4
    with pytest.raises(ValueError):
        p.validate_sidecar(x,"on")


def test_nonfinite_rejected_even_if_both_equal():
    x=sidecar("off")
    x["original29_5ms_state_rows"][0]["dq"][3]=float("nan")
    # json_sha must reject NaN before it can become a convincing checksum.
    with pytest.raises(ValueError):
        refresh(x)


def test_unreported_callback_count_rejected():
    x=sidecar("on")
    x["raw_callback_copies"]=[{"step_hint":1}]
    with pytest.raises(ValueError,match="COUNTER"):
        p.validate_sidecar(x,"on")


@pytest.mark.parametrize("value",[-1,0,4,True])
def test_orphan_callback_not_attested(value):
    x=sidecar("on")
    x["raw_callback_copies"]=[{"step_hint":value}]
    refresh(x)
    with pytest.raises(ValueError,match="CALLBACK_STEP"):
        p.validate_sidecar(x,"on")


def test_off_with_callback_impossible():
    x=sidecar("off")
    x["raw_callback_copies"]=[{"step_hint":1}]
    refresh(x)
    with pytest.raises(ValueError,match="OFF_HAS"):
        p.validate_sidecar(x,"off")


def test_real_source_sha_audit_and_6628_actual_rows():
    result=d.audit_all()
    assert result["real_historical_200hz_rows_checked"]==6628
    assert {name:c["actual_original29_200hz_rows"]
            for name,c in result["cases"].items()}==p.frozen()["original_v54_expected_rows"]
    assert sum(c["terminal_nonphysical_annotations"]
               for c in result["cases"].values())==1
    assert result["cases"]["cmu_10_05"]["last_real_5ms_state_s"]==5.14
    assert result["cases"]["cmu_10_05"]["native_wrist_hip_collider_manifold"]=="UNKNOWN"
    for c in result["cases"].values():
        assert c["all_window_metrics"]["pre"]["sample_count"]==80
        assert c["all_window_metrics"]["post"]["sample_count"]==250
        assert c["group_joint_count"]=={
            "left_leg":6,"right_leg":6,"waist":3,"left_arm":7,"right_arm":7}
        x=c["all_window_metrics"]["pre"]
        assert math.isclose(sum(x["group_absolute_sim_torque_speed_proxy_integrals_J"].values()),
                            x["sim_torque_abs_joint_power_proxy_integral_J"],
                            abs_tol=1e-7)


def test_independent_second_algorithm_matches_sha_locked_real_source():
    result=d.audit_all()
    independent_result=independent.manual_check(result)
    assert independent_result["independently_reviewed_policy_windows"]==12
    assert independent_result["max_absolute_group_sim_power_integral_recheck_difference_J"]<1e-7
    assert independent_result["native_collider_pair_truth"]=="UNKNOWN"
    assert independent_result["new_physx_episodes"]==0


def test_wrong_group_map_and_bad_matrix_rejected():
    from wholebody_safety_interface_v055 import G1_JOINT_ORDER
    assert sum(map(len,d.grouping(G1_JOINT_ORDER).values()))==29
    with pytest.raises(ValueError):
        d.grouping(list(G1_JOINT_ORDER[:-1])+["unclassified_joint"])
    with pytest.raises(ValueError):
        d._float_matrix([{"joint_vel":[0.]*28}],"joint_vel",29)


def test_independent_audit_rejects_fabricated_metric():
    result=d.audit_all()
    result["cases"]["frozen_v32"]["all_window_metrics"]["pre"][
        "group_absolute_sim_torque_speed_proxy_integrals_J"]["right_leg"]+=1
    with pytest.raises(ValueError,match="CONFLICT"):
        independent.manual_check(result)
