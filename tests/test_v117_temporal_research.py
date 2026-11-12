from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v117_causal_safety_statistics import true_joint_hazard, measured_support_and_work
from v117_witnessed_horizon_admission import (
    TemporalPlan,PHASES,validate_phase_schedule,examine_plan)
from v107_motor_admission import Limits29,Witness


def synthetic_physics():
    t=np.arange(0.005,2.005,.005)
    vel=np.zeros((len(t),29))
    vel[100,0]=2.5
    return dict(t=t,vel=vel,lim=np.ones(29),impact=1.6,
        names=[f"joint_{i}" for i in range(29)],
        groups=dict(waist=[0],left_arm=[1,2],right_leg=list(range(3,29))),
        tau=np.ones((len(t),29)),support=np.tile(np.array([[100.,50.]]),(len(t),1)))


def test_hazard_first_event_is_from_actual_velocity():
    a=synthetic_physics()
    result=true_joint_hazard(a)
    assert result["first_original_joint_speed_violation_s"]==pytest.approx(a["t"][100])
    assert result["number_exceeding_joints"]==1
    assert result["original29_joint_events"][0]["joint"]=="joint_0"
    assert result["original29_joint_events"][0]["actual_time_above_limit_s"]==.005
    assert result["postcontact_1p25s_censored"]


def test_hazard_never_creates_future_rows():
    a=synthetic_physics()
    a["vel"][:]=0
    result=true_joint_hazard(a)
    assert result["first_original_joint_speed_violation_s"] is None
    assert not result["complete_14s_recovery_observation"]
    assert result["stage_hazards"]["recovery_observed"]["real_200hz_rows"]==0


def test_support_positive_work_is_physics_proxy_only():
    a=synthetic_physics()
    k=measured_support_and_work(a)
    assert k["pre400_left_support_force_proxy_impulse_Ns"]>k["pre400_right_support_force_proxy_impulse_Ns"]
    assert not k["pre400_human_muscle_power_inferred"]
    assert not k["forceplate_grf_cop_calibrated"]
    assert k["pre400_positive_estimated_motor_work_by_group_J"]["waist"]>=0


def limits():
    return Limits29([-2]*29,[2]*29,[5]*29,[500]*29,[100]*29)


def unknown_witness():
    return Witness(age_seconds=0,native_pair_certified=False,
        signed_collision_clearance_m=None,support_certified=False,
        stance_foot_normal_force_N=None,stance_foot_slip_mps=None,
        body_geometry_certified=False,provenance_sha256=None)


def plan(source_kind="UNQUALIFIED_CANDIDATE"):
    n=len(PHASES)
    return TemporalPlan(q_targets=np.zeros((n,29)),
       dq_targets=np.zeros((n,29)),torque_targets=np.zeros((n,29)),
       phases=PHASES,source_kind=source_kind)


def test_all_motor_actions_rejected_without_native_witness():
    p=plan()
    result=examine_plan(p,initial_q=np.zeros(29),initial_dq=np.zeros(29),
        limits=limits(),witnesses=[unknown_witness()]*len(PHASES))
    assert not result["nominal_safety_screen_pass"]
    assert result["frames_with_any_refusal"]==len(PHASES)
    assert "UNCERTIFIED_ACTOR_COLLIDER_GEOMETRY" in result["refusal_reasons"]
    assert "UNQUALIFIED_HUMAN_ACTION_SOURCE" in result["refusal_reasons"]
    assert not result["physical_motor_command_issued"]


def test_even_safe_vectors_require_human_data_qualification():
    full=Witness(age_seconds=0,native_pair_certified=True,
        signed_collision_clearance_m=.1,support_certified=True,
        stance_foot_normal_force_N=100,stance_foot_slip_mps=0,
        body_geometry_certified=True,provenance_sha256="0"*64)
    result=examine_plan(plan(),initial_q=np.zeros(29),
        initial_dq=np.zeros(29),limits=limits(),witnesses=[full]*len(PHASES))
    assert "UNQUALIFIED_HUMAN_ACTION_SOURCE" in result["refusal_reasons"]
    assert not result["physical_motor_command_issued"]


def test_phase_order_and_full_sequence():
    with pytest.raises(ValueError,match="NON_MONOTONIC"):
        validate_phase_schedule(("backswing","approach"))
    assert validate_phase_schedule(PHASES)["whole_kick_sequence_complete"]
    assert not validate_phase_schedule(("approach","landing_recovery"))["whole_kick_sequence_complete"]


def test_no_hidden_dimensional_or_nan_bypass():
    x=plan()
    bad=TemporalPlan(q_targets=np.zeros((len(PHASES),28)),
        dq_targets=x.dq_targets,torque_targets=x.torque_targets,phases=PHASES)
    with pytest.raises(ValueError,match="INVALID_FINITE"):
        examine_plan(bad,initial_q=np.zeros(29),initial_dq=np.zeros(29),
                     limits=limits(),witnesses=[unknown_witness()]*len(PHASES))


def test_first_frame_hazard_at_zero_is_not_silently_dropped():
    a=synthetic_physics()
    a["t"][0]=0.0
    a["vel"][:]=0.0
    a["vel"][0,0]=1.5
    result=true_joint_hazard(a)
    assert result["first_original_joint_speed_violation_s"] == 0.0
    assert result["number_exceeding_joints"] == 1
