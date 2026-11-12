"""Frozen synthetic tests: 29DoF biomechanics, censoring, effort and paired real-physics alignment."""
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import v113_phase_biomechanics as m
from test_v113_coupled_dynamics import NAMES

def episode(n=2800,amp=1.0,torque=2.0):
    rows=[]
    for i in range(n):
        t=(i+1)*.005
        rows.append({
            'time_s':t,'physics_dt_s':.005,
            'joint_vel':[float(amp)]*29,'joint_pos':[.01]*29,
            'joint_torque_estimate':[float(torque)]*29,
            'whole_body_com_xyz':[t,0.,.72],
            'root_xyz':[t*.5,0.,.74],
            'support_foot_force_N':{'left':100.,'right':80.},
            'nonfoot_contact_N':0.})
    return {'root_fixed':False,'joint_names':list(NAMES),
            'joint_velocity_limits':[32.]*29,'records':[{'contact_time_s':1.6}],
            'traces':[rows]}

def test_all29_simulated_motor_work_and_support_integrals():
    a=m.load_physical_arrays(episode())
    out=m.phase_metrics(a)
    p=out['recorded_phase_metrics']['final_pre_400ms']
    assert p['observed_rows']==80
    assert p['support_left_impulse_proxy_Ns']==pytest.approx(40.)
    assert p['support_right_impulse_proxy_Ns']==pytest.approx(32.)
    assert p['all29_qdot_peak_ratio']==pytest.approx(1/32)
    work=sum(x['estimated_positive_motor_work_J'] for x in p['groups'].values())
    assert work==pytest.approx(29*2*.4)
    assert out['motor_work_is_simulated_estimated_torque']
    assert not out['human_grf_cop_or_true_impact_power_certified']
    assert not out['native_actor_collider_manifold_certified']

def test_negative_signed_motor_power_not_clipped_to_positive():
    out=m.phase_metrics(m.load_physical_arrays(episode(torque=-2)))
    p=out['recorded_phase_metrics']['final_pre_400ms']
    assert sum(v['estimated_positive_motor_work_J'] for v in p['groups'].values())==0
    assert sum(v['estimated_negative_motor_work_J'] for v in p['groups'].values())==pytest.approx(-29*2*.4)

def test_early_censored_without_fabricating_recovery():
    out=m.phase_metrics(m.load_physical_arrays(episode(n=360)))
    assert not out['actual_14s_recovery_coverage']
    assert out['recorded_phase_metrics']['post_1250ms']['right_censored']
    assert not out['recorded_phase_metrics']['later_recovery']['observed']

def test_full_14_seconds_observed_and_phase_coverage():
    out=m.phase_metrics(m.load_physical_arrays(episode()))
    assert out['actual_14s_recovery_coverage']
    assert not out['recorded_phase_metrics']['post_1250ms']['right_censored']
    assert not out['recorded_phase_metrics']['later_recovery']['right_censored']

def test_refuse_false_joint_order_and_nonfinite_raw():
    d=episode()
    d['joint_names'][0]=d['joint_names'][1]
    with pytest.raises(ValueError,match='29DOF'):
        m.load_physical_arrays(d)
    d=episode(n=360);d['traces'][0][20]['joint_torque_estimate'][0]=float('nan')
    with pytest.raises(ValueError,match='INVALID_200HZ_JOINT_TORQUE'):
        m.load_physical_arrays(d)

def test_invalid_synthetic_support_refused():
    d=episode(n=360);d['traces'][0][50]['support_foot_force_N']['left']=-1.0
    with pytest.raises(ValueError,match='INVALID_SUPPORT_PROXY'):
        m.load_physical_arrays(d)

def test_contact_genuinely_out_of_trace_refused():
    d=episode(n=360);d['records'][0]['contact_time_s']=4.0
    with pytest.raises(ValueError,match='NO_VALID_IMPACT'):
        m.load_physical_arrays(d)

def test_paired_clocks_and_participant_independent_controller_label():
    a=m.load_physical_arrays(episode(amp=1))
    b=m.load_physical_arrays(episode(amp=1.1))
    x=m.paired_controller_effect(a,b)
    assert x['shared_actual_physics_frames']==2800
    assert x['groups']['right_leg']['approach_rms_qdot_delta_radps']==pytest.approx(.1)
    assert x['groups']['waist']['first_measurable_difference_s']==pytest.approx(.005)
    assert x['not_proof_of_human_coordination']

def test_pairwise_stop_at_actual_common_prefix():
    a=m.load_physical_arrays(episode())
    b=m.load_physical_arrays(episode(n=360))
    x=m.paired_controller_effect(a,b)
    assert x['shared_actual_physics_frames']==360
    assert x['post_impact_prefix_censored']

def test_physics_timestamps_must_match():
    a=episode()
    b=episode()
    b['traces'][0][200]['time_s']+=.001
    with pytest.raises(ValueError,match='200HZ'):
        m.load_physical_arrays(b)

def test_write_once_code_and_no_motor_policy():
    protocol=(ROOT/'docs/V113_FROZEN_PROTOCOL.md').read_text()
    assert 'learned motor actuation disabled' in protocol
    assert 'force plates' in protocol
    assert 'write-once' in protocol
