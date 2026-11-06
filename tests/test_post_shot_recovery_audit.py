"""CPU checks that shot preservation and absent residual evidence cannot hide failure."""
import copy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from audit_landing_release import compare_primary_strike
from audit_post_shot_recovery import physical_result, residual_evidence


def physical_report():
    rows = [dict(time_s=i*.005, physics_dt_s=.005, contact_force_N=10. if i == 4 else 0.,
                 foot='right', ball_velocity=[4. if i >= 5 else 0., 0., 0.],
                 right_foot_point_velocity=[5., 0., 0.], joint_vel=[0.]*29,
                 joint_torque_estimate=[0.]*29, joint_velocity_target=[0.]*29,
                 joint_target=[10.]*29, terminated=False)
            for i in range(1, 2801)]
    return dict(traces=[rows], joint_names=[str(i) for i in range(29)],
                joint_velocity_limits=[[20.]*29], joint_effort_limits=[[100.]*29],
                records=[dict(contact_time_s=.020, success=True, surface_match=True,
                              goal=True, recovered=True, failed=False)],
                running_evidence={'running_success': True},
                stride_shot_evidence={'continuous_running_contact': True},
                sequence_records=[dict(illegal_body_ball_contact=False,
                    recovery=dict(passed=True, final_standing_seconds=8.,
                                  terminated_or_fallen=False))],
                root_fixed=False, post_initialization_state_writes=False,
                stride_shot_success=True, termination_reasons=[[]])


def test_identical_shot_cannot_override_actual_overspeed_or_full_task_failure():
    reference = physical_report()
    candidate = copy.deepcopy(reference)
    # PD targets above soft positions are not an observed joint-speed violation.
    assert physical_result(candidate)['passed']
    candidate['traces'][0][449]['joint_vel'][3] = -21.143692
    shot = compare_primary_strike(reference['traces'][0], candidate['traces'][0], [1., 0., 0.])
    assert shot['equal']
    observed = physical_result(candidate)
    assert not observed['passed']
    assert observed['actual_joint_speed']['actual_speed_exceedance_sample_joint_count'] == 1
    assert observed['actual_joint_speed']['maximum_actual_speed_ratio'] > 1.
    # Independently fail the original task even after removing the overspeed.
    candidate['traces'][0][449]['joint_vel'][3] = 0.
    candidate['stride_shot_success'] = False
    candidate['records'][0]['recovered'] = False
    observed = physical_result(candidate)
    assert not observed['passed']
    assert observed['checks']['actual_joint_speed']
    assert not observed['checks']['original_task']
    assert compare_primary_strike(reference['traces'][0], candidate['traces'][0], [1., 0., 0.])['equal']


def test_applied_residual_boundary_is_strict_and_missing_debug_is_not_zero():
    report = {'joint_strike': {'config': {'post_shot_recovery':
              {'maximum_rad': .12, 'maximum_step_rad': .02}}, 'feedback': []}}
    previous = np.zeros(29)
    for i, value in enumerate((0., 1e-8, .01)):
        current = np.zeros(29); current[4] = value
        report['joint_strike']['feedback'].append(dict(time_s=i*.02,
            post_shot_recovery=dict(actual_delta_rad=current.tolist(),
                previous_delta_rad=previous.tolist(), baseline_action=[0.]*29,
                final_action=current.tolist(), ball_state_write=False)))
        previous = current
    result = residual_evidence(report, .06)
    assert result['passed']
    assert result['first_nonzero_applied_decision_s'] == .02
    report['joint_strike']['feedback'][2]['post_shot_recovery']['actual_delta_rad'][4] = .05
    assert not residual_evidence(report, .06)['passed']
    del report['joint_strike']['feedback'][1]['post_shot_recovery']
    assert residual_evidence(report, .06)['status'] == 'unavailable'
    assert not residual_evidence({'joint_strike': {'feedback': []}}, .06).get('passed', False)
