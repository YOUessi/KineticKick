"""Shared provenance/initialization checks for execution resumes and offline analysis."""
import math


def verify_case_evidence(result, case, matrix_digest, source_hashes, expected_plan, center=None):
    evidence = result['strike_baseline']
    if evidence['matrix_sha256'] != matrix_digest or evidence['case'] != case:
        raise ValueError('Report belongs to a different frozen case')
    if any(evidence.get(key) != value for key, value in source_hashes.items()):
        raise ValueError('Evaluator or protocol source differs from the recorded run')
    if evidence['resolved_plan'] != expected_plan:
        raise ValueError('Resolved controller plan differs beyond permitted ball offset')
    if result['points'] != [expected_plan['ball_xy']] or result['goals'] != [expected_plan['goal_xy']]:
        raise ValueError('Physical ball/goal initialization differs from the case')
    velocity = evidence['initial_reset_observed']['root_linear_velocity_w']
    if (len(velocity) != 1 or len(velocity[0]) != 3
            or not all(math.isfinite(v) for v in velocity[0])):
        raise ValueError('Finite measured initial root velocity required')
    if center is None:
        if case['id'] != 'center' or case['root_velocity_y_delta_mps'] != 0. or any(case['ball_offset_xy_m']):
            raise ValueError('An unperturbed center must establish the reset reference first')
        return
    base = center['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0]
    delta = [0., case['root_velocity_y_delta_mps'], 0.]
    if any(abs(v-b-d) > 1e-6 for v,b,d in zip(velocity[0],base,delta)):
        raise ValueError('Measured initial root velocity differs from the requested offset')
    if result['initial_robot_state'] != center['initial_robot_state']:
        raise ValueError('Initial robot pose/joints changed beyond the requested velocity offset')
    for key in ('asset_sha256','policy_sha256','running_checkpoint_sha256','controller_sources_sha256',
                'motion_sources_sha256','sequence_sources_sha256','config_sha256'):
        if result.get(key) != center.get(key):
            raise ValueError('Immutable runtime fingerprint differs: '+key)
