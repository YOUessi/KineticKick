"""Small CPU evidence checks; no simulator or candidate physics is run."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from audit_ball_impulse import balance_bins, check_pair, check_reaction, compare_original


def impulse_bins():
    mass, gravity = .43, np.array([0., 0., -9.81])
    before = np.array([.1, -.2, .3])
    rows = []
    for i in range(2):
        forces = dict(right_foot=dict(normal=[2., -8., 4.], tangent=[-3., 1., 0.]),
                      left_foot=dict(normal=[0., 0., 0.], tangent=[0., 0., 0.]),
                      other_robot=dict(normal=[0., 0., 0.], tangent=[0., 0., 0.]),
                      ground=dict(normal=[0., 0., mass*9.81], tangent=[0., .1, 0.]))
        force = sum((np.array(p[k]) for p in forces.values() for k in ('normal', 'tangent')), np.zeros(3))
        after = before + .005*(force/mass+gravity)
        rows.append(dict(step_start_s=1.6+i*.005, step_end_s=1.605+i*.005,
                         velocity_before=before.tolist(), velocity_after=after.tolist(), pairs=forces))
        before = after
    return rows, mass, gravity


def test_measured_normal_tangent_ground_and_gravity_close_in_all_coordinates():
    rows, mass, gravity = impulse_bins()
    result = balance_bins(rows, mass, gravity)
    assert result['passed']
    assert result['aggregate']['residual_norm_Ns'] < 1e-14
    np.testing.assert_allclose(result['aggregate']['pair_impulses_Ns']['right_foot']['tangent'], [-.03, .01, 0.])
    np.testing.assert_allclose(result['aggregate']['gravity_impulse_Ns'], [0., 0., -.042183])
    assert result['aggregate']['heading']['normal_to_contact_total_change_deg'] < 0
    # Without measured ground, this must remain an unexplained vector residual.
    for row in rows:
        row['pairs']['ground'] = dict(normal=[0., 0., 0.], tangent=[0., 0., 0.])
    incomplete = balance_bins(rows, mass, gravity)
    assert not incomplete['passed']
    np.testing.assert_allclose(incomplete['aggregate']['pair_impulses_Ns']['right_foot']['tangent'], [-.03, .01, 0.])
    assert incomplete['aggregate']['residual_Ns'][2] > .04


def test_opposite_bin_errors_cannot_be_hidden_by_whole_window_closure():
    rows, mass, gravity = impulse_bins()
    rows[0]['velocity_after'][0] += .03/mass
    rows[1]['velocity_before'][0] += .03/mass
    result = balance_bins(rows, mass, gravity)
    assert result['aggregate']['passed']
    assert not result['passed']
    assert result['failed_bin_end_s'] == [1.605, 1.6099999999999999]
    assert result['maximum_bin_residual_Ns'] == pytest.approx(.03)


@pytest.mark.parametrize('change', ['time', 'continuity', 'nonfinite'])
def test_missing_overlapping_or_nonfinite_bin_state_is_not_balance_evidence(change):
    rows, mass, gravity = impulse_bins()
    if change == 'time':
        rows[1]['step_start_s'] += .005
    elif change == 'continuity':
        rows[1]['velocity_before'][0] += .001
    else:
        rows[0]['pairs']['right_foot']['tangent'][0] = float('nan')
    with pytest.raises(ValueError):
        balance_bins(rows, mass, gravity)


def test_full_original_replay_and_old_failure_are_preserved_exactly():
    rows = [dict(time_s=i*.005, physics_dt_s=.005, contact_force_N=0., ball_velocity=[0., 0., 0.])
            for i in range(1, 2801)]
    original = dict(traces=[rows], contacts=[[]], kinematics=[[]],
        records=[dict(goal=False, success=False)], sequence_records=[dict(recovery=True)],
        joint_strike=dict(feedback=[dict(time_s=0., phase_s=0.)], config={'version': 'frozen'}),
        dynamics_telemetry=dict(runner_source_sha256='old', controller_changed=False))
    observed = copy.deepcopy(original)
    observed['traces'][0][319]['ball_contact_detail'] = {'new': 'observation'}
    observed['dynamics_telemetry'].update(runner_source_sha256='new', ball_contact_detail={'new': 'metadata'})
    assert compare_original(original, observed)['passed']
    observed['records'][0]['goal'] = True
    assert not compare_original(original, observed)['passed']
    observed = copy.deepcopy(original)
    observed['joint_strike']['feedback'][0]['phase_s'] = .02
    assert not compare_original(original, observed)['passed']
    observed = copy.deepcopy(original)
    observed['traces'][0][2799]['ball_velocity'][0] = 1e-15
    assert not compare_original(original, observed)['passed']
    observed = copy.deepcopy(original)
    observed['traces'][0].pop()
    assert not compare_original(original, observed)['passed']


def raw_pair():
    return dict(normal_force_matrix_w_N=[[[0., 0., 20.], [0., 0., 10.]]],
        raw_normal_reconstruction_matrix_w_N=[[[0., 0., 20.], [0., 0., 10.]]],
        normal_force_on_ball_w_N=[0., 0., 30.], friction_raw_w_N=[-2., 0., 0.],
        friction_force_on_ball_w_N=[-2., 0., 0.], force_on_ball_w_N=[-2., 0., 30.],
        friction_raw_matrix_w_N=[[[-2., 0., 0.], [0., 0., 0.]]],
        normal_counts=[[1, 1]], normal_starts=[[0, 1]], normal_buffer_capacity=8,
        friction_counts=[[1, 0]], friction_starts=[[3, 0]], friction_buffer_capacity=8,
        normal_contacts=[dict(pair_index=[0, i], buffer_index=i, force_scalar_N=f,
            normal_w=[0., 0., 1.], point_w=[float(i), 0., 0.], separation_m=-.001)
            for i, f in enumerate((20., 10.))],
        friction_anchors=[dict(pair_index=[0, 0], buffer_index=3, point_w=[.5, 0., 0.],
                               force_raw_w_N=[-2., 0., 0.])],
        friction_sign=1, capacity_exhausted=False, normal_reconstruction_verified=True,
        force_on_ball_valid=True, normal_contact_count=2, friction_anchor_count=1,
        normal_reconstruction_tolerance_N=.001)


def test_independent_anchor_counts_and_indices_are_kept_and_fixed_sign_required():
    result = check_pair(raw_pair(), 1)
    assert result['normal_contact_count'] == 2
    assert result['friction_anchor_count'] == 1
    assert result['tangent'] == [-2., 0., 0.]
    with pytest.raises(ValueError, match='sign'):
        check_pair(raw_pair(), -1)


@pytest.mark.parametrize('change', ['cancelled_matrix', 'capacity', 'buffer_end', 'missing', 'overlap', 'nonfinite'])
def test_raw_coverage_and_per_filter_reconstruction_fail_closed(change):
    pair = raw_pair()
    if change == 'cancelled_matrix':
        # The aggregate is still 30 N; each actor/filter pair is wrong.
        pair['normal_force_matrix_w_N'][0][0][2] += 1.
        pair['normal_force_matrix_w_N'][0][1][2] -= 1.
    elif change == 'capacity':
        pair['normal_buffer_capacity'] = 2
    elif change == 'buffer_end':
        pair['friction_buffer_capacity'] = 4
    elif change == 'missing':
        pair['friction_anchors'] = []
    elif change == 'overlap':
        pair['normal_starts'] = [[0, 0]]
    else:
        pair['friction_anchors'][0]['force_raw_w_N'][0] = float('nan')
    with pytest.raises(ValueError):
        check_pair(pair, 1)


def test_reaction_sign_is_measured_and_zero_detail_capacity_is_not_matrix_truncation():
    measured = dict(normal_force_matrix_w_N=[[[0., 0., -20.]]], normal_force_on_foot_w_N=[0., 0., -20.],
        force_on='right_foot', sensor_paths=['/World/envs/env_0/Robot/right_ankle_roll_link'],
        filter_paths=[['/World/envs/env_0/SoccerBall']], verified=True, tolerance_N=.001,
        max_contact_data_count=0)
    assert check_reaction(measured, [0., 0., 20.], 'right')['opposed_error_N'] == 0.
    with pytest.raises(ValueError, match='opposing'):
        check_reaction(measured, [0., 0., -20.], 'right')


def test_explicit_signed_raw_decode_retains_original_signs_and_rechecks_ranges_and_reaction():
    # The collector rejected the signed scalar before returning its decoded matrix.
    rejected = dict(status='invalid', error='Negative normal-force magnitude',
        force_on_ball_valid=False, force_on_ball_w_N=None, friction_sign=1,
        raw_buffers=dict(normal=[[[-20.], [0.], [0.]], [[0., 0., 0.]]*3,
            [[0., 0., -1.], [0., 0., 0.], [0., 0., 0.]], [[-.001], [0.], [0.]], [[1]], [[0]]],
            friction=[[[-2., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.]]*3, [[1]], [[0]]],
            normal_force_matrix_w_N=[[[0., 0., 20.]]]))
    original = copy.deepcopy(rejected)
    with pytest.raises((ValueError, KeyError)):
        check_pair(rejected, 1)
    decoded = check_pair(rejected, 1, decode_signed_raw=True)
    assert rejected == original
    assert decoded['normal'] == [0., 0., 20.]
    assert decoded['tangent'] == [-2., 0., 0.]
    assert decoded['negative_normal_scalar_count'] == 1
    assert decoded['acquisition_force_on_ball_valid'] is False
    measured = dict(normal_force_matrix_w_N=[[[0., 0., -20.]]], normal_force_on_foot_w_N=[0., 0., -20.],
        force_on='right_foot', sensor_paths=['/World/envs/env_0/Robot/right_ankle_roll_link'],
        filter_paths=[['/World/envs/env_0/SoccerBall']], verified=False, tolerance_N=.001,
        opposed_ball_normal_error_N=None)
    assert check_reaction(measured, decoded['normal'], 'right', decode_signed_raw=True)['opposed_error_N'] == 0.
    measured['normal_force_matrix_w_N'][0][0][2] = 20.
    with pytest.raises(ValueError):
        check_reaction(measured, decoded['normal'], 'right', decode_signed_raw=True)
    for bad in ('wrong_matrix', 'bad_range', 'nan'):
        changed = copy.deepcopy(rejected)
        if bad == 'wrong_matrix': changed['raw_buffers']['normal_force_matrix_w_N'][0][0][2] *= -1
        if bad == 'bad_range': changed['raw_buffers']['normal'][-1] = [[3]]
        if bad == 'nan': changed['raw_buffers']['normal'][0][0][0] = 'NaN'
        with pytest.raises(ValueError):
            check_pair(changed, 1, decode_signed_raw=True)


def test_probe_path_exception_requires_same_real_file_and_verified_digest(tmp_path):
    import hashlib
    data = b'original probe'; (tmp_path/'probe.json').write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    rows = [dict(time_s=i*.005, physics_dt_s=.005, contact_force_N=0.) for i in range(1, 2801)]
    a = dict(traces=[rows], contacts=[[]], kinematics=[[]], dynamics_telemetry=dict(
        probe_path=str(tmp_path/'probe.json'), probe_sha256=digest))
    b = copy.deepcopy(a); b['dynamics_telemetry']['probe_path'] = 'probe.json'
    result = compare_original(a, b, root=tmp_path)
    assert result['passed'] and len(result['metadata_path_exceptions']) == 1
    (tmp_path/'other.json').write_bytes(data)
    b['dynamics_telemetry']['probe_path'] = 'other.json'
    assert not compare_original(a, b, root=tmp_path)['passed']
