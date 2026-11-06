"""CPU evidence checks for independent, potentially aliased PhysX contact buffers."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from ball_contact_observer import contact_evidence, raw_ball_contacts


def buffers():
    f = np.full((8, 1), np.nan); p = np.full((8, 3), np.nan)
    n = p.copy(); d = f.copy()
    for i, force, normal in [(1, 2., [0., 0., 1.]), (2, 3., [1., 0., 0.]), (5, 4., [0., -1., 0.])]:
        f[i] = force; p[i] = [i, 0., 0.]; n[i] = normal; d[i] = .001
    normal = (f, p, n, d, np.array([[2, 0, 1]]), np.array([[1, 999, 5]]))
    ff = np.full((7, 3), np.nan); fp = ff.copy()
    for i, force in [(0, [-.5, 1., 0.]), (4, [1., 0., 0.]), (5, [0., .5, 0.])]:
        ff[i] = force; fp[i] = [0., i, 0.]
    friction = (ff, fp, np.array([[1, 0, 2]]), np.array([[0, 999, 4]]))
    matrix = np.array([[[3., 0., 2.], [0., 0., 0.], [0., -4., 0.]]])
    return normal, friction, matrix


def test_independent_groups_survive_physx_reuse_of_counts_and_starts():
    normal, friction, matrix = buffers()
    class AliasedView:
        def get_contact_data(self, dt): return normal
        def get_friction_data(self, dt):
            normal[-2][:] = friction[-2]; normal[-1][:] = friction[-1]
            return (*friction[:2], normal[-2], normal[-1])
        def get_contact_force_matrix(self, dt): return matrix
    result = raw_ball_contacts(AliasedView(), .005, -1)
    assert result['force_on_ball_valid']
    assert result['normal_counts'] == [[2, 0, 1]]
    assert result['friction_counts'] == [[1, 0, 2]]
    assert [x['buffer_index'] for x in result['normal_contacts']] == [1, 2, 5]
    assert [x['buffer_index'] for x in result['friction_anchors']] == [0, 4, 5]
    np.testing.assert_allclose(result['friction_raw_w_N'], [.5, 1.5, 0.])
    np.testing.assert_allclose(result['force_on_ball_w_N'], [2.5, -5.5, 2.])


def test_per_pair_normal_reconstruction_cannot_hide_cancellation_or_flip_sign():
    normal, friction, matrix = buffers()
    swapped = matrix[:, [2, 1, 0]]
    assert np.array_equal(swapped.sum(axis=1), matrix.sum(axis=1))
    result = contact_evidence(normal, friction, swapped, 1)
    assert not result['normal_reconstruction_verified'] and result['force_on_ball_w_N'] is None
    assert result['normal_contacts'][0]['normal_w'] == [0., 0., 1.]
    result = contact_evidence(normal, friction, -matrix, 1)
    assert not result['force_on_ball_valid']
    np.testing.assert_allclose(result['raw_normal_reconstruction_matrix_w_N'], matrix)


def test_capacity_saturation_is_invalid_and_overlapping_or_bad_ranges_are_rejected():
    normal = (np.ones((2, 1)), np.zeros((2, 3)), np.array([[0., 0., 1.]]*2),
              np.zeros((2, 1)), np.array([[2]]), np.array([[0]]))
    friction = (np.zeros((2, 3)), np.zeros((2, 3)), np.array([[0]]), np.array([[999]]))
    result = contact_evidence(normal, friction, np.array([[[0., 0., 2.]]]), 1)
    assert result['normal_reconstruction_verified'] and result['capacity_exhausted']
    assert result['force_on_ball_w_N'] is None
    for counts, starts in [([[2, 0, 1]], [[1, 999, 2]]), ([[2, 0, 1]], [[7, 999, 5]]),
                           ([[1.5, 0, 1]], [[1, 999, 5]]), ([[float('nan'), 0, 1]], [[1, 999, 5]])]:
        normal, friction, matrix = buffers()
        altered = (*normal[:4], np.asarray(counts), np.asarray(starts))
        with pytest.raises(ValueError): contact_evidence(altered, friction, matrix, 1)


def test_active_nonfinite_data_is_rejected_and_raw_invalid_evidence_is_retained():
    normal, friction, matrix = buffers()
    normal[1][1, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        contact_evidence(normal, friction, matrix, 1)
    class InvalidView:
        def get_contact_data(self, dt): return normal
        def get_friction_data(self, dt): return friction
        def get_contact_force_matrix(self, dt): return matrix
    result = raw_ball_contacts(InvalidView(), .005, 1)
    assert result['status'] == 'invalid' and result['force_on_ball_w_N'] is None
    assert result['raw_buffers']['normal'][1][1][0] == 'NaN'
    normal, friction, matrix = buffers(); matrix[0, 1, 0] = np.inf
    with pytest.raises(ValueError, match='finite'):
        contact_evidence(normal, friction, matrix, 1)
