"""Synthetic sign, interval and rigid material-point checks for the post-hoc reader."""
import gzip
import json

import numpy as np
from scipy.spatial.transform import Rotation

from v160_tracking_ball_impulse import BALL, FOOT, approach_velocity, collect_contacts, loaded_intervals, momentum_window, point_geometry


def test_ball_order_and_unqualified_impulses_are_preserved_in_momentum_balance(tmp_path):
    def pair(actors, vector):
        vector = np.asarray(vector)
        n = vector / np.linalg.norm(vector)
        return {'actors': actors, 'shapes': actors, 'points': [[0, 0, 0, *n, *vector, 0]]}

    dt = 0.005
    ticks = [
        [],
        [pair([FOOT, BALL], [-.020, 0., 0.]), pair([BALL, 'ground'], [0., 0., .030])],
        [pair([BALL, FOOT], [0., .005, 0.])],
        [pair([BALL, FOOT], [0., .030, 0.])],
    ]
    path = tmp_path/'native.jsonl.gz'
    with gzip.open(path, 'wt') as stream:
        for tick, pairs in enumerate(ticks, 1):
            stream.write(json.dumps([tick, pairs])+'\n')
    time = np.arange(1, 5)*dt
    loaded, impulses, counts, _ = collect_contacts(path, time, dt)
    assert loaded_intervals(loaded) == [(2, 2), (4, 4)]
    vel = np.array([[0., 0., 0.], [.02, 0, -.02], [.02, .005, -.07], [.02, .035, -.12]])
    report = momentum_window(2, 4, time, vel, impulses, counts, 1., [0., 0., -10.], dt)
    np.testing.assert_allclose(report['reported_normal_impulse_on_ball_world_Ns']['foot'], [.02, .035, 0.])
    np.testing.assert_allclose(report['reported_normal_impulse_on_ball_world_Ns']['ground'], [0., 0., .03])
    np.testing.assert_allclose(report['closure_residual_world_Ns'], 0., atol=1e-14)
    assert report['state_clock']['pre_first']['tick'] == 1
    assert report['state_clock']['post_last']['tick'] == 4


def test_signed_gap_and_pose_difference_track_the_same_material_point():
    rot = Rotation.from_euler('z', 90., degrees=True)
    origin = np.array([1., 2., 3.])
    # Shape0 is the foot; native normal -Y, so the foot points outward in world +Y.
    row = np.array([1., 2.22, 3., 0., -1., 0., 0., -.2, 0., .02])
    local, normal = point_geometry(row, -1, origin, rot, foot_witness=True)
    np.testing.assert_allclose(local, [.2, 0., 0.], atol=1e-13)
    np.testing.assert_allclose(normal, [1., 0., 0.], atol=1e-13)
    angles = Rotation.from_euler('z', [-90., 0., 90.], degrees=True)
    pos = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    np.testing.assert_allclose(approach_velocity(local, pos, angles, 3, .5)['world_mps'], [1.6, .4, 0.])
