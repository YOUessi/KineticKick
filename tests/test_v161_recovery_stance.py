"""Verify recovery never edits the shooting prefix and uses foot-local torque signs."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from v161_recovery_stance import ankle_adjustment, make_controller


ROOT = Path(__file__).resolve().parents[1]
SIDES = ('left', 'right')
SCHEMA = {
    'joint_names': [f'{s}_ankle_{a}_joint' for s in SIDES for a in ('pitch', 'roll')]
                   + [f'other_{i}' for i in range(25)],
    'body_names': [f'{s}_ankle_roll_link' for s in SIDES],
}


def settings(variant):
    return json.loads((ROOT / 'configs' / f'v161_recovery_{variant}.json').read_text())


def sample():
    return dict(t=4., dt=.02, frame=1, first_contact_s=2., foot_load=np.array([160., 160.]),
                body_pos=np.array([[0., .1, .04], [0., -.1, .04]]),
                body_quat=np.array([[1., 0., 0., 0.]] * 2), com=np.array([0., 0., .76]),
                com_velocity=np.zeros(3), root_quat=np.array([1., 0., 0., 0.]),
                root_pos=np.array([0., 0., .8]), root_velocity=np.zeros(6))


def test_legal_shooting_prefix_and_bilateral_entry_are_unchanged():
    cfg = settings('capture')
    controller = make_controller({'joint_pos': np.zeros((3, 29))}, SCHEMA, cfg)
    target = np.arange(29, dtype=float)
    ctx = sample()
    for i in range(30):
        ctx.update(t=4 + .02 * i, frame=1, first_contact_s=2.)
        assert controller(ctx, target) is target
    ctx.update(t=4.7, frame=2, foot_load=[160., 0.])
    assert controller(ctx, target) is target
    ctx.update(t=5., frame=2, foot_load=[160., 160.], first_contact_s=None)
    for _ in range(20):
        assert controller(ctx, target) is target


def test_loaded_handoff_keeps_the_measured_targets_with_feedback_ablated():
    cfg = settings('hold')
    controller = make_controller({'joint_pos': np.zeros((2, 29))}, SCHEMA, cfg)
    acquired = np.arange(29, dtype=float) / 29.
    ctx = sample()
    for i in range(cfg['stable_steps']):
        ctx['t'] = 4. + i * ctx['dt']
        assert controller(ctx, acquired) is acquired
    ctx['t'] += cfg['blend_s'] + .05
    ctx['com_velocity'] = np.array([1., 1., 0.])
    np.testing.assert_allclose(controller(ctx, np.full(29, -2.)), acquired)


def test_forward_capture_shift_commands_positive_pitch_and_rotates_into_foot_roll():
    ctx = sample()
    ctx['com_velocity'][0] = .02
    cfg = settings('capture')
    correction = ankle_adjustment(ctx, [0, 1], np.zeros(2), cfg)
    assert 0 < correction[0] == correction[2] <= cfg['max_ankle_delta_rad']
    np.testing.assert_allclose(correction[[1, 3]], 0.)
    quat = Rotation.from_euler('z', 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    ctx['body_quat'][:] = quat
    rotated = ankle_adjustment(ctx, [0, 1], np.zeros(2), cfg)
    np.testing.assert_allclose(rotated[[0, 2]], 0., atol=1e-12)
    np.testing.assert_allclose(rotated[[1, 3]], correction[[0, 2]])
    ctx['foot_load'][:] = 0.
    np.testing.assert_allclose(ankle_adjustment(ctx, [0, 1], np.zeros(2), cfg), 0.)
