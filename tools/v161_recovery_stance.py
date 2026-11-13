"""Recover on the measured, loaded stance after the complete source motion.

The captured motor targets retain the standing preload of the original drives.
The ankle feedback adds joint position targets in the local pitch and roll axes.
"""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation


def smootherstep(time_fraction):
    value = float(np.clip(time_fraction, 0., 1.))
    return value ** 3 * (10. - 15. * value + 6. * value ** 2)


def ankle_adjustment(ctx, feet, capture_reference, cfg):
    foot_pos = np.asarray(ctx['body_pos'])[feet]
    foot_quat = np.asarray(ctx['body_quat'])[feet]
    orientations = Rotation.from_quat(foot_quat[:, [1, 2, 3, 0]]).as_matrix()
    support_mid = foot_pos[:, :2].mean(axis=0)
    com = np.asarray(ctx['com'])
    height = max(.25, com[2] - np.min(foot_pos[:, 2]) + .035)
    capture = com[:2] + np.asarray(ctx['com_velocity'])[:2] * np.sqrt(height / 9.81)
    cop_shift = cfg['capture_gain'] * (capture - support_mid - capture_reference)
    length = np.linalg.norm(cop_shift)
    if length > cfg['max_cop_shift_m']:
        cop_shift *= cfg['max_cop_shift_m'] / length

    load = np.maximum(np.asarray(ctx['foot_load'], dtype=float), 0.)
    if load.sum() < cfg['minimum_total_load_N']:
        return np.zeros(4)
    force = [0., 0., cfg['plant_mass_kg'] * 9.81]
    floor_moment = np.cross(np.r_[cop_shift, 0.], force)
    motor_delta = []
    for fraction, basis in zip(load / load.sum(), orientations):
        # The ankle drive exerts the opposite moment on the foot.
        local_torque = -fraction * (basis.T @ floor_moment)
        motor_delta.extend(local_torque[[1, 0]] / cfg['ankle_stiffness_Nm_per_rad'])
    return np.clip(motor_delta, -cfg['max_ankle_delta_rad'], cfg['max_ankle_delta_rad'])


def make_controller(ref, schema, config):
    names = list(schema['joint_names'])
    bodies = list(schema['body_names'])
    sides = ('left', 'right')
    feet = [bodies.index(f'{side}_ankle_roll_link') for side in sides]
    ankles = [names.index(f'{side}_ankle_{axis}_joint')
              for side in sides for axis in ('pitch', 'roll')]
    history = deque(maxlen=config['stable_steps'])
    final_frame = len(ref['joint_pos']) - 1
    captured = capture_reference = start = None
    ankle_delta = np.zeros(4)

    def command(ctx, target):
        nonlocal captured, capture_reference, start
        now = float(ctx['t'])
        if start is None:
            first = ctx['first_contact_s']
            if first is None or now - first < config['minimum_followthrough_s']:
                return target
            root_quat = np.asarray(ctx['root_quat'])
            upright = Rotation.from_quat(root_quat[[1, 2, 3, 0]]).as_matrix()[2, 2]
            stable = (min(ctx['foot_load']) >= config['minimum_bilateral_load_N']
                      and ctx['root_pos'][2] >= config['minimum_root_height_m']
                      and np.linalg.norm(ctx['root_velocity'][:2]) <= config['entry_speed_mps']
                      and upright >= np.cos(np.deg2rad(config['entry_tilt_deg'])))
            if not stable:
                history.clear()
                return target
            support_mid = np.asarray(ctx['body_pos'])[feet, :2].mean(axis=0)
            history.append((np.array(target), np.asarray(ctx['com'])[:2] - support_mid))
            if ctx['frame'] < final_frame or len(history) < config['stable_steps']:
                return target
            start = now
            captured = np.mean([sample[0] for sample in history], axis=0)
            capture_reference = np.mean([sample[1] for sample in history], axis=0)

        delta = ankle_adjustment(ctx, feet, capture_reference, config)
        max_change = config['ankle_slew_rad_per_s'] * ctx['dt']
        ankle_delta[:] += np.clip(delta - ankle_delta, -max_change, max_change)
        goal = captured.copy()
        goal[ankles] += ankle_delta
        blend = smootherstep((now - start) / config['blend_s'])
        if blend == 0:
            return target
        return (1. - blend) * np.asarray(target) + blend * goal

    return command
