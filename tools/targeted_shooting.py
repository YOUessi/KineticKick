"""Fixed-goal aim and outgoing-speed evidence for physical shooting.

This module is safe to import before Isaac Sim starts: it uses the standard
library only, so it cannot pre-load an incompatible numerical runtime.
"""
from __future__ import annotations

import math
import copy

POLICY_SHA256 = '489711f8654faf8aba49584a39266a33a9068a8a6bc8122e88f2bc7f06739ab1'
ASSET_SHA256 = '34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761'
STYLES = ('inside', 'instep', 'outside')
POWERS = ('low', 'medium', 'high')
LEG_JOINTS = {'right_' + name + '_joint' for name in
              ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length and all(_finite(x) for x in value)


def _window(value):
    if _vector(value, 3):
        value = [value[0], value[1], value[1], value[2]]
    return (_vector(value, 4) and 0 <= value[0] < value[1] <= value[2] < value[3] < 410)


def validate_target_config(config):
    """Reject incompatible or untested command envelopes before launching Isaac."""
    try:
        if (config['schema_version'] != 1 or config['version'] != '0.1.2'
                or config['policy_sha256'] != POLICY_SHA256 or config['asset_sha256'] != ASSET_SHA256
                or config['ball'] != [.14, -.93] or config['goal'] != [0, -3.7]
                or config['goal_forward'] != [0, -1] or config['trial_steps'] != 410
                or config['interpolation'] is not False):
            raise ValueError('Calibration metadata, model or fixed geometry is incompatible')
        points, seeds = config['target_x_m'], config['acceptance_seeds']
        if (not _vector(points, 3) or any(abs(x) > .69 for x in points)
                or any(b-a < .25 for a, b in zip(points, points[1:]))
                or len(seeds) != 2 or len(set(seeds)) != 2 or any(type(s) is not int for s in seeds)
                or set(config['styles']) != set(STYLES)):
            raise ValueError('Expected three separated targets, three styles and two unique seeds')
        for style in STYLES:
            data = config['styles'][style]
            speeds = [data['speeds_mps'][p] for p in POWERS]
            if not _vector(speeds, 3) or min(speeds) <= 0 or any(b-a < .30-1e-8 for a, b in zip(speeds, speeds[1:])):
                raise ValueError('Three positive, separated physical speed requests are required')
            nodes = data['nodes']
            if len(nodes) != 9 or {(n['power'], n['target_x_m']) for n in nodes} != {(p, x) for p in POWERS for x in points}:
                raise ValueError('Each target/power command must occur exactly once')
            for node in nodes:
                control = node['control']
                if (set(control) - {'policy_goal', 'strike', 'cartesian'} or not _vector(control['policy_goal'], 2)
                        or math.dist(control['policy_goal'], config['ball']) < .5):
                    raise ValueError('Invalid finite virtual aim or unknown control fields')
                strike = control.get('strike', {})
                if set(strike) - {'window', 'offsets', 'targets'} or not _window(strike.get('window', [80, 110, 150])):
                    raise ValueError('Invalid strike window')
                for kind in ('offsets', 'targets'):
                    if any(name not in LEG_JOINTS or not _finite(v) for name, v in strike.get(kind, {}).items()):
                        raise ValueError('Invalid strike joint or nonfinite target')
                if 'cartesian' in control:
                    servo = control['cartesian']
                    if (not _window(servo['window']) or not _vector(servo['normal_local'], 3)
                            or math.hypot(*servo['normal_local']) < .5 or not _vector(servo['point_local'], 3)
                            or not _vector(servo['orientation_xyzw'], 4)
                            or abs(math.hypot(*servo['orientation_xyzw'])-1) > .01
                            or not _vector(servo['seed'], 6)
                            or not all(_finite(servo[k]) for k in ('foot_speed_mps', 'contact_frame', 'elevation_rad', 'aim_x', 'max_joint_correction_rad'))
                            or not .1 <= servo['foot_speed_mps'] <= 5
                            or not 0 < servo['max_joint_correction_rad'] <= .3
                            or not 80 <= servo['contact_frame'] <= 150):
                        raise ValueError('Invalid or unbounded Cartesian strike control')
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError('Incomplete or malformed targeted-shooting calibration') from error


def resolve_targeted_shot(config, style, target_x, *, speed_mps=None, power=None):
    """Resolve one calibrated discrete command; never extrapolate or clamp."""
    validate_target_config(config)
    if style not in STYLES:
        raise ValueError(f'Unsupported style; choose from {STYLES}')
    if (speed_mps is None) == (power is None):
        raise ValueError('Specify exactly one of power or speed_mps')
    if not _finite(target_x) or speed_mps is not None and not _finite(speed_mps):
        raise ValueError('Target and speed must be finite numbers')
    points = config['target_x_m']
    point = next((x for x in points if abs(target_x-x) <= 1e-8), None)
    if point is None:
        raise ValueError(f'Target x is not supported; calibrated points are {points} m')
    speeds = config['styles'][style]['speeds_mps']
    if speed_mps is not None:
        power = next((p for p in POWERS if abs(speed_mps-speeds[p]) <= 1e-8), None)
    if power not in POWERS:
        raise ValueError(f'Speed/power is not supported for {style}; choose {speeds}')
    node = next(n for n in config['styles'][style]['nodes'] if n['power'] == power and n['target_x_m'] == point)
    key = f'{style}_{power}_x{points.index(point)}'
    return dict(name=key, controller_key=key, style=style, power=power,
                ball=copy.deepcopy(config['ball']), goal=copy.deepcopy(config['goal']),
                goal_forward=copy.deepcopy(config['goal_forward']),
                target=dict(target_x=point, speed_mps=speeds[power]), **copy.deepcopy(node['control']))


class TargetShotTracker:
    def __init__(self, *, target_x, speed_mps, goal_y=-3.7):
        self.target_x, self.speed_mps, self.goal_y = map(float, (target_x, speed_mps, goal_y))
        if not all(math.isfinite(x) for x in (self.target_x, self.speed_mps, self.goal_y)) or self.speed_mps <= 0:
            raise ValueError('Target and goal must be finite and requested speed must be positive')
        self.previous = None
        self.last_time = None
        self.contact_time = None
        self.crossing = None
        self.outgoing_speed = None
        self.failed = False

    def update(self, sample):
        now = float(sample['time_s'])
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('Trial samples must have strictly increasing times')
        self.last_time = now
        position, velocity = sample['ball_xyz'], sample['ball_velocity']
        force = float(sample['contact_force_N'])
        valid = (len(position) == len(velocity) == 3
                 and all(math.isfinite(x) for x in (*position, *velocity, now, force)))
        self.failed |= not valid or bool(sample['terminated'])
        if self.failed:
            return
        if force > 2 and self.contact_time is None:
            self.contact_time = now
        if self.contact_time is not None and self.previous is not None:
            before = self.previous
            measure_time = self.contact_time + .10
            if self.outgoing_speed is None and before['time_s'] <= measure_time <= now + 1e-10:
                fraction = (measure_time - before['time_s']) / (now - before['time_s'])
                v = [a + (b-a)*fraction for a, b in zip(before['ball_velocity'], velocity)]
                self.outgoing_speed = math.hypot(v[0], v[1])
            a, b = before['ball_xyz'], position
            if self.crossing is None and a[1] > self.goal_y >= b[1]:
                fraction = (self.goal_y - a[1]) / (b[1] - a[1])
                self.crossing = dict(time_s=before['time_s'] + (now-before['time_s'])*fraction,
                    position=[x + (y-x)*fraction for x, y in zip(a, b)],
                    velocity=[x + (y-x)*fraction for x, y in zip(before['ball_velocity'], velocity)])
        self.previous = sample

    def summary(self):
        aim_error = abs(self.crossing['position'][0] - self.target_x) if self.crossing else None
        speed_error = abs(self.outgoing_speed - self.speed_mps) if self.outgoing_speed is not None else None
        target_hit = aim_error is not None and aim_error <= .15 + 1e-10
        speed_match = speed_error is not None and speed_error <= max(.20, .12*self.speed_mps) + 1e-10
        return dict(target_x_m=self.target_x, requested_speed_mps=self.speed_mps,
                    crossing=self.crossing, aim_error_m=aim_error,
                    outgoing_speed_mps=self.outgoing_speed, speed_error_mps=speed_error,
                    target_hit=target_hit, speed_match=speed_match,
                    target_success=bool(target_hit and speed_match and not self.failed))
