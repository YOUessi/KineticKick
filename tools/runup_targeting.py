"""Discrete aim/power commands and replayed full-sequence acceptance.

This module contains no calibration table. A frozen command matrix is an input,
not evidence that its commands work. ``evaluate`` replays every required trial;
recorded success flags cannot substitute for physical evidence. Running/flight
claims require a separate measured gait gate; this module certifies the existing
runup_metrics approach, requested leg mode, legal contact and recovery contract.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

from free_shooting_metrics import ShotTracker
from targeted_shooting import TargetShotTracker, POLICY_SHA256, ASSET_SHA256, LEG_JOINTS
from runup_metrics import summarize_trial

POWERS = ('low', 'medium', 'high')
REQUIRED_SOURCES = frozenset('tools/'+name for name in (
    'eval_free_shooting.py', 'runup_sequence.py', 'runup_metrics.py', 'runup_targeting.py',
    'targeted_shooting.py', 'free_shooting_metrics.py', 'shooting_control.py',
    'cartesian_strike.py', 'soccer_onnx_policy.py', 'strike_velocity.py', 'running_metrics.py',
    'show_targeted_runup.py'))
CONTROL_FIELDS = frozenset(('policy_goal', 'strike', 'reference', 'coordination', 'strike_velocity'))
BASE_FIELDS = CONTROL_FIELDS | {'name', 'mode', 'style', 'ball', 'goal', 'goal_forward',
                               'motion_file', 'approach_offset_xy', 'approach_offset_fade_s'}


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _vec(x, n):
    return isinstance(x, (list, tuple)) and len(x) == n and all(_finite(v) for v in x)


def _digest(x):
    return isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def _relative(path):
    return isinstance(path, str) and bool(path) and not Path(path).is_absolute() and '..' not in Path(path).parts


def _window(window, steps):
    return (_vec(window, 3) and 0 <= window[0] < window[1] < window[2] <= steps
            or _vec(window, 4) and 0 <= window[0] < window[1] <= window[2] < window[3] <= steps)


def _joint_map(values, bound, *, recovery=False):
    shoulders = {'left_shoulder_roll_joint', 'right_shoulder_roll_joint'} if recovery else set()
    return (isinstance(values, dict) and set(values) <= LEG_JOINTS | shoulders
            and all(_finite(v) and abs(v) <= (.25 if k in shoulders else bound)
                    for k, v in values.items()))


def _strike(entry, steps, *, reference=False, nested=False):
    if not isinstance(entry, dict):
        raise ValueError('A strike/reference control must be an object')
    allowed = {'window', 'offsets'} if reference else {'window', 'offsets', 'targets', 'continuation'}
    if (set(entry) - allowed or 'window' in entry and not _window(entry['window'], steps)
            or not _joint_map(entry.get('offsets', {}), .5, recovery=nested)
            or not _joint_map(entry.get('targets', {}), 3.2)):
        raise ValueError('Invalid or unbounded strike/reference control')
    if 'continuation' in entry:
        if nested:
            raise ValueError('Only one continuation layer is supported')
        _strike(entry['continuation'], steps, nested=True)


def _controls(control, steps, ball):
    if not isinstance(control, dict) or set(control) - CONTROL_FIELDS:
        raise ValueError('Control overrides may not change physical geometry, reference or mode')
    if control.get('coordination') and control.get('strike_velocity'):
        raise ValueError('Coordination and strike velocity cannot share one trial')
    if 'policy_goal' in control:
        goal = control['policy_goal']
        if not _vec(goal, 2) or math.dist(goal, ball) < .5 or math.dist(goal, ball) > 15.:
            raise ValueError('Virtual policy goal must be finite and bounded')
    for key in ('strike', 'reference'):
        if key in control:
            _strike(control[key], steps, reference=key == 'reference')
    if 'coordination' in control:
        c = control['coordination']
        allowed = {'start_frame', 'times_s', 'prepare', 'strike', 'velocity_feedforward',
                   'minimum_support_N', 'maximum_ball_distance_m', 'velocity_feedforward_gain'}
        if not isinstance(c, dict) or set(c) - allowed:
            raise ValueError('Unknown coordination control')
        times, prep, strike = c.get('times_s'), c.get('prepare'), c.get('strike')
        if (not _finite(c.get('start_frame')) or not 0 <= c['start_frame'] < steps
                or not _vec(times, 3) or not 0 < times[0] < times[1] < times[2] <= 2.
                or not prep or not _joint_map(prep, 3.2) or not _joint_map(strike, 3.2)
                or set(prep) != set(strike)
                or type(c.get('velocity_feedforward', False)) is not bool
                or not _finite(c.get('velocity_feedforward_gain', 1.)) or not 0 <= c.get('velocity_feedforward_gain', 1.) <= 2.
                or not _finite(c.get('minimum_support_N', 80.)) or not 0 < c.get('minimum_support_N', 80.) <= 1000
                or not _finite(c.get('maximum_ball_distance_m', .8)) or not 0 < c.get('maximum_ball_distance_m', .8) <= 1.5):
            raise ValueError('Invalid or unbounded support-gated coordination')


    if 'strike_velocity' in control:
        c = control['strike_velocity']
        if (not isinstance(c, dict) or set(c) - {'window', 'velocities', 'minimum_support_N'}
                or not _vec(c.get('window'), 4)
                or not 0 <= c['window'][0] < c['window'][1] < c['window'][2] < c['window'][3] <= steps
                or not c.get('velocities')
                or not _joint_map(c['velocities'], 12.)
                or not _finite(c.get('minimum_support_N', 80.)) or not 0 < c.get('minimum_support_N', 80.) <= 1000.):
            raise ValueError('Invalid or unbounded joint velocity target control')


def validate_runup_target_config(config):
    """Validate a frozen discrete envelope; this does not certify its success."""
    try:
        if (config['schema_version'] != 1 or not isinstance(config['version'], str)
                or type(config['trial_steps']) is not int or config['trial_steps'] <= 0
                or config['policy_dt_s'] != .02 or config['physics_dt_s'] != .005
                or config['goal'] != [0., -3.7] or config['goal_forward'] != [0., -1.]
                or config['goal_width_m'] != 1.6 or config['goal_height_m'] != 1.1
                or config['interpolation'] is not False
                or config['policy_sha256'] != POLICY_SHA256 or config['asset_sha256'] != ASSET_SHA256):
            raise ValueError('Incompatible fixed physical geometry, policy, asset or timing')
        points, seeds = config['target_x_m'], config['acceptance_seeds']
        if (not _vec(points, 3) or any(abs(x) > .69 for x in points)
                or any(b-a < .25-1e-10 for a, b in zip(points, points[1:]))
                or not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2
                or any(type(s) is not int for s in seeds)):
            raise ValueError('Three separated in-goal targets and two unique seeds required')
        sources = config['source_sha256']
        if (not isinstance(sources, dict) or not REQUIRED_SOURCES <= sources.keys()
                or any(not _relative(p) or not _digest(h) for p, h in sources.items())
                or not _digest(config['upstream_sources_sha256'])):
            raise ValueError('Missing frozen controller/source fingerprints')
        modes = config['modes']
        if not isinstance(modes, dict) or not modes or set(modes) - {'hip', 'knee'}:
            raise ValueError('Unsupported or empty requested motion modes')
        for mode, styles in modes.items():
            if not isinstance(styles, dict) or not styles or set(styles) - {'inside', 'instep', 'outside'}:
                raise ValueError('Unsupported or empty requested contact styles')
            for style, entry in styles.items():
                base = entry['base_trial']
                if (not isinstance(base, dict) or set(base) - BASE_FIELDS
                        or base.get('mode') != mode or base.get('style') != style
                        or not _vec(base['ball'], 2) or math.dist(base['ball'], config['goal']) < 1.5
                        or base.get('goal', config['goal']) != config['goal']
                        or base.get('goal_forward', config['goal_forward']) != config['goal_forward']
                        or not _relative(base['motion_file']) or not base['motion_file'].endswith('.npz')
                        or not _digest(entry['motion_sha256'])):
                    raise ValueError('Invalid frozen per-mode base geometry/reference')
                from runup_sequence import approach_offset
                approach_offset(base, 0.)
                _controls({k: v for k, v in base.items() if k in CONTROL_FIELDS}, config['trial_steps'], base['ball'])
                speed_map = entry['speeds_mps']
                if set(speed_map) != set(POWERS):
                    raise ValueError('Exactly three named power levels required')
                speeds = [speed_map[p] for p in POWERS]
                if (not _vec(speeds, 3) or min(speeds) <= 0
                        or any(b-a < .30-1e-10 for a, b in zip(speeds, speeds[1:]))):
                    raise ValueError('Positive requested speeds must be separated by at least 0.30 m/s')
                nodes = entry['nodes']
                if (len(nodes) != 9 or {(n['power'], n['target_x_m']) for n in nodes}
                        != {(p, x) for p in POWERS for x in points}):
                    raise ValueError('Exactly nine unique target/power nodes required')
                for node in nodes:
                    _controls(node['control'], config['trial_steps'], base['ball'])
                    merged = {k: v for k, v in base.items() if k in CONTROL_FIELDS}
                    merged.update(node['control'])
                    _controls(merged, config['trial_steps'], base['ball'])
                    if 'policy_goal' not in node['control'] and 'policy_goal' not in base:
                        raise ValueError('Each command must freeze its virtual policy goal')
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError('Incomplete or malformed runup targeting configuration') from error


def verify_source_files(config, root):
    """Check actual local bytes, not hashes echoed from the configuration."""
    for name, digest in config['source_sha256'].items():
        if not _relative(name) or not _digest(digest):
            raise ValueError('Invalid source fingerprint')
        if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Source changed after calibration: {name}')


def resolve_runup_targeted_trial(config, mode, style, target_x, *, power=None, speed_mps=None):
    """Resolve only exact frozen commands; preserve the base ball and real goal."""
    validate_runup_target_config(config)
    if (speed_mps is None) == (power is None) or not _finite(target_x):
        raise ValueError('A finite target and exactly one speed/power command are required')
    try:
        entry = config['modes'][mode][style]
    except KeyError as error:
        raise ValueError('The requested motion/contact combination has no frozen commands') from error
    point = next((x for x in config['target_x_m'] if abs(x-target_x) <= 1e-8), None)
    if point is None:
        raise ValueError('The target is outside the discrete calibrated envelope')
    if speed_mps is not None:
        if not _finite(speed_mps):
            raise ValueError('Requested speed must be finite')
        power = next((p for p in POWERS if abs(entry['speeds_mps'][p]-speed_mps) <= 1e-8), None)
    if power not in POWERS:
        raise ValueError('The speed/power is outside the discrete calibrated envelope')
    node = next(n for n in entry['nodes'] if n['power'] == power and n['target_x_m'] == point)
    trial = copy.deepcopy(entry['base_trial'])
    trial.update(copy.deepcopy(node['control']))
    key = f'{mode}_{style}_{power}_x{config["target_x_m"].index(point)}'
    trial.update(name=key, controller_key=key, power=power,
                 goal=copy.deepcopy(config['goal']), goal_forward=copy.deepcopy(config['goal_forward']),
                 target=dict(target_x=point, speed_mps=entry['speeds_mps'][power]))
    return trial


def _commands(config):
    return {t['controller_key']: t for mode, styles in config['modes'].items() for style in styles
            for p in POWERS for x in config['target_x_m']
            for t in [resolve_runup_targeted_trial(config, mode, style, x, power=p)]}


def validate_manifest(config, runs, config_sha):
    """Require every seed/command and exact frozen controls, geometry and hashes."""
    validate_runup_target_config(config)
    if not _digest(config_sha):
        raise ValueError('A frozen config fingerprint is required')
    commands = _commands(config)
    expected = {(seed, key) for seed in config['acceptance_seeds'] for key in commands}
    seen, entries = set(), []
    for run in runs:
        if (run.get('running_diagnostic') is True
                or run.get('runup_targeting_config_sha256') != config_sha
                or run.get('runup_targeting_sources_sha256') != config['source_sha256']
                or run.get('upstream_sources_sha256') != config['upstream_sources_sha256']
                or run.get('policy_sha256') != config['policy_sha256'] or run.get('asset_sha256') != config['asset_sha256']
                or run.get('goal_width_m') != 1.6 or run.get('goal_height_m') != 1.1
                or run.get('root_fixed') is not False or run.get('post_initialization_state_writes') is not False
                or run.get('perturb_reset') is not True or run.get('contact_evidence_version') != 2
                or run.get('sequence_evidence_version') != 1):
            raise ValueError('Run provenance differs from the frozen physical contract')
        trials, files = run.get('trials', []), run.get('motion_files', [])
        roots = run.get('initial_robot_state', {}).get('root_position', [])
        if (not trials or any(len(run.get(k, [])) != len(trials) for k in ('traces', 'points', 'goals', 'policy_goals'))
                or len(roots) != len(trials) or len(files) not in (1, len(trials))):
            raise ValueError('Incomplete per-environment evidence arrays')
        for i, trial in enumerate(trials):
            pair = (run.get('seed'), trial.get('controller_key'))
            if pair not in expected or pair in seen:
                raise ValueError('Unexpected or duplicate acceptance seed/command')
            command = commands[pair[1]]
            if (trial != command or run['points'][i] != command['ball'] or run['goals'][i] != command['goal']
                    or run['policy_goals'][i] != command['policy_goal'] or not _vec(roots[i], 3)):
                raise ValueError('Frozen command controls or geometry changed')
            path = files[0] if len(files) == 1 else files[i]
            reference = Path(command['motion_file']).parts
            if (Path(path).parts[-len(reference):] != reference
                    or run.get('motion_sources_sha256', {}).get(path)
                    != config['modes'][command['mode']][command['style']]['motion_sha256']):
                raise ValueError('The physical reference differs from its frozen content')
            seen.add(pair); entries.append((run, i))
    if seen != expected:
        raise ValueError(f'Incomplete command matrix: expected {len(expected)}, received {len(seen)}')
    return entries


def replay_trial(run, index, *, trial_steps):
    """Recompute shot, target, approach, mode, legality and recovery from 5ms rows."""
    trial, trace = run['trials'][index], run['traces'][index]
    if 'target' not in trial:
        raise ValueError('Untargeted trials cannot certify an aim/power command')
    if (trial.get('goal') != [0., -3.7] or trial.get('goal_forward') != [0., -1.]
            or type(trial_steps) is not int or trial_steps <= 0):
        raise ValueError('Fixed target plane and positive duration required')
    duration = trial_steps*.02
    if (not trace or trace[-1]['time_s'] > duration+1e-6
            or abs(trace[-1]['time_s']-duration) > 1e-6 and not trace[-1]['terminated']):
        raise ValueError('Incomplete trace without a recorded terminal failure')
    initial = run['initial_robot_state']['root_position'][index]
    shot = ShotTracker(initial_root=initial, initial_ball=[*trial['ball'], .113], goal=trial['goal'],
                       forward=trial['goal_forward'], style=trial['style'])
    target = TargetShotTracker(**trial['target'], goal_y=trial['goal'][1])
    last = 0.
    for k, sample in enumerate(trace):
        now = sample['time_s']
        terminal_marker = sample['terminated'] and 0 < now-last <= 1.1e-6
        if (not _finite(now) or abs(sample.get('physics_dt_s', 0)-.005) > 1e-10
                or not terminal_marker and abs(now-last-.005) > 1e-7
                or sample['terminated'] and k != len(trace)-1):
            raise ValueError('Missing or invalid contiguous 5 ms physics samples')
        events = sample.get('contacts')
        if (not isinstance(events, list) or any(not _finite(e['force_N']) or e['force_N'] < 0
                or e.get('foot') not in ('left', 'right')
                or e['force_N'] > 0 and (not _vec(e.get('point_foot'), 3) or not _vec(e.get('normal_foot'), 3)) for e in events)
                or not _finite(sample['contact_force_N'])
                or abs(sum(e['force_N'] for e in events)-sample['contact_force_N']) > 1e-4):
            raise ValueError('Missing or inconsistent individual physical contacts')
        shot.update(sample); target.update(sample); last = now
    shot_record = shot.summary()
    recomputed = dict(run)
    recomputed['records'] = [{} for _ in run['trials']]
    recomputed['records'][index] = shot_record
    sequence = summarize_trial(recomputed, index)
    record = {**shot_record, **target.summary(), 'sequence': sequence}
    record.update(sequence_success=sequence['sequence_success'],
                  joint_success=bool(sequence['sequence_success'] and record['target_success'] and run.get('running_diagnostic') is not True),
                  seed=run.get('seed'), mode=trial['mode'], style=trial['style'], power=trial.get('power'),
                  controller_key=trial.get('controller_key'))
    return record


def aggregate_results(rows, config):
    """All 18 trials per combination plus paired aim and strength separation."""
    validate_runup_target_config(config)
    commands = _commands(config)
    expected = {(s, t['mode'], t['style'], t['power'], t['target']['target_x'])
                for s in config['acceptance_seeds'] for t in commands.values()}
    by_key = {}
    for row in rows:
        key = (row['seed'], row['mode'], row['style'], row['power'], row['target_x_m'])
        if key not in expected or key in by_key:
            raise ValueError('Unexpected or duplicate scored command')
        by_key[key] = row
    if by_key.keys() != expected:
        raise ValueError('Cannot aggregate an incomplete acceptance matrix')
    combinations = {}
    for mode, styles in config['modes'].items():
        for style in styles:
            group = [r for key, r in by_key.items() if key[1:3] == (mode, style)]
            speed_checks, direction_checks, nodes = [], [], []
            for seed in config['acceptance_seeds']:
                for x in config['target_x_m']:
                    values = [by_key[(seed, mode, style, p, x)]['outgoing_speed_mps'] for p in POWERS]
                    passed = (all(_finite(v) for v in values)
                              and all(b-a >= .30-1e-10 for a, b in zip(values, values[1:])))
                    speed_checks.append(dict(seed=seed, target_x_m=x, speeds_mps=values, passed=passed))
                for power in POWERS:
                    crossing = [by_key[(seed, mode, style, power, x)]['crossing'] for x in config['target_x_m']]
                    values = [c['position'][0] if c else None for c in crossing]
                    passed = (all(_finite(v) for v in values)
                              and all(b-a >= .25-1e-10 for a, b in zip(values, values[1:])))
                    direction_checks.append(dict(seed=seed, power=power, crossing_x_m=values, passed=passed))
            for power in POWERS:
                for x in config['target_x_m']:
                    values = [by_key[(seed, mode, style, power, x)] for seed in config['acceptance_seeds']]
                    nodes.append(dict(power=power, target_x_m=x, passed=sum(bool(r['joint_success']) for r in values), total=2))
            passed_count = sum(bool(r['joint_success']) for r in group)
            speed_pass = all(c['passed'] for c in speed_checks)
            direction_pass = all(c['passed'] for c in direction_checks)
            combinations[f'{mode}/{style}'] = dict(passed=passed_count, total=len(group), nodes=nodes,
                strength_separated=speed_pass, direction_separated=direction_pass,
                speed_checks=speed_checks, direction_checks=direction_checks,
                accepted=bool(passed_count == 18 and speed_pass and direction_pass))
    return dict(passed=all(c['accepted'] for c in combinations.values()),
                success_count=sum(bool(r['joint_success']) for r in rows), total=len(rows), combinations=combinations)


def evaluate(config, runs, config_sha):
    entries = validate_manifest(config, runs, config_sha)
    rows = [replay_trial(run, i, trial_steps=config['trial_steps']) for run, i in entries]
    return {**aggregate_results(rows, config), 'records': rows}
