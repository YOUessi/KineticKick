#!/usr/bin/env python3
"""Replay the complete frozen aim/power matrix; failed trials stay in the count."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from free_shooting_metrics import ShotTracker
from targeted_shooting import TargetShotTracker, resolve_targeted_shot, validate_target_config, STYLES, POWERS


def validate_manifest(config, runs, config_hash):
    validate_target_config(config)
    expected = {resolve_targeted_shot(config, s, x, power=p)['controller_key']:
                resolve_targeted_shot(config, s, x, power=p)
                for s in STYLES for p in POWERS for x in config['target_x_m']}
    wanted = {(seed, key) for seed in config['acceptance_seeds'] for key in expected}
    seen, entries = set(), []
    for run in runs:
        if (run.get('targeted_config_sha256') != config_hash
                or run.get('policy_sha256') != config['policy_sha256']
                or run.get('asset_sha256') != config['asset_sha256']
                or run.get('root_fixed') is not False or run.get('post_initialization_state_writes') is not False
                or run.get('perturb_reset') is not True or run.get('contact_evidence_version') != 2
                or run.get('goal_width_m') != 1.6 or run.get('goal_height_m') != 1.1):
            raise ValueError('Run metadata differs from the frozen free-root/contact/configuration contract')
        trials = run.get('trials', [])
        if not trials or any(len(run.get(k, [])) != len(trials) for k in ('traces', 'points', 'goals', 'policy_goals')):
            raise ValueError('Incomplete trial arrays')
        roots = run.get('initial_robot_state', {}).get('root_position', [])
        if len(roots) != len(trials):
            raise ValueError('Missing initial root states')
        for i, trial in enumerate(trials):
            key = (run.get('seed'), trial.get('controller_key'))
            if key not in wanted or key in seen:
                raise ValueError('Unexpected or duplicate seed/command in acceptance evidence')
            command = expected[key[1]]
            if (trial != command or run['points'][i] != command['ball']
                    or run['goals'][i] != command['goal'] or run['policy_goals'][i] != command['policy_goal']):
                raise ValueError('Trial control or geometry changed after calibration was frozen')
            seen.add(key)
            entries.append((run, i))
    if seen != wanted:
        raise ValueError(f'Acceptance matrix incomplete: expected {len(wanted)}, received {len(seen)}')
    return entries


def replay_trial(trial, trace, initial_root):
    if not trace or (abs(trace[-1]['time_s'] - 8.2) > 1e-6 and not trace[-1]['terminated']):
        raise ValueError('Trace is incomplete without a recorded terminal failure')
    shot = ShotTracker(initial_root=initial_root, initial_ball=[*trial['ball'], .113],
                       goal=trial['goal'], forward=trial['goal_forward'], style=trial['style'])
    target = TargetShotTracker(**trial['target'], goal_y=trial['goal'][1])
    last_time = 0.
    for sample in trace:
        now = sample['time_s']
        terminal_marker = sample['terminated'] and 0 < now-last_time <= 1.1e-6
        if (not math.isfinite(now) or abs(sample.get('physics_dt_s', 0)-.005) > 1e-10
                or not terminal_marker and abs(now-last_time-.005) > 1e-7):
            raise ValueError('Missing or invalid 5 ms physics samples')
        events = sample.get('contacts')
        if not isinstance(events, list) or any(not math.isfinite(e['force_N']) or e['force_N'] < 0 for e in events):
            raise ValueError('Missing individual contact-point evidence')
        if abs(sum(e['force_N'] for e in events) - sample['contact_force_N']) > 1e-4:
            raise ValueError('Contact force total differs from its individual points')
        shot.update(sample)
        target.update(sample)
        last_time = now
    result = {**shot.summary(), **target.summary()}
    result['targeted_success'] = bool(result['success'] and result['target_success'])
    return result


def aggregate_results(rows):
    styles = {}
    for style in STYLES:
        subset = [r for r in rows if r['style'] == style]
        speed_groups = {}
        for power in POWERS:
            group = [r for r in subset if r['power'] == power]
            values = [r['outgoing_speed_mps'] for r in group if r['outgoing_speed_mps'] is not None]
            speed_groups[power] = dict(passed=sum(r['targeted_success'] for r in group), total=len(group),
                                      mean_speed_mps=mean(values) if values else None)
        target_groups = []
        for point in sorted({r['target_x_m'] for r in subset}):
            group = [r for r in subset if r['target_x_m'] == point]
            values = [r['crossing']['position'][0] for r in group if r['crossing'] is not None]
            target_groups.append(dict(target_x_m=point, mean_crossing_x_m=mean(values) if values else None))
        speed_values = [speed_groups[p]['mean_speed_mps'] for p in POWERS]
        aim_values = [g['mean_crossing_x_m'] for g in target_groups]
        separated_speed = all(v is not None for v in speed_values) and all(b-a >= .20 for a, b in zip(speed_values, speed_values[1:]))
        separated_aim = len(aim_values) == 3 and all(v is not None for v in aim_values) and all(b-a >= .20 for a, b in zip(aim_values, aim_values[1:]))
        passed = sum(r['targeted_success'] for r in subset)
        styles[style] = dict(passed=passed, total=len(subset), powers=speed_groups, targets=target_groups,
                             strength_separated=separated_speed, direction_separated=separated_aim,
                             accepted=bool(len(subset) == 18 and passed >= 15 and separated_speed and separated_aim
                                 and all(g['total'] == 6 and g['passed'] >= 5 for g in speed_groups.values())))
    return dict(passed=all(v['accepted'] for v in styles.values()),
                success_count=sum(r['targeted_success'] for r in rows), total=len(rows), styles=styles)


def evaluate(config, runs, config_hash):
    entries = validate_manifest(config, runs, config_hash)
    rows = []
    for run, i in entries:
        trial = run['trials'][i]
        result = replay_trial(trial, run['traces'][i], run['initial_robot_state']['root_position'][i])
        result.update(seed=run['seed'], controller_key=trial['controller_key'], style=trial['style'], power=trial['power'])
        rows.append(result)
    return {**aggregate_results(rows), 'records': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--config', default='configs/targeted_shooting_v1.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.config)
    config_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        result = evaluate(json.loads(path.read_text()), [json.loads(Path(p).read_text()) for p in args.runs], config_hash)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    result.update(config_sha256=config_hash,
                  inputs={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.runs})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'inputs')}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
