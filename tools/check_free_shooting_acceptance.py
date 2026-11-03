#!/usr/bin/env python3
"""Check the entire frozen matrix and replay every claimed success from traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_matrix(config, reports):
    if sorted(r['seed'] for r in reports) != sorted(config['acceptance_seeds']):
        raise ValueError('Reports must contain exactly the declared independent seeds')
    if len({r['config_sha256'] for r in reports}) != 1:
        raise ValueError('Controller configuration changed between seeds')
    expected = {(style, tuple(c['ball']), tuple(c['goal']))
                for style in config['styles'] for c in config['acceptance_conditions']}
    counts = {}
    for report in reports:
        for key in ('policy_sha256', 'asset_sha256'):
            if report[key] != config[key]:
                raise ValueError(f'Unexpected {key}')
        if report['root_fixed'] or report['post_initialization_state_writes'] or not report['perturb_reset']:
            raise ValueError('Invalid physical/reset contract')
        trials, records = report['trials'], report['records']
        actual = {(t['style'], tuple(t['ball']), tuple(t['goal'])) for t in trials}
        if len(trials) != len(expected) or len(records) != len(expected) or actual != expected:
            raise ValueError('Missing or duplicated matrix condition')
        counts[str(report['seed'])] = {style: 0 for style in config['styles']}
        for trial, record in zip(trials, records):
            control = config['styles'][trial['style']]
            if any(trial.get(k, {}) != control.get(k, {}) for k in ('strike', 'reference')):
                raise ValueError('A trial uses a different controller')
            counts[str(report['seed'])][trial['style']] += int(record['success'])
    required = len(config['acceptance_conditions']) * .8
    return dict(passed=all(n >= required for seed in counts.values() for n in seed.values()),
                counts=counts, total_successes=sum(n for seed in counts.values() for n in seed.values()),
                total_trials=sum(len(r['records']) for r in reports))


def main():
    from free_shooting_metrics import ShotTracker
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/free_shooting_v1.json'))
    parser.add_argument('reports', nargs='+')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    reports = [json.loads(Path(p).read_text()) for p in args.reports]
    result = validate_matrix(config, reports)
    for report in reports:
        for i, row in enumerate(report['records']):
            if not row['success']:
                continue
            trial = report['trials'][i]
            if report['points'][i] != trial['ball'] or report['goals'][i] != trial['goal']:
                raise ValueError('Trial manifest differs from the simulated ball/goal positions')
            tracker = ShotTracker(initial_root=report['initial_robot_state']['root_position'][i],
                initial_ball=[*trial['ball'], .113], goal=trial['goal'],
                forward=[g-b for g, b in zip(trial['goal'], trial['ball'])], style=trial['style'])
            trace = report['traces'][i]
            if abs(trace[-1]['time_s'] - config['trial_steps']*config['policy_dt_s']) > .001:
                raise ValueError('Successful trace is shorter than the frozen trial duration')
            for sample in trace:
                tracker.update(sample)
            if not tracker.summary()['success']:
                raise ValueError('Claimed success fails independent trace replay')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
