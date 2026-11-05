#!/usr/bin/env python3
"""Run the frozen development cases and two prespecified combinations serially."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HOLDOUTS = [
    dict(id='combined_left_slower', ball_offset_xy_m=[-.025, 0.], root_velocity_y_delta_mps=.10),
    dict(id='combined_right_faster', ball_offset_xy_m=[.025, 0.], root_velocity_y_delta_mps=-.10),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/state_conditioned_strike_v1.json')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--video-cases', nargs='*', default=[])
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix:
        parser.error('Prefix must be a filename component')
    protocol = json.loads((ROOT/'configs/strike_baseline_v1.json').read_text())
    cases = {case['id']: case for case in protocol['cases'] + HOLDOUTS}
    if len(set(args.cases)) != len(args.cases) or any(name not in cases for name in args.cases):
        parser.error('Unique declared case ids required')
    if any(name not in args.cases for name in args.video_cases):
        parser.error('Video cases must be included in the run')
    digest = hashlib.sha256(args.config.read_bytes()).hexdigest()
    from strike_baseline_metrics import summarize_strike
    for name in args.cases:
        if hashlib.sha256(args.config.read_bytes()).hexdigest() != digest:
            raise ValueError('Model changed during the matrix')
        report = ROOT/'reports'/f'{args.prefix}_{name}.json'
        log = report.with_suffix('.log')
        if report.exists() or log.exists():
            raise FileExistsError(report)
        case = cases[name]
        cmd = [sys.executable, str(ROOT/'tools/run_placement_strike.py'), '--config', str(args.config),
               '--output', str(report), '--ball-offset', *map(str, case['ball_offset_xy_m']),
               '--velocity-delta', str(case['root_velocity_y_delta_mps'])]
        if name in args.video_cases:
            cmd += ['--video-dir', str(ROOT/'experiments/closed_loop_v026'/f'{args.prefix}_{name}_frames')]
        print('CASE_START', name, flush=True)
        with log.open('w') as stream:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        # App shutdown may mask a Python exception. Require the completed report.
        result = summarize_strike(json.loads(report.read_text()))
        print('CASE_RESULT', name, json.dumps({key: result[key] for key in
              ['outcome', 'contact', 'final_standing_seconds', 'completed_physics_seconds']}), flush=True)


if __name__ == '__main__':
    main()
