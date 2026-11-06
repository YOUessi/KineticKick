#!/usr/bin/env python3
"""Run the frozen development cases and declared combinations serially."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
from strike_cases import load_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/state_conditioned_strike_v5.json')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--video-cases', nargs='*', default=[])
    parser.add_argument('--dynamics-probe',type=Path,help='Record normal/friction and segment telemetry with the probed observer')
    parser.add_argument('--stop-on-physical-failure',action='store_true',help='Stop after preserving a failed recovery/physical episode or observed joint overspeed')
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix:
        parser.error('Prefix must be a filename component')
    cases = load_cases(ROOT)
    if len(set(args.cases)) != len(args.cases) or any(name not in cases for name in args.cases):
        parser.error('Unique declared case ids required')
    if args.dynamics_probe and args.video_cases:parser.error('Instrumented matrix and rendered cases use separate explicit runs')
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
        if args.dynamics_probe:
            cmd=[sys.executable,str(ROOT/'tools/run_instrumented_strike.py'),'--controller','placement','--config',str(args.config.resolve()),'--case',name,'--probe',str(args.dynamics_probe.resolve()),'--output',str(report)]
        if name in args.video_cases:
            cmd += ['--video-dir', str(ROOT/'experiments/state_conditioned_v028'/f'{args.prefix}_{name}_frames')]
        print('CASE_START', name, flush=True)
        with log.open('w') as stream:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        # App shutdown may mask a Python exception. Require the completed report.
        recorded=json.loads(report.read_text())
        result = summarize_strike(recorded)
        print('CASE_RESULT', name, json.dumps({key: result[key] for key in
              ['outcome', 'contact', 'final_standing_seconds', 'completed_physics_seconds']}), flush=True)
        if args.stop_on_physical_failure:
            from audit_p0_dynamic_strike import limit_observations
            limits=limit_observations(recorded['traces'][0],recorded)
            good=recorded['sequence_records'][0]['recovery']['passed'] and not any(recorded['termination_reasons'])
            good &= recorded['records'][0]['surface_match'] and not recorded['sequence_records'][0]['illegal_body_ball_contact']
            good &= limits['maximum_actual_speed_ratio']<=1.0001
            if not good:
                print('MATRIX_STOP_PHYSICAL_FAILURE',name,json.dumps(limits),flush=True)
                raise SystemExit(2)


if __name__ == '__main__':
    main()
