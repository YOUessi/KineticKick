#!/usr/bin/env python3
"""Run one continuous physical approach/shot/recovery with a frozen candidate.

The config reports which combinations passed acceptance. Unsupported combinations
fail explicitly; they are never replaced by another contact style or joint mode.
"""
from __future__ import annotations

import argparse
import copy
import math
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def verify_runup_config(config, root):
    if not config.get('source_sha256'):
        raise ValueError('Missing frozen source fingerprints')
    for name, digest in config['source_sha256'].items():
        if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Source changed after the recorded validation: {name}')
    from targeted_shooting import controller_provenance, UPSTREAM_SOURCE
    measured = controller_provenance(root)['controller_sources_sha256'][UPSTREAM_SOURCE]
    if config.get('upstream_sources_sha256') != measured:
        raise ValueError('The upstream physics/controller sources differ from the frozen validation')


def select_runup_trial(config, mode, style, target_x=None):
    entry = config['modes'].get(mode, {}).get(style)
    if entry is None:
        raise ValueError(f'No validated candidate for {mode}/{style}')
    if target_x is not None:
        if not math.isfinite(target_x):
            raise ValueError('Target must be finite')
        entry = next((command for command in entry.get('aim_commands', [])
                      if abs(command['target_x_m']-target_x) <= 1e-8), None)
        if entry is None or entry.get('status') != 'validated_in_listed_conditions':
            raise ValueError('This fixed-power target has not passed acceptance; no interpolation')
    return copy.deepcopy(entry['trial']), entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('hip', 'knee'), default='hip')
    parser.add_argument('--style', choices=('inside', 'instep', 'outside'), default='inside')
    parser.add_argument('--config', default=str(ROOT/'configs/runup_shooting_v2.json'))
    parser.add_argument('--target-x', type=float, help="A validated goal point at this motion's native strength")
    parser.add_argument('--seed', type=int, default=1710)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--video-dir')
    parser.add_argument('--output')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    try:
        verify_runup_config(config, ROOT)
        trial, entry = select_runup_trial(config, args.mode, args.style, args.target_x)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    trial.update(runup_config=str(config_path),
                 runup_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    output = Path(args.output).resolve() if args.output else ROOT/'reports'/f'runup_{args.mode}_{args.style}_{time.time_ns()}.json'
    if output.exists():
        parser.error('Choose a new output path to preserve prior evidence')
    command = [sys.executable, '-u', str(ROOT/'tools/eval_free_shooting.py'),
               '--sequence-evidence', '--steps', str(config['trial_steps']),
               '--style', args.style,
               '--seed', str(args.seed), '--output', str(output)]
    if args.video_dir:
        command += ['--video-dir', str(Path(args.video_dir).resolve())]
    elif not args.headless:
        command += ['--gui', '--hold']
    print(f"KineticKick {config['version']} | {args.mode}/{args.style} | {entry['status']}", flush=True)
    print(entry['scope'], flush=True)
    with tempfile.TemporaryDirectory(prefix='kinetickick-runup-') as temporary:
        manifest = Path(temporary)/'trial.json'
        manifest.write_text(json.dumps([trial], indent=2))
        subprocess.run([*command, '--trials', str(manifest)], cwd=ROOT, check=True)
    if not output.is_file():
        raise RuntimeError('Simulator exited without a measured result; this run did not complete')
    result = json.loads(output.read_text())
    print('RUNUP_MEASURED_RESULT', json.dumps(result.get('sequence_records', [])), flush=True)
    print('Evidence:', output, flush=True)


if __name__ == '__main__':
    main()
