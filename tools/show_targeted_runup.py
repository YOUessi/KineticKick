#!/usr/bin/env python3
"""Preview or collect one seed of an explicitly supplied frozen target matrix.

No default or example calibration is supplied. This parent-process wrapper may
import NumPy; the Isaac child receives only a manifest and existing evaluation
flags. Provenance is attached only after child output and unchanged on-disk
sources are verified. A collected run is evidence, not an acceptance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from runup_targeting import validate_runup_target_config, resolve_runup_targeted_trial, POWERS
from targeted_shooting import CONTROLLER_SOURCES, UPSTREAM_SOURCE

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = '.pretrained_checkpoints/paid_official/policy_30000.onnx'
ASSET_FILE = 'configs/paid_original/paid_main_resolved.urdf'
SEQUENCE_SOURCES = ('runup_sequence.py', 'runup_metrics.py', 'cartesian_strike.py', 'strike_velocity.py')


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Explicit frozen runup-targeting JSON; no calibration is bundled')
    parser.add_argument('--mode', choices=('hip', 'knee'), default='hip')
    parser.add_argument('--style', choices=('inside', 'instep', 'outside'), default='instep')
    parser.add_argument('--target-x', type=float)
    strength = parser.add_mutually_exclusive_group()
    strength.add_argument('--power', choices=POWERS)
    strength.add_argument('--speed-mps', type=float)
    parser.add_argument('--acceptance', action='store_true', help='Collect this mode/style nine-node grid at one frozen seed')
    parser.add_argument('--seed', type=int, default=1710)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--video-dir')
    parser.add_argument('--output')
    return parser


def build_request(args, config, *, manifest_path, output_path, root=ROOT, python_executable=sys.executable):
    """Resolve controls and construct child arguments without launching Isaac."""
    validate_runup_target_config(config)
    if args.acceptance:
        if (args.seed not in config['acceptance_seeds'] or args.target_x is not None
                or args.power is not None or args.speed_mps is not None or args.video_dir):
            raise ValueError('Acceptance requires a frozen seed, all nine commands and no rendering/command overrides')
        trials = [resolve_runup_targeted_trial(config, args.mode, args.style, x, power=p)
                  for p in POWERS for x in config['target_x_m']]
    else:
        if args.target_x is None:
            raise ValueError('Preview requires --target-x and exactly one of --power or --speed-mps')
        trials = [resolve_runup_targeted_trial(config, args.mode, args.style, args.target_x,
                                              power=args.power, speed_mps=args.speed_mps)]
    command = [python_executable, '-u', str(Path(root)/'tools/eval_free_shooting.py'),
               '--sequence-evidence', '--steps', str(config['trial_steps']), '--style', args.style,
               '--seed', str(args.seed), '--model', str(Path(root)/POLICY_FILE),
               '--output', str(output_path), '--trials', str(manifest_path)]
    if args.acceptance:
        command.append('--perturb-reset')
    elif args.video_dir:
        command += ['--video-dir', str(Path(args.video_dir).resolve())]
    elif not args.headless:
        command += ['--gui', '--hold']
    return dict(trials=trials, command=command, seed=args.seed, acceptance=args.acceptance,
                motion_files=[str((Path(root)/t['motion_file']).resolve()) for t in trials])


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure_provenance(config, root):
    """Verify actual files for a schema already checked by build_request.

    Kept separate from command validation so provenance can be tested against
    small real files without creating policy/checkpoint mocks.
    """
    root = Path(root)
    sources = {name: _hash(root/name) for name in config['source_sha256']}
    if sources != config['source_sha256']:
        raise ValueError('Controller/wrapper sources changed after freezing')
    policy, asset = _hash(root/POLICY_FILE), _hash(root/ASSET_FILE)
    if policy != config['policy_sha256'] or asset != config['asset_sha256']:
        raise ValueError('Policy or physical asset differs from the frozen config')
    motions = {}
    for styles in config['modes'].values():
        for entry in styles.values():
            path = (root/entry['base_trial']['motion_file']).resolve()
            digest = _hash(path)
            if digest != entry['motion_sha256']:
                raise ValueError('Motion reference differs from the frozen config')
            motions[str(path)] = digest
    upstream = root/UPSTREAM_SOURCE
    paths = sorted(upstream.rglob('*.py'))
    if not paths:
        raise ValueError('Missing upstream controller source tree')
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(upstream).as_posix().encode()+b'\0')
        digest.update(path.read_bytes()+b'\0')
    if digest.hexdigest() != config['upstream_sources_sha256']:
        raise ValueError('Upstream dynamics/controller sources changed after freezing')
    return dict(source_sha256=sources, policy_sha256=policy, asset_sha256=asset,
                motion_sources_sha256=motions, upstream_sources_sha256=digest.hexdigest())


def validate_child_result(result, config, request, measured):
    """Refuse to attest results from changed inputs, geometry or runtime files."""
    trials = request['trials']
    if (result.get('trials') != trials or result.get('seed') != request['seed']
            or result.get('perturb_reset') is not request['acceptance']
            or result.get('points') != [t['ball'] for t in trials]
            or result.get('goals') != [t['goal'] for t in trials]
            or result.get('policy_goals') != [t['policy_goal'] for t in trials]
            or result.get('root_fixed') is not False or result.get('post_initialization_state_writes') is not False
            or result.get('running_diagnostic') is not False or result.get('contact_evidence_version') != 2
            or result.get('sequence_evidence_version') != 1
            or result.get('goal_width_m') != 1.6 or result.get('goal_height_m') != 1.1
            or result.get('policy_sha256') != measured['policy_sha256']
            or result.get('asset_sha256') != measured['asset_sha256']):
        raise ValueError('Child output differs from the requested frozen physical trial')
    expected_controller = {name: measured['source_sha256'][name] for name in CONTROLLER_SOURCES}
    expected_controller[UPSTREAM_SOURCE] = measured['upstream_sources_sha256']
    expected_sequence = {name: measured['source_sha256']['tools/'+name] for name in SEQUENCE_SOURCES}
    if (result.get('controller_sources_sha256') != expected_controller
            or any(result.get('sequence_sources_sha256', {}).get(name) != value for name, value in expected_sequence.items())
            or result.get('motion_files') != request['motion_files']
            or result.get('motion_sources_sha256') != {p: measured['motion_sources_sha256'][p] for p in request['motion_files']}):
        raise ValueError('Child output content fingerprints differ from verified files')
    if any(len(result.get(name, [])) != len(trials) for name in ('traces', 'records', 'sequence_records')):
        raise ValueError('Child output is missing complete per-trial evidence arrays')
    end = config['trial_steps']*config['policy_dt_s']
    for trace in result['traces']:
        if (not trace or trace[-1]['time_s'] > end+1e-6
                or abs(trace[-1]['time_s']-end) > 1e-6 and not trace[-1]['terminated']):
            raise ValueError('Child output has an incomplete nonterminal trial')


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        config_path = Path(args.config).resolve()
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        output = Path(args.output).resolve() if args.output else ROOT/'reports'/f'targeted_runup_{args.mode}_{args.style}_{time.time_ns()}.json'
        if output.exists():
            raise ValueError('Choose a new output path to preserve previous evidence')
        with tempfile.TemporaryDirectory(prefix='kinetickick-targeted-runup-') as temporary:
            manifest = Path(temporary)/'trials.json'
            request = build_request(args, config, manifest_path=manifest, output_path=output)
            before = measure_provenance(config, ROOT)
            manifest.write_text(json.dumps(request['trials'], indent=2)+'\n')
            print(f"KineticKick {config['version']} | {args.mode}/{args.style} | frozen commands; measured outcome pending", flush=True)
            subprocess.run(request['command'], cwd=ROOT, check=True)
            if not output.is_file():
                raise RuntimeError('Simulator exited without a measured result; no provenance attached')
            if config_path.read_bytes() != config_bytes or measure_provenance(config, ROOT) != before:
                raise ValueError('Config or actual source/reference content changed during simulation')
            result = json.loads(output.read_text())
            validate_child_result(result, config, request, before)
            provenance = dict(runup_targeting_config_sha256=config_hash,
                runup_targeting_sources_sha256=before['source_sha256'],
                upstream_sources_sha256=before['upstream_sources_sha256'])
            if any(key in result and result[key] != value for key, value in provenance.items()):
                raise ValueError('Child output contains inconsistent targeting provenance')
            result.update(provenance)
            with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, prefix=output.name+'.', suffix='.tmp', delete=False) as saved:
                temporary_output = Path(saved.name)
                json.dump(result, saved, indent=2); saved.write('\n')
            try:
                temporary_output.replace(output)
            finally:
                temporary_output.unlink(missing_ok=True)
        print('RUNUP_TARGETED_MEASURED_RESULT', json.dumps([
            dict(name=t['name'], sequence_success=s['sequence_success'], target_success=r.get('target_success',False),
                 joint_success=bool(s['sequence_success'] and r.get('target_success',False)))
            for t, s, r in zip(result['trials'], result['sequence_records'], result['records'])]), flush=True)
        print('Evidence:', output, flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
