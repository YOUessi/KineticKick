#!/usr/bin/env python3
"""Evaluate the unchanged v0.1.15 runner with one frozen initial-condition case.

Only the initial reset configuration changes. No action interception, external
forces, post-initialization state writes, or controller parameter changes.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from strike_baseline_protocol import case_plan, verify_files
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=ROOT/'configs/strike_baseline_v1.json')
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    matrix = json.loads(args.matrix.read_text())
    matches = [c for c in matrix['cases'] if c['id'] == args.case]
    if len(matches) != 1:
        parser.error('Case must uniquely match the frozen matrix')
    case = matches[0]
    verify_files(ROOT, matrix['frozen_files_sha256'])
    plan = case_plan(json.loads((ROOT/matrix['baseline_plan']).read_text()), case)
    if matrix['steps'] != 700 or matrix['seed'] != 1201:
        raise ValueError('This matrix freezes the original 14s duration and seed1201')
    original_spec = importlib.util.spec_from_file_location
    source = ROOT/'tools/eval_free_shooting.py'

    def instrumented_spec(name, location, *aa, **kk):
        spec = original_spec(name, location, *aa, **kk)
        if Path(location).resolve() != source:
            return spec
        original_exec = spec.loader.exec_module

        def load(module):
            original_exec(module)
            native_run = module.run

            def observed_run(runtime_args, points):
                # Executed after AppLauncher, never import Torch/NumPy before it.
                import gymnasium as gym
                native_make = gym.make
                initial = {}

                def make(*pos, **kw):
                    cfg = kw['cfg']
                    dy = case['root_velocity_y_delta_mps']
                    cfg.commands.motion.velocity_range['y'] = (dy, dy)
                    env = native_make(*pos, **kw)
                    native_reset = env.reset

                    def reset(*rr, **rk):
                        result = native_reset(*rr, **rk)
                        robot = env.unwrapped.scene['robot']
                        initial.update(root_linear_velocity_w=robot.data.root_lin_vel_w.cpu().tolist(),
                                       root_angular_velocity_w=robot.data.root_ang_vel_w.cpu().tolist())
                        return result
                    env.reset = reset
                    return env
                gym.make = make
                try:
                    value = native_run(runtime_args, points)
                    result = json.loads(args.output.read_text())
                    result['strike_baseline'] = dict(case=case, initial_reset_observed=initial,
                        matrix_sha256=hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
                        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        protocol_sha256=hashlib.sha256((ROOT/'tools/strike_baseline_protocol.py').read_bytes()).hexdigest(),
                        frozen_files_sha256=matrix['frozen_files_sha256'], resolved_plan=plan,
                        initialization_only=True, controller_unchanged=True)
                    args.output.write_text(json.dumps(result, indent=2)+'\n')
                    return value
                finally:
                    gym.make = native_make
            module.run = observed_run
        spec.loader.exec_module = load
        return spec

    with tempfile.TemporaryDirectory(prefix='kinetickick-frozen-strike-') as folder:
        resolved = Path(folder)/'plan.json'
        resolved.write_text(json.dumps(plan))
        spec = original_spec('_frozen_original_stride', ROOT/'tools/run_stride_styles.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        previous_argv = sys.argv
        try:
            importlib.util.spec_from_file_location = instrumented_spec
            sys.argv = [str(ROOT/'tools/run_stride_styles.py'), '--plan', str(resolved),
                '--output', str(args.output), '--steps', str(matrix['steps']), '--seed', str(matrix['seed'])]
            if args.video_dir:
                sys.argv += ['--video-dir', str(args.video_dir)]
            runner.main()
        finally:
            importlib.util.spec_from_file_location = original_spec
            sys.argv = previous_argv


if __name__ == '__main__':
    main()
