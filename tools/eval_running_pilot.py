#!/usr/bin/env python3
"""Evaluate a running actor with an explicit 0/1-frame reference lead.

Reuses eval_free_shooting unchanged. Only actor observation columns [0:58]
(reference q/dq) and [61:64] (reference torso world angular velocity) can change.
Isaac starts before this wrapper imports NumPy or the ONNX policy adapter.

Example:
  python tools/eval_running_pilot.py --trials trials.json --output result.json \
    --model running.onnx --running-diagnostic --sequence-evidence --steps 110

Other options are forwarded to eval_free_shooting. Default reference lead is 1,
matching its training actor's persistent reference lead after automatic resets.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_root():
    for candidate in (Path(__file__).resolve().parents[1], Path.cwd()):
        if (candidate / 'tools/eval_free_shooting.py').is_file():
            return candidate
    return Path('/home/you/projects/kinetickick')


def validate_references(root, trials_path):
    """Standard-library-only preflight; safe before AppLauncher."""
    root = Path(root).resolve()
    trials = json.loads(Path(trials_path).read_text())
    if not isinstance(trials, list) or not 1 <= len(trials) <= 24:
        raise ValueError('Require a list containing 1..24 running trials')
    schema_path = root / 'reports/paid_physics_parity_clean_paid.json'
    schema = json.loads(schema_path.read_text())
    if len(schema['joint_names']) != 29 or len(schema['body_names']) != 30:
        raise ValueError('Expected the verified original 29-joint, 30-body runtime schema')
    references = []
    for trial in trials:
        if not isinstance(trial, dict) or not isinstance(trial.get('motion_file'), str):
            raise ValueError('Every trial must explicitly name its running motion_file')
        if trial.get('reference'):
            raise ValueError('Running reference lead cannot be combined with reference residual control')
        motion = Path(trial['motion_file'])
        if not motion.is_absolute():
            motion = root / motion
        motion = motion.resolve()
        manifest_path = motion.with_suffix('.json')
        manifest = json.loads(manifest_path.read_text())
        if motion.suffix != '.npz' or sha(motion) != manifest['output_sha256']:
            raise ValueError(f'Running NPZ hash differs from companion manifest: {motion}')
        for key in ('joint_names', 'body_names'):
            if manifest[key] != schema[key]:
                raise ValueError(f'Running reference {key} order differs from verified runtime: {motion}')
        if manifest.get('angular_velocity_frame') != 'world':
            raise ValueError('Reference torso angular velocity must be expressed in world coordinates')
        if float(manifest['fps']) != 50. or int(manifest['frames']) < 2:
            raise ValueError('Expected the original 50 Hz control reference with at least two frames')
        references.append(dict(motion_file=str(motion), motion_sha256=sha(motion),
            manifest_file=str(manifest_path), manifest_sha256=sha(manifest_path),
            frames=int(manifest['frames']), fps=50., joint_names=manifest['joint_names'],
            body_names=manifest['body_names'], torso_index=manifest['body_names'].index('torso_link')))
    return references


def load_arrays(references):
    """Called only after AppLauncher in the simulation path."""
    import numpy as np
    loaded = []
    for ref in references:
        if sha(ref['motion_file']) != ref['motion_sha256'] or sha(ref['manifest_file']) != ref['manifest_sha256']:
            raise ValueError('Reference changed after preflight')
        with np.load(ref['motion_file'], allow_pickle=False) as source:
            q = np.array(source['joint_pos'], dtype=np.float32, copy=True)
            dq = np.array(source['joint_vel'], dtype=np.float32, copy=True)
            omega = np.array(source['body_ang_vel_w'], dtype=np.float32, copy=True)
            if q.shape != (ref['frames'], 29) or dq.shape != q.shape or omega.shape != (ref['frames'], 30, 3):
                raise ValueError('Reference array dimensions differ from companion manifest')
            if float(np.asarray(source['fps']).reshape(-1)[0]) != ref['fps']:
                raise ValueError('Reference fps differs from companion manifest')
        if not all(np.isfinite(array).all() for array in (q, dq, omega)):
            raise ValueError('Reference contains nonfinite actor commands')
        loaded.append((q, dq, omega[:, ref['torso_index']]))
    return loaded


def apply_reference_lead(observations, frames, arrays, lead):
    """Pure transformation; measured state, previous actions and points untouched."""
    import numpy as np
    if lead not in (0, 1):
        raise ValueError('reference lead must be 0 or 1')
    obs = np.asarray(observations, dtype=np.float32)
    if obs.shape != (len(arrays), 160) or not np.isfinite(obs).all():
        raise ValueError('Expected finite (num_trials, 160) actor observations')
    steps = np.broadcast_to(np.asarray(frames), (len(arrays),))
    if not np.isfinite(steps).all() or np.any(steps < 0) or not np.equal(steps, np.floor(steps)).all():
        raise ValueError('Reference frames must be finite nonnegative integers')
    result = obs.copy()
    if lead:
        for i, (q, dq, omega) in enumerate(arrays):
            index = min(int(steps[i]) + lead, len(q) - 1)
            result[i, :29] = q[index]
            result[i, 29:58] = dq[index]
            result[i, 61:64] = omega[index]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=default_root())
    parser.add_argument('--reference-lead', type=int, choices=(0, 1), default=1)
    parser.add_argument('--trials', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--running-diagnostic', action='store_true', required=True)
    parser.add_argument('--sequence-evidence', action='store_true', required=True)
    own, remaining = parser.parse_known_args(argv)
    root = own.root.resolve()
    output = Path(own.output)
    if output.exists():
        parser.error('Preserve existing evidence; choose a new output path')
    references = validate_references(root, own.trials)
    base_path = root / 'tools/eval_free_shooting.py'
    sys.path.insert(0, str(base_path.parent))
    spec = importlib.util.spec_from_file_location('_running_pilot_base_evaluator', base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)  # Base module imports only standard-library modules.
    native_run = base.run

    def run_after_isaac(args, points):
        # base.main() invokes this hook only AFTER AppLauncher has initialized.
        import soccer_onnx_policy as policy_module
        original_policy = policy_module.OnnxSoccerPolicy
        resolved = [str(Path(path).resolve()) for path in args.motion_files]
        if resolved != [ref['motion_file'] for ref in references] or args.num_envs != len(references):
            raise ValueError('Resolved evaluator motion order differs from the validated trials')
        if not args.running_diagnostic or not args.sequence_evidence:
            raise ValueError('This wrapper requires running diagnostic and physical sequence evidence')
        arrays = load_arrays(references)

        class RunningLeadPolicy(original_policy):
            def __init__(self, path, num_envs):
                super().__init__(path, num_envs)
                if self.observation_names[:3] != ['command', 'projected_gravity', 'motion_ref_ang_vel']:
                    raise ValueError('Actor observation ordering does not support the reference slices')
                if any(self.joint_names != ref['joint_names'] for ref in references):
                    raise ValueError('Actor joint order differs from running reference manifest')

            def step(self, observations, frame):
                aligned = apply_reference_lead(observations, frame, arrays, own.reference_lead)
                return super().step(aligned, frame)

        args.source_provenance.update(
            running_evaluation_wrapper=str(Path(__file__).resolve()),
            running_evaluation_wrapper_sha256=sha(__file__),
            reference_lead_frames=own.reference_lead,
            reference_lead_seconds=own.reference_lead / 50.,
            actor_reference_observation_slices=[[0, 58], [61, 64]] if own.reference_lead else [],
            actor_reference_alignment='next-frame actor command after auto-reset' if own.reference_lead else 'same-frame actor command',
            reference_manifest_sources=references,
            running_runtime_schema_sha256=sha(root / 'reports/paid_physics_parity_clean_paid.json'))
        policy_module.OnnxSoccerPolicy = RunningLeadPolicy
        try:
            return native_run(args, points)
        finally:
            policy_module.OnnxSoccerPolicy = original_policy

    base.run = run_after_isaac
    old_argv = sys.argv
    sys.argv = [str(base_path), *remaining, '--trials', own.trials, '--output', own.output,
                '--running-diagnostic', '--sequence-evidence']
    try:
        base.main()
    finally:
        sys.argv = old_argv
        base.run = native_run


if __name__ == '__main__':
    main()
