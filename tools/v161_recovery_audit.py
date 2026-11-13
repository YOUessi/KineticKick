"""Audit complete post-kick runs against the frozen CPU episode with real data."""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from v160_tracking_ball_outcomes import summarize as ball_readout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_trace(folder):
    result = json.loads((folder / 'tracking_result.json').read_text())
    for name in ('rollout.npz', 'native_contacts.jsonl.gz'):
        if digest(folder / name) != result['local_artifact_sha256'][name]:
            raise ValueError(f'Changed trace {folder / name}')
    return np.load(folder / 'rollout.npz', allow_pickle=False), result


def audit(folder, baseline, protocol, limits):
    trial = json.loads((folder / 'trial.json').read_text())
    if trial['ball_initial_xyz_m'] != protocol['ball_initial_xyz_m'] or trial['clip'] != protocol['clip']:
        raise ValueError('Only the preregistered original scene is eligible for prefix comparison')
    tr, result = checked_trace(folder)
    base, original = checked_trace(baseline)
    with tr, base:
        times = tr['time_s']
        dt = protocol['physics_dt_s']
        if not np.array_equal(times, base['time_s']) or len(times) * dt != 14.:
            raise ValueError('The paired runs must cover all 14 seconds')
        end = original['reference_last_timestamp_s']
        prefix = times <= end + 1e-9
        prefix_commands = int(round(end / protocol['control_dt_s'])) + 1
        sampled = [key for key in base.files if base[key].shape[0] == len(times)]
        checks = {key: np.array_equal(tr[key][prefix], base[key][prefix]) for key in sampled}
        commands = 'policy_commands_time_frame_action_target'
        for key in [commands, 'policy_observations']:
            checks[key] = np.array_equal(tr[key][:prefix_commands], base[key][:prefix_commands])
        steps = int(np.sum(prefix))
        with gzip.open(folder / 'native_contacts.jsonl.gz', 'rb') as new, gzip.open(
                baseline / 'native_contacts.jsonl.gz', 'rb') as old:
            checks['native_contacts_200Hz_decompressed_lines'] = (
                list(itertools.islice(new, steps)) == list(itertools.islice(old, steps)))
        q = tr['joint_q_rad']
        dq = tr['joint_dq_rad_s']
        p = tr['root_pose_wxyz']
        vel = np.linalg.norm(tr['root_velocity_w'][:, :2], axis=1)
        tilt = np.degrees(np.arccos(np.clip(
            Rotation.from_quat(p[:, [4, 5, 6, 3]]).as_matrix()[:, 2, 2], -1., 1.)))
        load = tr['foot_ground_normal_load_N']
        nonfoot = tr['nonfoot_net_contact_peak_N']
        standing = ((p[:, 2] > .55) & (tilt <= 25.) & (vel < .3)
                    & np.all(load > 30., axis=1) & (nonfoot <= 20.))
        unstable = np.flatnonzero(~standing)
        tail_begin = int(unstable[-1] + 1) if len(unstable) else 0
        tail = (len(times) - tail_begin) * dt
        position_excess = np.maximum(limits[:, 0] - q, q - limits[:, 1])
        velocity_excess = np.abs(dq) - limits[:, 2]
        last = times > times[-1] - 1.
        last_load = load[last]
        edited = np.any(tr[commands][:, -29:] != tr['paid_targets_before_controller'], axis=1)
        payload = {
            'run': folder.name, 'git_head': trial['git_commit'],
            'controller': trial['controllers'][0]['file'],
            'sha256': result['local_artifact_sha256'], 'prefix_end_s': end,
            'prefix_checked_physics_samples': steps, 'prefix_checks': checks,
            'prefix_bit_identical_to_no_intervention': all(checks.values()),
            'first_changed_target_s': float(tr[commands][np.flatnonzero(edited)[0], 0]) if edited.any() else None,
            'max_nonfoot_net_N': float(nonfoot.max()),
            'max_joint_speed_fraction': float((abs(dq) / limits[:, 2]).max()),
            'joint_position_sample_joint_violations': int((position_excess > 1e-6).sum()),
            'joint_velocity_sample_joint_violations': int((velocity_excess > 1e-6).sum()),
            'min_measured_joint_position_limit_margin_rad': float(-position_excess.max()),
            'final_standing_time_s': tail, 'standing_first_sample_of_final_streak_s':
                float(times[tail_begin]) if tail > 0 else None,
            'final_pelvis_height_m': float(p[-1, 2]), 'final_tilt_deg': float(tilt[-1]),
            'final_speed_xy_mps': float(vel[-1]), 'final_left_right_ground_load_N': load[-1].tolist(),
            'last_one_second': {'pelvis_height_range_m': [float(p[last, 2].min()), float(p[last, 2].max())],
                'max_tilt_deg': float(tilt[last].max()), 'max_speed_xy_mps': float(vel[last].max()),
                'min_ground_load_left_right_N': last_load.min(axis=0).tolist()},
        }
    if not np.isclose(tail, result['final_standing_seconds_cpu_definition']) or (
            payload['joint_position_sample_joint_violations'] != result['q_position_limit_sample_joint_count']
            or payload['joint_velocity_sample_joint_violations'] != result['qdot_limit_sample_joint_count']):
        raise ValueError('Independent safety calculations differ from runner')
    payload['ball'] = ball_readout(folder, protocol)
    return payload


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--baseline', type=Path, required=True)
    cli.add_argument('--runs', nargs='+', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads((repo / 'configs/v160_tracking_protocol.json').read_text())
    cfg = json.loads((repo / 'configs/v161_recovery_support.json').read_text())
    model = ET.parse(cfg['urdf'])
    original = json.loads((args.baseline / 'tracking_result.json').read_text())
    specs = [model.find(f"joint[@name='{name}']/limit") for name in original['source_joint_names']]
    limits = np.array([[float(joint.get(key)) for key in ('lower', 'upper', 'velocity')] for joint in specs])
    reports = [audit(run, args.baseline, protocol, limits) for run in args.runs]
    out = {'scope': 'Reread previously completed center-scene CPU rounds, no new physics runs. '
                    'Original CPU collision proxy and approximate material contact classification still apply.',
           'audit_sha256': digest(Path(__file__)), 'plant_urdf_sha256': digest(Path(cfg['urdf'])),
           'baseline': str(args.baseline), 'runs': reports}
    with args.output.open('x') as write:
        json.dump(out, write, indent=2, allow_nan=False)
        write.write('\n')


if __name__ == '__main__':
    main()
