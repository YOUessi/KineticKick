"""Post-hoc momentum balance and contact material-point estimate on recorded CPU PhysX ticks."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from free_shooting_metrics import paid_contact_patch


BALL = 'SoccerBall'
FOOT = 'right_ankle_roll_link'
PATCHES = ('inside', 'outside', 'instep', 'sole', 'toe', 'heel', 'ambiguous')
SOURCE_NAMES = ('foot', 'ground', 'other')
CONFIG = Path(__file__).resolve().parents[1] / 'configs/v160_tracking_protocol.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_contacts(path, time_s, dt):
    """Keep the contact-pair discovery test separate from summing *all* ball-point impulses."""
    count = len(time_s)
    impulses = np.zeros((count, len(SOURCE_NAMES), 3))
    pairs_seen = np.zeros((count, len(SOURCE_NAMES)), dtype=int)
    loaded = np.zeros(count, dtype=bool)
    foot_points = []
    end = 0
    with gzip.open(path, 'rt') as lines:
        for end, line in enumerate(lines, 1):
            tick, pairs = json.loads(line)
            if tick != end or tick > count:
                raise ValueError('Contact clock differs from velocity clock')
            for pair in pairs:
                actors = pair['actors']
                if BALL not in actors:
                    continue
                ball_id = actors.index(BALL)
                # PxContactPairPoint.normal goes from shape 1 toward shape 0.
                sign = 1 if ball_id == 0 else -1
                partner = actors[1-ball_id]
                group = 0 if partner == FOOT else 1 if partner == 'ground' else 2
                points = np.asarray(pair['points'], dtype=float)
                impulse_on_ball = sign * points[:, 6:9]
                pair_impulse = impulse_on_ball.sum(axis=0)
                impulses[tick-1, group] += pair_impulse
                pairs_seen[tick-1, group] += 1
                if partner == FOOT:
                    if np.linalg.norm(pair_impulse) / dt > 2.0:
                        loaded[tick-1] = True
                    for point, vector in zip(points, impulse_on_ball):
                        magnitude = np.linalg.norm(vector)
                        if magnitude > 0:
                            foot_points.append((tick, point, sign, magnitude,
                                               pair['shapes'][1-ball_id], pair['shapes'][ball_id]))
    if end != count:
        raise ValueError('Truncated contact clock')
    return loaded, impulses, pairs_seen, foot_points


def loaded_intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1) + 1, np.flatnonzero(edges == -1)))


def momentum_window(first, last, times, velocities, impulses, counts, mass, gravity, dt):
    """Recorded states are post-step; integrate ticks [first,last] using v(last)-v(first-1)."""
    if first < 2 or last > len(times) or last < first:
        raise ValueError('Need a measured pre-state and post-state for the momentum window')
    section = slice(first-1, last)
    sums = impulses[section].sum(axis=0)
    gravity_j = mass * np.asarray(gravity) * dt * (last-first+1)
    dp = mass * (velocities[last-1] - velocities[first-2])
    residual = dp - gravity_j - sums.sum(axis=0)
    ticks = dict(pre_first=first-1, post_first=first, pre_last=last-1, post_last=last)
    if last < len(times):
        ticks['following_tick'] = last+1
    return {
        'first_last_ticks_inclusive': [int(first), int(last)],
        'state_clock': {name: {'tick': int(k), 'time_s': float(times[k-1]),
                               'ball_velocity_world_mps': velocities[k-1].tolist()} for name,k in ticks.items()},
        'pair_records_including_zero_impulse': dict(zip(SOURCE_NAMES, counts[section].sum(axis=0).tolist())),
        'reported_normal_impulse_on_ball_world_Ns': dict(zip(SOURCE_NAMES, sums.tolist())),
        'gravity_world_Ns': gravity_j.tolist(), 'ball_mass_delta_velocity_world_Ns': dp.tolist(),
        'closure_residual_world_Ns': residual.tolist(), 'closure_residual_norm_Ns': float(np.linalg.norm(residual)),
    }


def point_geometry(point, sign, origin, rot, *, foot_witness):
    outward = sign * point[3:6]
    # This run's native point equals the *pre-step* ball-side witness centre - radius*normal.
    # Reconstruct the opposing foot-side witness from the signed gap along the normal.
    world = point[:3] - (point[9] * outward if foot_witness else 0.)
    return rot.inv().apply(world-origin), rot.inv().apply(outward)


def approach_velocity(local_point, positions, rotations, last_pre_tick, dt):
    positions_w = np.array([rotations[k-1].apply(local_point) + positions[k-1]
                            for k in (last_pre_tick-2, last_pre_tick-1, last_pre_tick)])
    earlier, previous, current = positions_w
    first_order = (current-previous) / dt
    second_order = (3*current-4*previous+earlier) / (2*dt)
    return {'world_mps': first_order.tolist(), 'speed_mps': float(np.linalg.norm(first_order)),
            'bdf2_world_mps': second_order.tolist(),
            'difference_from_bdf2_norm_mps': float(np.linalg.norm(second_order-first_order))}


def contact_patch_readout(foot_points, times, foot_xyz, foot_quat, ball_xyz, radius, dt):
    rotations = Rotation.from_quat(foot_quat[:, [1, 2, 3, 0]])
    first_tick = foot_points[0][0]
    if first_tick < 4 or sum(p[0] == first_tick for p in foot_points) != 1:
        raise ValueError('Expected one first loaded point and three measured pre-contact poses')
    totals = {}
    first_readouts = {}
    witness_errors = []
    normal_orientation = []
    events = []
    for tick, point, sign, magnitude, foot_shape, ball_shape in foot_points:
        outward = sign * point[3:6]
        prediction = ball_xyz[tick-2] - radius * outward
        witness_errors.append(np.linalg.norm(point[:3]-prediction))
        normal_orientation.append(float(outward @ (ball_xyz[tick-2]-foot_xyz[tick-2])))
        row = {'tick': int(tick), 'scalar_normal_impulse_Ns': float(magnitude),
               'applied_world_Ns': (sign*point[6:9]).tolist(), 'raw_point_world_m': point[:3].tolist(),
               'outward_normal_world': outward.tolist(), 'separation_m': float(point[9]),
               'foot_shape': foot_shape, 'ball_shape': ball_shape}
        for proxy, offset, foot_witness in [('pre_foot_witness', -1, True), ('pre_native_point', -1, False),
                                          ('post_foot_witness', 0, True), ('post_native_point', 0, False)]:
            index = tick+offset-1
            local, local_n = point_geometry(point, sign, foot_xyz[index], rotations[index], foot_witness=foot_witness)
            patch = paid_contact_patch(local, local_n, foot='right')
            weights = totals.setdefault(proxy, dict.fromkeys(PATCHES, 0.0))
            weights[patch] += float(magnitude)
            if tick == first_tick:
                first_readouts[proxy] = {'local_xyz_m': local.tolist(), 'outward_normal_local': local_n.tolist(),
                                          **approach_velocity(local, foot_xyz, rotations, first_tick-1, dt)}
            if proxy == 'pre_foot_witness':
                row.update(local_xyz_m=local.tolist(), outward_normal_local=local_n.tolist(), patch_proxy=patch)
        events.append(row)
    patch_shares = {key: {s: j/sum(amounts.values()) for s,j in amounts.items()} for key,amounts in totals.items()}
    return {
        'loaded_points': events, 'normal_scalar_total_Ns': sum(totals['pre_foot_witness'].values()),
        'per_patch_reported_normal_scalar_Ns': totals['pre_foot_witness'], 'per_patch_scalar_fractions': patch_shares,
        'max_native_point_distance_to_prestep_ball_witness_m': float(max(witness_errors)),
        'min_outward_normal_dot_ball_minus_foot_origin_m': float(min(normal_orientation)),
        'material_point_prestep_pose_times_s': times[first_tick-4:first_tick-1].tolist(),
        'material_point_velocity_primary': 'pre_foot_witness', 'first_point_estimates': first_readouts,
        'scope': 'Pre-step foot witness = reported ball-side point minus separation times foot-to-ball normal. '
                 'World velocity is the backward rigid-point difference between the last two pre-contact states. '
                 'BDF2 and witness/pose variations are sensitivities, not an accuracy bound. '
                 'The legacy PAiD patch rule uses an uncalibrated CPU seven-capsule proxy; formal instep qualification is unknown.',
    }


def summarize(run, config_path):
    cfg = json.loads(config_path.read_text())
    saved = json.loads((run / 'tracking_result.json').read_text())
    schema = json.loads((run / 'source_inventory.json').read_text())
    hashes = {name: digest(run/name) for name in ['rollout.npz', 'native_contacts.jsonl.gz', 'source_inventory.json']}
    if digest(config_path) != saved['protocol_sha256'] or any(hashes[name] != value for name,value in saved['local_artifact_sha256'].items()):
        raise ValueError('Input differs from original recorded rollout or frozen protocol')
    dt = cfg['physics_dt_s']
    with np.load(run/'rollout.npz', allow_pickle=False) as trace:
        times = trace['time_s']
        if not np.allclose(times, np.arange(1, len(times)+1)*dt, atol=1e-8, rtol=0):
            raise ValueError('Irregular state clock')
        loaded, impulses, counts, points = collect_contacts(run/'native_contacts.jsonl.gz', times, dt)
        ranges = loaded_intervals(loaded)
        if not ranges:
            raise ValueError('No loaded right-foot interval in saved run')
        labels = [(f'episode_{i}', pair) for i,pair in enumerate(ranges, 1)]
        labels.append(('including_inter_episode_gap', (ranges[0][0], ranges[-1][1])))
        velocity = trace['ball_velocity_w_mps'].astype(float)
        windows = {name: momentum_window(a, b, times, velocity, impulses, counts, cfg['ball_mass_kg'],
                                          cfg['gravity_w_mps2'], dt) for name,(a,b) in labels}
        foot = schema['body_names'].index(FOOT)
        window_points = [p for p in points if ranges[0][0] <= p[0] <= ranges[-1][1]]
        patch = contact_patch_readout(window_points, times, trace['body_position_w_m'][:, foot],
                                     trace['body_quat_wxyz'][:, foot], trace['ball_position_w_m'].astype(float),
                                     cfg['ball_radius_m'], dt)
    return {'analysis': 'post-hoc on a single archived CPU trajectory', 'new_physical_runs': 0,
            'input_sha256': {**hashes, 'v160_tracking_protocol.json': digest(config_path)},
            'code_sha256': {p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name('free_shooting_metrics.py')]},
            'units_axes': 'N s and m/s in the original right-handed world xyz, +z up',
            'sign': 'Native normal goes shape1 to shape0; impulse on ball is +raw for ball as shape0, -raw for shape1.',
            'windows': windows, 'foot_patch_proxy_and_precontact_material_velocity': patch,
            'full_source_impulse_including_unobserved_friction_world_Ns': {'foot': None, 'ground': None},
            'native_scope': 'SAPIEN simulation_callback.hpp calls PxContactPair::extractContacts; '
                            'PxSimulationEventCallback.h constructs point.impulse = point.normal * scalar_impulse. '
                            'Friction-anchor impulses are absent. The residual m*delta_v - gravity - sum(ALL reported '
                            'normal impulses) is unassigned. A pair record with zero impulse is distinguished from no pair.',
            'formal_instep_patch_qualification': None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.run, CONFIG)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write('\n')
    print(f'{args.output}: sha256 {digest(args.output)}')
