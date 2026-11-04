"""Offline contact/ball/balance diagnostics; these metrics do not certify control success."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from shooting_posture import rotate_vector


def physical_rows(trace):
    rows = list(trace)
    if rows and rows[-1].get('terminated'):
        if (len(rows) < 2 or not np.isfinite([rows[-1]['time_s'], rows[-2]['time_s']]).all() or
                abs(rows[-1]['time_s']-rows[-2]['time_s']-1e-6) > 1e-8):
            raise ValueError('Invalid terminal marker')
        rows.pop()  # Evaluator duplicates the last physical row for termination.
    if not rows:
        raise ValueError('Missing physical samples')
    times = np.asarray([row['time_s'] for row in rows], float)
    steps = np.asarray([row['physics_dt_s'] for row in rows], float)
    if (not np.all(np.isfinite(times)) or not np.all(np.isfinite(steps)) or
            not np.allclose(steps, .005, rtol=0, atol=1e-9) or
            not np.allclose(np.diff(times), steps[1:], rtol=0, atol=1e-7) or
            any(row.get('terminated') for row in rows)):
        raise ValueError('Expected finite, uninterrupted 5 ms physical samples')
    return rows


def contact_diagnostics(trace, direction, foot='right'):
    rows = physical_rows(trace)
    direction = np.asarray(direction, float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1e-8:
        raise ValueError('Expected a finite nonzero shot direction')
    direction = direction / np.linalg.norm(direction)
    velocities = np.asarray([row['ball_velocity'] for row in rows], float)
    if velocities.shape != (len(rows), 3) or not np.all(np.isfinite(velocities)):
        raise ValueError('Invalid ball velocity evidence')
    episodes, active, first_index = [], None, None
    total_vector = np.zeros(3)
    total_scalar = 0.
    for i, row in enumerate(rows):
        scalar, vector = 0., np.zeros(3)
        for event in row['contacts']:
            force = float(event['force_N'])
            if not np.isfinite(force) or force < 0:
                raise ValueError('Invalid normal force')
            if event['foot'] != foot or force == 0:
                continue
            normal = np.asarray(event['normal_foot'], float)
            if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-8:
                raise ValueError('Invalid contact normal')
            world_normal = rotate_vector(event['foot_quat_w'], normal/np.linalg.norm(normal))
            impulse = force * row['physics_dt_s']
            scalar += impulse
            vector += impulse * world_normal
        if scalar > 0:
            if first_index is None:
                first_index = i
            if active is None:
                start = row['time_s'] - row['physics_dt_s']
                active = dict(first_force_sample_s=row['time_s'], interval_start_s=start,
                    last_force_sample_s=row['time_s'], sampled_force_duration_s=0.,
                    normal_impulse_magnitude_sum_Ns=0., normal_impulse_vector_Ns=np.zeros(3),
                    gap_from_previous_s=start-episodes[-1]['last_force_sample_s'] if episodes else None,
                    ended_with_zero_force_sample=False, first_zero_force_sample_s=None,
                    first_zero_force_ball_velocity_mps=None)
                episodes.append(active)
            active['last_force_sample_s'] = row['time_s']
            active['sampled_force_duration_s'] += row['physics_dt_s']
            active['normal_impulse_magnitude_sum_Ns'] += scalar
            active['normal_impulse_vector_Ns'] += vector
            total_scalar += scalar
            total_vector += vector
        elif active is not None:
            active.update(ended_with_zero_force_sample=True, first_zero_force_sample_s=row['time_s'],
                          first_zero_force_ball_velocity_mps=velocities[i].tolist())
            active = None
    for episode in episodes:
        vector = episode['normal_impulse_vector_Ns']
        episode['normal_impulse_toward_goal_Ns'] = float(vector @ direction)
        episode['normal_impulse_vector_Ns'] = vector.tolist()
        velocity = episode['first_zero_force_ball_velocity_mps']
        episode['first_zero_force_ball_speed_xy_mps'] = float(np.linalg.norm(velocity[:2])) if velocity is not None else None
        episode['first_zero_force_ball_speed_toward_goal_mps'] = float(np.asarray(velocity) @ direction) if velocity is not None else None
    precontact = velocities[:first_index] if first_index is not None else velocities
    return dict(foot=foot, positive_force_episodes=episodes,
                normal_impulse_magnitude_sum_Ns=total_scalar,
                normal_impulse_vector_Ns=total_vector.tolist(),
                normal_impulse_toward_goal_Ns=float(total_vector @ direction),
                max_ball_speed_xy_before_first_foot_force_mps=float(np.max(np.linalg.norm(precontact[:, :2], axis=1))) if len(precontact) else None)


def balance_proxies(trace, start_s, end_s):
    """Standing-envelope descriptors, not COM/support-polygon stability criteria."""
    rows = physical_rows(trace)
    if not np.isfinite([start_s, end_s]).all() or end_s <= start_s:
        raise ValueError('Invalid observation window')
    selected = [row for row in rows if row['time_s'] > start_s and row['time_s']-row['physics_dt_s'] < end_s]
    overlaps = [max(0., min(end_s, row['time_s'])-max(start_s, row['time_s']-row['physics_dt_s'])) for row in selected]
    # Rounding-only overlap must not admit the next sample into extrema.
    observed = [(row, weight) for row, weight in zip(selected, overlaps) if weight > 1e-9]
    if not observed:
        return dict(coverage_complete=False, observed_duration_s=0.)
    selected, overlaps = zip(*observed)
    values = np.asarray([[*row['root_xyz'], row['root_speed_xy'], row['root_tilt_deg'],
                          row['feet_support_N'], row['nonfoot_contact_N']] for row in selected], float)
    if values.shape != (len(selected), 7) or not np.all(np.isfinite(values)):
        raise ValueError('Invalid root/contact evidence')
    weights = np.asarray(overlaps)
    return dict(coverage_complete=bool(rows[0]['time_s']-rows[0]['physics_dt_s'] <= start_s+1e-8 and rows[-1]['time_s'] >= end_s-1e-8),
        observed_duration_s=float(weights.sum()), root_height_min_m=float(values[:, 2].min()),
        root_speed_xy_max_mps=float(values[:, 3].max()), root_tilt_max_deg=float(values[:, 4].max()),
        time_root_tilt_above_25deg_s=float(weights[values[:, 4] > 25].sum()),
        time_foot_net_vertical_force_below_80N_s=float(weights[values[:, 5] < 80].sum()),
        nonfoot_net_force_peak_N=float(values[:, 6].max()))


def audit_functional_evidence(run, index):
    ball, goal = np.asarray(run['points'][index], float), np.asarray(run['goals'][index], float)
    trial = run['trials'][index]
    if 'target' in trial:
        goal = goal.copy()
        goal[0] = trial['target']['target_x']
    result = contact_diagnostics(run['traces'][index], np.r_[goal-ball, 0.])
    result.update(style=trial.get('style', run.get('requested_style')), diagnostic_only=True,
        limitations=['Force episodes do not prove geometric separation.',
            'Normal-only impulse excludes friction and other robot-body contacts.',
            'Root/ankle-net-force proxies are not COM, ground support geometry or a stability certificate.'])
    episodes = result['positive_force_episodes']
    if episodes:
        start = episodes[0]['first_force_sample_s']
        result['first_750ms_balance_proxies'] = balance_proxies(run['traces'][index], start, start+.75)
        result['recovery_250_750ms_balance_proxies'] = balance_proxies(run['traces'][index], start+.25, start+.75)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results = []
    for path in args.runs:
        raw = Path(path).read_bytes()
        run = json.loads(raw)
        for i in range(len(run['trials'])):
            row = audit_functional_evidence(run, i)
            row.update(source=path, source_sha256=hashlib.sha256(raw).hexdigest(), index=i)
            results.append(row)
            print(path, i, 'force episodes', len(row['positive_force_episodes']),
                  'normal impulse Ns', round(row['normal_impulse_magnitude_sum_Ns'], 6))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
