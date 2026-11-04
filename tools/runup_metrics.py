"""Measured short approach, leg coordination and final recovery evidence.

Velocity contributions are signed world-frame kinematic contributions at one
fixed foot point. They are not motor work, impact energy or causal attribution
of ball acceleration. A sequence pass must still be combined with independent
ball/goal and contact-surface checks. No gait-flight/running claim is made.
"""
from __future__ import annotations

import numpy as np


def _vector(value, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'Expected a finite {size}-vector')
    return result


def summarize_trial(result, index):
    """Combine sequence and shot gates without certifying illegal body strikes."""
    trace, trial, shot = result['traces'][index], result['trials'][index], result['records'][index]
    direction = trial.get('goal_forward', np.asarray(result['goals'][index])-result['points'][index])
    summary = summarize_sequence(trace, result['joint_names'], shot['contact_time_s'],
        forward=direction, initial_root=result['initial_robot_state']['root_position'][index],
        requested_mode=trial.get('mode', 'hip'))
    coverage, illegal, first = True, False, None
    legal = {'left_ankle_roll_link', 'right_ankle_roll_link'}
    for row in trace:
        forces = row.get('robot_ball_contact_force_N')
        if not isinstance(forces, dict):
            coverage = False
            continue
        for body, force in forces.items():
            if not np.isfinite(force) or force < 0:
                raise ValueError('Finite nonnegative all-body contact force required')
            if force > 2.:
                illegal |= body not in legal
                if first is None:
                    first = dict(time_s=row['time_s'], body=body, force_N=float(force))
    primary_foot_match = bool(first and first['body'] == result.get('contribution_foot', '')+'_ankle_roll_link')
    physical = result.get('root_fixed') is False and result.get('post_initialization_state_writes') is False
    summary.update(all_body_contact_coverage_complete=coverage, illegal_body_ball_contact=illegal,
        first_body_ball_contact=first, contribution_foot_matches_contact=primary_foot_match,
        physical_integrity=physical, shot_success=bool(shot['success']),
        sequence_success=bool(summary['motion_sequence_pass'] and shot['success'] and physical
                              and coverage and not illegal and primary_foot_match))
    return summary


def summarize_sequence(trace, joint_names, contact_time_s, *, forward, initial_root, requested_mode):
    """Summarize 5 ms, post-substep telemetry beginning at trial time zero.

    The mode window ends strictly before the first observed positive foot-ball
    force, including forces below the older shot tracker's 2 N threshold. The
    final 5 ms collision bin is excluded. At least 95% of the preceding 150 ms
    must have valid, contiguous joint and contribution evidence.
    """
    if requested_mode not in ('knee', 'hip'):
        raise ValueError('Requested mode must be knee or hip')
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError('Expected unique joint names')
    direction = np.array(forward, dtype=float, copy=True)
    if direction.shape == (2,):
        direction = np.r_[direction, 0.]
    direction = _vector(direction)
    direction[2] = 0.
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        raise ValueError('Forward must have a horizontal component')
    direction /= length
    initial = _vector(initial_root)
    if contact_time_s is not None and (not np.isfinite(contact_time_s) or contact_time_s < 0):
        raise ValueError('Contact time must be finite and nonnegative')

    rows = list(trace)
    times, intervals, roots, speeds, tilts, nonfoot = [], [], [], [], [], []
    observed_contact = None
    contact_forces = []
    for row in rows:
        t, dt = float(row['time_s']), float(row.get('physics_dt_s', .005))
        scalars = [t, dt, row['root_speed_xy'], row['root_tilt_deg'], row['nonfoot_contact_N']]
        if not np.all(np.isfinite(scalars)) or t < 0 or dt <= 0 or dt > .0075:
            raise ValueError('Expected finite telemetry sampled at 5 ms resolution')
        if times and t <= times[-1]:
            raise ValueError('Telemetry times must be strictly increasing')
        if times and t - dt < times[-1] - 1e-9 and not row.get('terminated', False):
            raise ValueError('Physical sampling bins must not overlap')
        if min(float(row[key]) for key in ('root_speed_xy', 'root_tilt_deg', 'nonfoot_contact_N')) < 0:
            raise ValueError('Speeds, tilt magnitude and contact magnitude must be nonnegative')
        force = 0.
        for event in row.get('contacts', []):
            value = float(event['force_N'])
            if not np.isfinite(value) or value < 0:
                raise ValueError('Contact force must be finite and nonnegative')
            if event.get('foot') in ('left', 'right'):
                force += value
        if force > 0 and observed_contact is None:
            observed_contact = t
        times.append(t)
        intervals.append(dt)
        roots.append(_vector(row['root_xyz']))
        speeds.append(float(row['root_speed_xy']))
        tilts.append(float(row['root_tilt_deg']))
        nonfoot.append(float(row['nonfoot_contact_N']))
        contact_forces.append(force)

    times = np.asarray(times)
    intervals = np.asarray(intervals)
    roots = np.asarray(roots).reshape(-1, 3)
    gaps = np.diff(times)
    contiguous = bool(len(rows) and times[0] <= intervals[0] + 1e-9
                      and np.all(gaps <= intervals[1:] * 1.5 + 1e-9))
    failed = bool(any(row.get('terminated', False) for row in rows)
                  or np.any(roots[:, 2] < .45) or np.any(np.asarray(tilts) > 60.))
    nonfoot_seen = bool(any(value >= 20. for value in nonfoot))

    # Integration uses only bins entirely preceding actual contact. Missing
    # values remove their intervals rather than being filled with favorable zeros.
    window_end = observed_contact
    window_start = max(0., window_end - .15) if window_end is not None else None
    amounts = {'hip': 0., 'knee': 0.}
    duration = 0.
    valid_times = []
    if window_end is not None:
        for row, t, dt in zip(rows, times, intervals):
            overlap = min(t, window_end) - max(t - dt, window_start)
            if t >= window_end - 1e-9 or overlap <= 1e-9:
                continue
            try:
                _vector(row['joint_pos'], len(joint_names))
                _vector(row['joint_vel'], len(joint_names))
                parts = {name: _vector(row['foot_velocity_contributions'][name])
                         for name in ('hip', 'knee', 'ankle', 'base')}
            except (KeyError, TypeError, ValueError):
                continue
            valid_times.append(t)
            duration += overlap
            for name in amounts:
                amounts[name] += max(float(parts[name] @ direction), 0.) * overlap
    mode_coverage = bool(window_end is not None and window_end >= .15
                         and duration >= .15 * .95 - 1e-9
                         and valid_times and np.all(np.diff(valid_times) <= .0075 + 1e-9))
    total = sum(amounts.values())
    fractions = {name: amounts[name] / total if total > 1e-8 else None for name in amounts}
    mode_pass = bool(mode_coverage and fractions[requested_mode] is not None
                     and fractions[requested_mode] >= .55)

    pre = times <= observed_contact + 1e-9 if observed_contact is not None else np.zeros(len(rows), bool)
    forward_distance = float((roots[pre][-1] - initial) @ direction) if np.any(pre) else 0.
    peak_speed = float(np.max(np.asarray(speeds)[pre])) if np.any(pre) else 0.
    pre_times, pre_intervals = times[pre], intervals[pre]
    approach_coverage = bool(len(pre_times) and pre_times[0] <= pre_intervals[0] + 1e-9
                             and np.all(np.diff(pre_times) <= pre_intervals[1:] * 1.5 + 1e-9))
    runup_pass = bool(approach_coverage and forward_distance >= .7 and peak_speed >= .5)

    standing_start = None
    final_standing = 0.
    missing_recovery_rows = 0
    for i, (row, t) in enumerate(zip(rows, times)):
        stable = False
        if observed_contact is not None and t > observed_contact:
            try:
                _vector(row['whole_body_com_xyz'])
                com_velocity = _vector(row['whole_body_com_velocity'])
                forces = row['support_foot_force_N']
                support = np.array([forces['left'], forces['right']], dtype=float)
                if not np.all(np.isfinite(support)) or np.any(support < 0):
                    raise ValueError('Ground support must be nonnegative')
                stable = bool(roots[i, 2] >= .55 and tilts[i] <= 25. and speeds[i] <= .30
                              and np.linalg.norm(com_velocity[:2]) <= .30
                              and abs(com_velocity[2]) <= .20 and support.sum() >= 80.
                              and nonfoot[i] < 20. and contact_forces[i] == 0.)
            except (KeyError, TypeError, ValueError):
                missing_recovery_rows += 1
        if i and times[i] - times[i - 1] > intervals[i] * 1.5 + 1e-9:
            standing_start = None
        if stable:
            if standing_start is None:
                standing_start = t
            final_standing = t - standing_start
        else:
            standing_start = None
            final_standing = 0.
    recovered = bool(final_standing >= 1. - 1e-9 and not failed and not nonfoot_seen)
    passed = bool(contiguous and runup_pass and mode_pass and recovered)
    return dict(
        requested_mode=requested_mode, motion_sequence_pass=passed,
        trace_coverage_complete=contiguous,
        contact=dict(observed=observed_contact is not None, observed_contact_time_s=observed_contact,
                     supplied_contact_time_s=contact_time_s),
        runup=dict(passed=runup_pass, forward_displacement_m=forward_distance,
                   peak_root_speed_xy_mps=peak_speed, coverage_complete=approach_coverage,
                   interpretation='Short approach only; running flight phase is not established'),
        mode_evidence=dict(passed=mode_pass, coverage_complete=mode_coverage,
            window_start_s=window_start, window_end_s=window_end, observed_duration_s=float(duration),
            hip_positive_displacement_m=amounts['hip'], knee_positive_displacement_m=amounts['knee'],
            hip_fraction=fractions['hip'], knee_fraction=fractions['knee'],
            interpretation='Positive forward foot-velocity contributions; not energy or ball-work shares'),
        recovery=dict(passed=recovered, final_standing_seconds=float(final_standing),
            terminated_or_fallen=failed, nonfoot_contact_seen=nonfoot_seen,
            missing_evidence_rows=missing_recovery_rows,
            ground_support_source='support_foot_force_N, ground-filtered by the producer',
            nonfoot_gate_source='nonfoot_contact_N; a conservative net-contact gate unless ground-filtered'))
