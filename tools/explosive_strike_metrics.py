"""Offline event-aligned robot diagnostics, without a naturalness or transfer score.

Times identify post-step observations. Integrals use clipped rectangular physics
bins (t - dt, t]; preparatory measurements exclude the entire first contact bin.
Joint signs are the robot's recorded joint coordinates, not anatomical labels.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from strike_baseline_metrics import task_outcome, validated_episode, validated_rows


EPS = 1e-9
SEGMENTS = ('pelvis', 'thorax', 'left_upper_arm', 'right_upper_arm')
FORCE_CERTIFICATE = dict(includes_normal=True, includes_friction=True, force_on='robot',
                         frame='world', ground_filtered=True, coverage_complete=True,
                         sign_probe_verified=True)


def unavailable(reason, *evidence):
    return dict(status='unavailable', reason=reason, evidence_needed=list(evidence))


def _array(rows, key, width):
    """Optional evidence must be complete, finite, and in the declared order."""
    try:
        value = np.asarray([r[key] for r in rows], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    return value if value.shape == (len(rows), width) and np.isfinite(value).all() else None


def _direction(direction):
    value = np.asarray(direction, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or abs(value[2]) > EPS:
        raise ValueError('Finite horizontal world direction required')
    norm = np.linalg.norm(value)
    if norm < EPS:
        raise ValueError('Nonzero shot direction required')
    return value / norm


def _window(rows, contact_s, window_s):
    if not math.isfinite(contact_s) or contact_s <= 0 or not math.isfinite(window_s) or window_s <= 0:
        raise ValueError('Positive finite contact time and window required')
    start = max(0., contact_s - window_s)
    samples = [r for r in rows if start + EPS < r['time_s'] < contact_s - EPS]
    return start, samples, _weights(samples, start)


def _weights(rows, start):
    return np.asarray([max(0., r['time_s'] - max(start, r['time_s'] - r['physics_dt_s']))
                       for r in rows], dtype=float)


def signed_extrema(times, values, contact_s):
    """Retain both signs, all exact global ties, and every unsmoothed local peak.

    A flat peak is one local event, represented by its earliest sample and full
    plateau span. A maximum at a window edge is marked, since its true peak may
    lie outside the observed window. No event time is invented for an absent sign.
    """
    t, v = np.asarray(times, float), np.asarray(values, float)
    if (t.ndim != 1 or not len(t) or v.shape != t.shape or not np.isfinite(t).all()
            or not np.isfinite(v).all() or np.any(np.diff(t) <= 0) or not math.isfinite(contact_s)):
        raise ValueError('Finite increasing times and paired scalar values required')

    def peak(indices):
        i = int(indices[0])
        return dict(value=float(v[i]), time_s=float(t[i]), time_to_contact_s=float(t[i] - contact_s),
                    times_s=t[indices].tolist(), values=v[indices].tolist(), tie_count=len(indices),
                    window_boundary=bool(i == 0 or indices[-1] == len(t) - 1))

    positive = peak(np.flatnonzero(v == v.max())) if v.max() > 0 else None
    negative = peak(np.flatnonzero(v == v.min())) if v.min() < 0 else None
    absolute = peak(np.flatnonzero(abs(v) == abs(v).max()))
    positive_local, negative_local = [], []
    i = 0
    while i < len(v):
        end = i
        while end + 1 < len(v) and v[end + 1] == v[i]:
            end += 1
        is_max = (i == 0 or v[i] > v[i - 1]) and (end == len(v) - 1 or v[end] > v[end + 1])
        is_min = (i == 0 or v[i] < v[i - 1]) and (end == len(v) - 1 or v[end] < v[end + 1])
        if (v[i] > 0 and is_max) or (v[i] < 0 and is_min):
            event = peak(np.arange(i, end + 1))
            event.update(plateau_start_s=float(t[i]), plateau_end_s=float(t[end]))
            (positive_local if v[i] > 0 else negative_local).append(event)
        i = end + 1
    return dict(positive=positive, negative=negative, absolute=absolute,
                positive_local_extrema=positive_local, negative_local_extrema=negative_local,
                method='Unsmoothed sample extrema; exact ties retained; earliest tie represents the event')


def joint_dynamics(rows, names, contact_s, window_s=.30):
    rows = validated_rows(rows)
    if not names or len(set(names)) != len(names):
        raise ValueError('Unique ordered joint names required')
    start, samples, weights = _window(rows, contact_s, window_s)
    if not samples:
        return unavailable('No precontact samples in the requested window', 'joint_vel before measured contact')
    t = [r['time_s'] for r in samples]
    velocity = _array(samples, 'joint_vel', len(names))
    torque = _array(samples, 'joint_torque_estimate', len(names))
    joints = {}
    for j, name in enumerate(names):
        entry = dict(velocity=unavailable('Incomplete or invalid measured joint velocity', 'joint_vel in joint_names order'),
                     power=unavailable('Incomplete measured velocity or estimated torque',
                                       'joint_torque_estimate and joint_vel in joint_names order'))
        if velocity is not None:
            entry['velocity'] = dict(status='available', units='rad/s',
                                     peaks=signed_extrema(t, velocity[:, j], contact_s))
        if velocity is not None and torque is not None:
            power = velocity[:, j] * torque[:, j]
            entry['power'] = dict(status='available', units='W', peaks=signed_extrema(t, power, contact_s),
                                  positive_work_J=float(np.maximum(power, 0) @ weights),
                                  negative_work_J=float(np.minimum(power, 0) @ weights))
        joints[name] = entry
    return dict(status='available', window_start_s=start, window_end_s=contact_s,
                last_sample_s=t[-1], measured_duration_s=float(weights.sum()),
                coverage_fraction=float(weights.sum() / (contact_s - start)), joints=joints,
                torque_definition='Estimated clipped implicit-PD torque times measured joint velocity; not hardware power',
                sign_definition='Recorded robot joint coordinates; positive and negative are not inferred anatomical actions',
                integration='Post-step rectangular bins, clipped at window start; first contact bin excluded')


def kinematic_contributions(rows, contact_s, direction, window_s=.30):
    """Keep the existing Jacobian foot-velocity decomposition separate from power."""
    rows, direction = validated_rows(rows), _direction(direction)
    start, samples, weights = _window(rows, contact_s, window_s)
    if not samples:
        return unavailable('No precontact kinematic samples', 'foot_velocity_contributions before contact')
    parts, vectors = {}, []
    for name in ('hip', 'knee', 'ankle', 'base'):
        try:
            v = np.asarray([r['foot_velocity_contributions'][name] for r in samples], float)
        except (KeyError, TypeError, ValueError):
            v = np.empty((0, 3))
        if v.shape != (len(samples), 3) or not np.isfinite(v).all():
            return unavailable('Missing or invalid kinematic contribution', 'foot_velocity_contributions.' + name)
        vectors.append(v)
        forward = v @ direction
        parts[name] = dict(signed_integral_m=float(forward @ weights),
                           positive_integral_m=float(np.maximum(forward, 0) @ weights),
                           negative_integral_m=float(np.minimum(forward, 0) @ weights))
    point = _array(samples, 'right_foot_point_velocity', 3)
    closure = float(np.linalg.norm(sum(vectors) - point, axis=1).max()) if point is not None else None
    return dict(status='available', window_start_s=start, window_end_s=contact_s,
                measured_duration_s=float(weights.sum()), parts=parts,
                point_velocity_closure_max_error_mps=closure,
                definition='Integral of Jacobian joint-group foot-velocity components and floating-base remainder projected on shot direction; not energy or ball-work shares')


def support_braking(rows, contact_s, direction, window_s=.30, *, metadata=None, side='left', threshold_N=80.):
    """COM deceleration during final loaded stance; optional certified total GRF.

    The legacy support scalar is clipped vertical ground-contact force. It gates
    the stance interval only and is never used to infer horizontal force/work.
    """
    rows, direction = validated_rows(rows), _direction(direction)
    start, _, _ = _window(rows, contact_s, window_s)
    if not math.isfinite(threshold_N) or threshold_N <= 0:
        raise ValueError('Positive finite support threshold required')
    pre = [r for r in rows if r['time_s'] < contact_s - EPS]
    try:
        vertical = np.asarray([r['support_foot_force_N'][side] for r in pre], float)
    except (KeyError, TypeError, ValueError):
        return unavailable('Missing support-force evidence', 'support_foot_force_N.' + side)
    if not len(pre) or not np.isfinite(vertical).all() or np.any(vertical < 0):
        return unavailable('Invalid or absent support-force evidence', 'finite nonnegative ground-filtered vertical support')
    if vertical[-1] < threshold_N:
        return unavailable('The last precontact sample is not loaded at the declared threshold',
                           'a contiguous loaded support episode reaching the last precontact sample')
    first = len(pre) - 1
    while first > 0 and vertical[first - 1] >= threshold_N:
        first -= 1
    onset = pre[first]['time_s']
    lower = max(0., onset - pre[first]['physics_dt_s'])
    samples = [r for r in pre[first:] if r['time_s'] > start + EPS]
    weights = _weights(samples, max(start, lower))
    if not samples:
        return unavailable('No loaded samples in the preparation window', 'loaded precontact support samples')
    value = dict(status='available', support_side=side, vertical_force_threshold_N=threshold_N,
                 first_loaded_sample_s=onset, support_onset_bracket_s=[lower, onset],
                 onset_left_censored=bool(first == 0), last_precontact_sample_s=samples[-1]['time_s'],
                 window_start_s=max(start, lower), window_end_s=contact_s,
                 precontact_loaded_duration_s=float(weights.sum()),
                 first_loaded_sample_time_to_contact_s=onset - contact_s,
                 interpretation='Loaded-stance kinematics are a motion proxy; the vertical gate is not horizontal GRF')
    com = _array(samples, 'whole_body_com_velocity', 3)
    if com is None or len(samples) < 2:
        value['com_kinematics'] = unavailable('Incomplete COM velocity or fewer than two loaded samples',
                                             'whole_body_com_velocity in world axes at every loaded sample')
    else:
        t = np.asarray([r['time_s'] for r in samples])
        forward = com @ direction
        delta = np.diff(forward)
        acceleration = delta / np.diff(t)
        i = int(np.argmin(acceleration))
        value['com_kinematics'] = dict(status='available', first_observed_time_s=float(t[0]),
            last_observed_time_s=float(t[-1]), first_forward_velocity_mps=float(forward[0]),
            last_forward_velocity_mps=float(forward[-1]), delta_forward_velocity_mps=float(forward[-1] - forward[0]),
            accumulated_speed_decrease_mps=float(-np.minimum(delta, 0).sum()),
            accumulated_speed_increase_mps=float(np.maximum(delta, 0).sum()),
            peak_deceleration_mps2=float(max(0., -acceleration[i])),
            peak_deceleration_interval_s=[float(t[i]), float(t[i + 1])] if acceleration[i] < 0 else None,
            definition='Finite difference of measured whole-body COM velocity projected on the fixed shot direction; no force attribution')
    force = unavailable('Total horizontal foot-ground force is absent or uncertified',
                        'support_foot_ground_force_w_N with normal and friction components',
                        'dynamics_telemetry.support_ground_force certifying world force ON robot, ground filtering, coverage and sign probe')
    certificate = (metadata or {}).get('support_ground_force', {})
    if all(certificate.get(k) == v for k, v in FORCE_CERTIFICATE.items()):
        try:
            vectors = np.asarray([r['support_foot_ground_force_w_N'][side] for r in samples], float)
        except (KeyError, TypeError, ValueError):
            vectors = np.empty((0, 3))
        if vectors.shape == (len(samples), 3) and np.isfinite(vectors).all():
            forward = vectors @ direction
            force = dict(status='available', signed_forward_impulse_Ns=float(forward @ weights),
                         braking_impulse_Ns=float(np.minimum(forward, 0) @ weights),
                         propulsive_impulse_Ns=float(np.maximum(forward, 0) @ weights),
                         peak_braking_force_N=float(min(0., forward.min())),
                         definition='Certified total ground force ON support foot; negative opposes shot direction; clipped post-step bins')
    value['horizontal_ground_reaction'] = force
    return value


def contact_metrics(rows, direction):
    rows, direction = validated_rows(rows), _direction(direction)
    episodes = []
    i = 0
    while i < len(rows):
        if rows[i]['contact_force_N'] <= 0:
            i += 1
            continue
        start = i
        while i < len(rows) and rows[i]['contact_force_N'] > 0:
            i += 1
        episodes.append((start, i))
    if not episodes:
        return unavailable('No measured foot-ball contact', 'positive physical foot-ball normal force')
    start, end = episodes[0]
    force_rows = rows[start:end]
    feet = sorted({event.get('foot', 'unknown') for r in force_rows
                   for event in (r.get('contacts') or [{'foot': r.get('foot', 'unknown')}])})
    value = dict(status='available', first_force_s=rows[start]['time_s'], last_force_s=rows[end - 1]['time_s'],
                 contact_onset_bracket_s=[rows[start]['time_s'] - rows[start]['physics_dt_s'], rows[start]['time_s']],
                 force_duration_s=sum(r['physics_dt_s'] for r in force_rows),
                 normal_impulse_Ns=sum(r['contact_force_N'] * r['physics_dt_s'] for r in force_rows),
                 peak_normal_force_N=max(r['contact_force_N'] for r in force_rows),
                 separation_sample_s=rows[end]['time_s'] if end < len(rows) else None,
                 episode_count=len(episodes), observed_contact_feet=feet,
                 all_episodes_normal_impulse_Ns=sum(r['contact_force_N'] * r['physics_dt_s'] for r in rows),
                 normal_impulse_definition='Integral of actual sampled foot-ball contact-point normal-force magnitudes in first contiguous episode; not net vector impulse',
                 preimpact_foot=unavailable('No complete right-foot observation immediately before contact',
                                           'right_foot_point_velocity at the last noncontact sample and right-foot contact'),
                 outgoing_ball=unavailable('No complete first separation observation', 'ball_velocity at the first zero-force sample'),
                 actual_contact_point_velocity=unavailable('Fixed reference-point velocity is not collision-point velocity',
                                                           'collision-point world position and measured foot rigid-body twist'))
    if start and feet == ['right']:
        v = _array([rows[start - 1]], 'right_foot_point_velocity', 3)
        if v is not None:
            value['preimpact_foot'] = dict(status='available', time_s=rows[start - 1]['time_s'],
                time_to_contact_s=rows[start - 1]['time_s'] - rows[start]['time_s'],
                velocity_w_mps=v[0].tolist(), speed_mps=float(np.linalg.norm(v[0])),
                forward_speed_mps=float(v[0] @ direction), definition='Measured velocity of fixed right-foot reference point')
    if end < len(rows):
        v = _array([rows[end]], 'ball_velocity', 3)
        if v is not None:
            value['outgoing_ball'] = dict(status='available', time_s=rows[end]['time_s'],
                velocity_w_mps=v[0].tolist(), speed_xy_mps=float(np.linalg.norm(v[0, :2])),
                speed_3d_mps=float(np.linalg.norm(v[0])))
    return value


def follow_through(rows, contact, direction, window_s=.30):
    rows, direction = validated_rows(rows), _direction(direction)
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError('Positive finite follow-through window required')
    start = contact.get('separation_sample_s')
    if start is None:
        return unavailable('No observed end of the first contact episode', 'first zero-force sample after contact')
    next_contact = next((r['time_s'] for r in rows if r['time_s'] > start + EPS and r['contact_force_N'] > 0), None)
    end = min(start + window_s, next_contact - EPS if next_contact is not None else math.inf)
    samples = [r for r in rows if start - EPS <= r['time_s'] <= end + EPS
               and (next_contact is None or r['time_s'] < next_contact - EPS)]
    velocity = _array(samples, 'right_foot_point_velocity', 3)
    if velocity is None or len(samples) < 2 or contact.get('observed_contact_feet') != ['right']:
        return unavailable('Incomplete right-foot follow-through observations', 'right_foot_point_velocity at and after right-foot separation')
    weights = _weights(samples, start)
    forward = velocity @ direction
    continuous = 0.
    if forward[0] > 0:
        for i in range(1, len(samples)):
            if forward[i] <= 0:
                break
            continuous += weights[i]
    reverse = next((r['time_s'] for r, speed in zip(samples, forward) if speed <= 0), None)
    initial = contact.get('preimpact_foot', {})
    alignment = None
    if initial.get('status') == 'available' and initial['speed_mps'] > EPS and np.linalg.norm(velocity[0]) > EPS:
        alignment = float(np.clip(np.asarray(initial['velocity_w_mps']) @ velocity[0]
                                  / (initial['speed_mps'] * np.linalg.norm(velocity[0])), -1., 1.))
    return dict(status='available', window_start_s=start, requested_window_end_s=start + window_s,
                last_observed_time_s=samples[-1]['time_s'], measured_duration_s=float(weights.sum()),
                coverage_fraction=float(weights.sum() / window_s), coverage_complete=bool(weights.sum() >= window_s - EPS),
                censored_by_recontact=bool(next_contact is not None and next_contact <= start + window_s + EPS),
                separation_foot_speed_mps=float(np.linalg.norm(velocity[0])),
                separation_forward_speed_mps=float(forward[0]),
                preimpact_to_separation_direction_cosine=alignment,
                signed_forward_displacement_estimate_m=float(forward @ weights),
                reference_point_path_length_estimate_m=float(np.linalg.norm(velocity, axis=1) @ weights),
                continuous_forward_duration_s=float(continuous), first_nonforward_sample_s=reverse,
                positive_forward_duration_s=float(weights[forward > 0].sum()),
                definition='Unsmoothed fixed-point velocities after first zero-force sample; rectangular integration; reversal is descriptive, not a failure score')


def _heading_series(times, quaternions):
    t, q = np.asarray(times, float), np.asarray(quaternions, float)
    if (t.ndim != 1 or len(t) < 2 or q.shape != (len(t), 4) or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0) or not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=1) < EPS)):
        raise ValueError('Two or more finite ordered orientations required')
    heading = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()[:, :2, 0]
    if np.any(np.linalg.norm(heading, axis=1) < 1e-6):
        raise ValueError('Body forward axis has an undefined horizontal heading')
    return t, np.unwrap(np.arctan2(heading[:, 1], heading[:, 0]))


def segment_sequencing(result, rows, contact_s, window_s=.30):
    rows = validated_rows(rows)
    start, samples, _ = _window(rows, contact_s, window_s)
    segments, series = {}, {}
    for name in SEGMENTS:
        evidence = unavailable('No complete measured world segment orientation sequence',
                               'segment_quaternions_wxyz.' + name + ' at each precontact sample')
        source = 'World segment orientation at physics sampling'
        try:
            times = [r['time_s'] for r in samples]
            q = [r['segment_quaternions_wxyz'][name] for r in samples]
        except (KeyError, TypeError):
            times, q = [], []
            has_segment = any(isinstance(r.get('segment_quaternions_wxyz'), dict)
                              and name in r['segment_quaternions_wxyz'] for r in samples)
            if name == 'pelvis' and not has_segment:
                poses = [p for p in (result.get('kinematics') or [[]])[0]
                         if start - EPS <= p['time_s'] < contact_s - EPS]
                try:
                    times = [p['time_s'] for p in poses]
                    q = [p['root_pose'][3:] for p in poses]
                except (KeyError, TypeError):
                    times, q = [], []
                source = 'Recorded root/pelvis world orientation at policy sampling (normally 20 ms)'
        try:
            t, yaw = _heading_series(times, q)
        except (ValueError, TypeError):
            segments[name] = evidence
            continue
        rate = np.diff(yaw) / np.diff(t)
        midpoint = (t[1:] + t[:-1]) / 2
        evidence = dict(status='available', source=source, sample_count=len(t),
            first_observed_time_s=float(t[0]), last_observed_time_s=float(t[-1]),
            max_sample_interval_s=float(np.diff(t).max()),
            net_projected_yaw_deg=float(np.degrees(yaw[-1] - yaw[0])),
            projected_yaw_range_deg=float(np.degrees(np.ptp(yaw))),
            projected_yaw_rate_peaks_rad_s=signed_extrema(midpoint, rate, contact_s),
            rate_definition='Finite difference of unwrapped projected body-X heading, timed at interval midpoint; not world-z angular velocity')
        omega = None
        try:
            omega = np.asarray([r['segment_angular_velocity_w_rad_s'][name] for r in samples], float)
        except (KeyError, TypeError, ValueError):
            pass
        if omega is not None and omega.shape == (len(samples), 3) and np.isfinite(omega).all():
            evidence['world_z_angular_velocity_peaks_rad_s'] = signed_extrema(
                [r['time_s'] for r in samples], omega[:, 2], contact_s)
        else:
            evidence['world_z_angular_velocity'] = unavailable('World angular velocity not recorded completely',
                                                               'segment_angular_velocity_w_rad_s.' + name)
        segments[name], series[name] = evidence, (t, yaw)
    relative = unavailable('Synchronized pelvis and thorax headings are unavailable',
                           'measured world pelvis and thorax orientations with common timestamps')
    if 'pelvis' in series and 'thorax' in series:
        pelvis = {round(float(t), 9): y for t, y in zip(*series['pelvis'])}
        thorax = {round(float(t), 9): y for t, y in zip(*series['thorax'])}
        common = sorted(set(pelvis) & set(thorax))
        if len(common) >= 2:
            yaw = np.unwrap([math.atan2(math.sin(thorax[t] - pelvis[t]), math.cos(thorax[t] - pelvis[t])) for t in common])
            relative = dict(status='available', sample_count=len(common),
                            net_change_deg=float(np.degrees(yaw[-1] - yaw[0])),
                            range_deg=float(np.degrees(np.ptp(yaw))),
                            definition='Difference of projected world headings; not anatomical axial torsion')
    order = sorted([dict(segment=name, time_to_contact_s=value['projected_yaw_rate_peaks_rad_s']['absolute']['time_to_contact_s'],
                         signed_rate_rad_s=value['projected_yaw_rate_peaks_rad_s']['absolute']['value'],
                         observation_interval_s=value['max_sample_interval_s'])
                    for name, value in segments.items() if value['status'] == 'available'],
                   key=lambda item: item['time_to_contact_s'])
    return dict(segments=segments, pelvis_thorax_relative_heading=relative, observed_absolute_peak_order=order,
                interpretation='Observed kinematic timing only; missing segments prevent full sequencing claims; no energy-transfer attribution')


def _force_component(vectors, weights, times, direction, body_weight_N):
    norms = np.linalg.norm(vectors, axis=1)
    peak, vertical = int(np.argmax(norms)), int(np.argmax(vectors[:, 2]))
    forward = vectors @ direction
    impulse = vectors.T @ weights
    return dict(impulse_w_Ns=impulse.tolist(), signed_forward_impulse_Ns=float(forward @ weights),
                braking_impulse_Ns=float(np.minimum(forward, 0) @ weights),
                propulsive_impulse_Ns=float(np.maximum(forward, 0) @ weights),
                mean_force_w_N=(impulse / weights.sum()).tolist(),
                peak_norm_force_N=float(norms[peak]), peak_norm_sample_s=float(times[peak]),
                force_w_N_at_peak_norm=vectors[peak].tolist(),
                peak_positive_vertical_force_N=float(max(0., vectors[vertical, 2])),
                peak_vertical_sample_s=float(times[vertical]) if vectors[vertical, 2] > 0 else None,
                peak_norm_body_weights=float(norms[peak] / body_weight_N) if body_weight_N else None)


def _landing_forces(samples, weights, direction, metadata, gravity_w):
    certificate = metadata.get('support_ground_force', {})
    needs = ['certified normal + friction world forces ON each foot with complete ground-filtered coverage']
    if not all(certificate.get(k) == v for k, v in FORCE_CERTIFICATE.items()):
        return unavailable('Ground-force semantics are absent or uncertified', *needs), None
    if any(r.get('support_contact_capacity_exhausted') for r in samples):
        return unavailable('A sampled contact buffer reached capacity despite global coverage metadata', *needs), None
    mass = metadata.get('robot_total_mass_kg')
    body_weight = (float(mass * np.linalg.norm(gravity_w))
                   if isinstance(mass, (int, float)) and math.isfinite(mass) and mass > 0 else None)
    times = np.asarray([r['time_s'] for r in samples])
    feet, foot_arrays = {}, {}
    fields = dict(normal='support_foot_normal_force_w_N', friction='support_foot_friction_force_w_N',
                  total='support_foot_ground_force_w_N')
    for foot in ('left', 'right'):
        arrays = {}
        for part, field in fields.items():
            try:
                v = np.asarray([r[field][foot] for r in samples], float)
            except (KeyError, TypeError, ValueError):
                v = np.empty((0, 3))
            if v.shape != (len(samples), 3) or not np.isfinite(v).all():
                break
            arrays[part] = v
        if len(arrays) != 3:
            feet[foot] = unavailable('Missing or invalid per-foot force components', *[f + '.' + foot for f in fields.values()])
            continue
        closure = arrays['total'] - arrays['normal'] - arrays['friction']
        if not np.allclose(arrays['total'], arrays['normal'] + arrays['friction'], atol=1e-6, rtol=1e-6):
            feet[foot] = unavailable('Recorded total force does not equal normal plus friction', 'consistent per-foot force component arrays')
            continue
        feet[foot] = dict(status='available', components={k: _force_component(v, weights, times, direction, body_weight)
                                                        for k, v in arrays.items()},
                          component_closure_max_error_N=float(np.linalg.norm(closure, axis=1).max()))
        foot_arrays[foot] = arrays
    combined = unavailable('Both feet must have complete consistent component telemetry', *needs)
    total = None
    if len(foot_arrays) == 2:
        arrays = {part: foot_arrays['left'][part] + foot_arrays['right'][part] for part in fields}
        combined = dict(status='available', components={k: _force_component(v, weights, times, direction, body_weight)
                                                        for k, v in arrays.items()})
        total = arrays['total']
    return dict(status='available', feet=feet, combined=combined,
                complete_bilateral_evidence=len(foot_arrays) == 2, body_weight_N=body_weight,
                definition='Recorded foot-ground normal and friction components, plus their sum; these do not by themselves certify whole-robot external momentum closure'), total


def _landing_segments(samples):
    times = [r['time_s'] for r in samples]
    segments, headings, orientations = {}, {}, {}
    for name in SEGMENTS:
        missing = unavailable('Incomplete window segment orientation observations', 'segment_quaternions_wxyz.' + name)
        try:
            q = np.asarray([r['segment_quaternions_wxyz'][name] for r in samples], float)
        except (KeyError, TypeError, ValueError):
            q = np.empty((0, 4))
        if (len(samples) < 2 or q.shape != (len(samples), 4) or not np.isfinite(q).all()
                or np.any(np.linalg.norm(q, axis=1) < EPS)):
            segments[name] = missing
            continue
        rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])
        changes = rotation * rotation[0].inv()
        angles = np.degrees(changes.magnitude())
        tilt = np.degrees(np.arccos(np.clip(rotation.as_matrix()[:, 2, 2], -1., 1.)))
        segments[name] = dict(status='available', first_observed_time_s=times[0], last_observed_time_s=times[-1],
            net_rotation_angle_deg=float(angles[-1]), net_rotation_vector_w_deg=np.degrees(changes[-1].as_rotvec()).tolist(),
            max_rotation_from_window_start_deg=float(angles.max()),
            observed_rotation_path_deg=float(np.degrees((rotation[1:] * rotation[:-1].inv()).magnitude()).sum()),
            initial_body_z_tilt_deg=float(tilt[0]), final_body_z_tilt_deg=float(tilt[-1]), max_body_z_tilt_deg=float(tilt.max()))
        orientations[name] = rotation
        try:
            _, yaw = _heading_series(times, q)
            headings[name] = yaw
            segments[name].update(net_projected_yaw_deg=float(np.degrees(yaw[-1] - yaw[0])),
                                   projected_yaw_range_deg=float(np.degrees(np.ptp(yaw))))
        except ValueError:
            segments[name]['projected_heading'] = unavailable('Projected heading is singular in this window', 'nonvertical projected body-X axis')
        try:
            omega = np.asarray([r['segment_angular_velocity_w_rad_s'][name] for r in samples], float)
        except (KeyError, TypeError, ValueError):
            omega = np.empty((0, 3))
        if omega.shape == (len(samples), 3) and np.isfinite(omega).all():
            k = int(np.argmax(np.linalg.norm(omega, axis=1)))
            segments[name].update(peak_angular_speed_rad_s=float(np.linalg.norm(omega[k])),
                                   peak_angular_speed_time_s=times[k], angular_velocity_w_rad_s_at_peak=omega[k].tolist())
    relative = unavailable('Synchronized pelvis/thorax window orientations are unavailable',
                           'pelvis and thorax world orientations at common times')
    if 'pelvis' in orientations and 'thorax' in orientations:
        rotation = orientations['pelvis'].inv() * orientations['thorax']
        relative = dict(status='available', net_relative_rotation_angle_deg=float(np.degrees((rotation[-1] * rotation[0].inv()).magnitude())))
        if 'pelvis' in headings and 'thorax' in headings:
            yaw = np.unwrap(headings['thorax'] - headings['pelvis'])
            relative.update(net_change_deg=float(np.degrees(yaw[-1] - yaw[0])), range_deg=float(np.degrees(np.ptp(yaw))))
    return segments, relative


def _joint_limit_samples(samples, names, limits):
    if not names or len(set(names)) != len(names):
        return unavailable('Missing unique joint order', 'joint_names and matching joint_velocity_limits')
    try:
        limits = np.asarray(limits, float)
    except (TypeError, ValueError):
        limits = np.empty(0)
    if limits.shape == (1, len(names)):
        limits = limits[0]
    velocity = _array(samples, 'joint_vel', len(names))
    if limits.shape != (len(names),) or not np.isfinite(limits).all() or np.any(limits <= 0) or velocity is None:
        return unavailable('Incomplete velocity or positive joint-limit evidence', 'joint_vel and joint_velocity_limits in joint_names order')
    ratios = abs(velocity) / limits
    observed = []
    for r, speed, ratio in zip(samples, velocity, ratios):
        k = int(np.argmax(ratio))
        observed.append(dict(time_s=r['time_s'], max_speed_to_limit_ratio=float(ratio[k]), maximum_ratio_joint=names[k],
            overspeed_joints=[dict(joint=names[j], signed_velocity_rad_s=float(speed[j]), limit_rad_s=float(limits[j]), ratio=float(ratio[j]))
                              for j in range(len(names)) if abs(speed[j]) > limits[j] + 1e-6]))
    return dict(status='available', max_speed_to_limit_ratio=float(ratios.max()),
                all_overspeed_joints=[name for j, name in enumerate(names) if np.any(abs(velocity[:, j]) > limits[j] + 1e-6)],
                samples=observed, interpretation='Observed qdot against recorded original limits; exceeding a limit does not identify a solver correction or its cause')


def _landing_momentum(rows, samples, weights, total_force, metadata, start_s, end_s, gravity_w):
    need = ['measured COM velocity at exact window endpoints and each intervening physics boundary',
            'robot_total_mass_kg and complete consistent bilateral force components']
    missing = unavailable('The partial momentum balance lacks exact endpoint or force/mass evidence', *need)
    mass = metadata.get('robot_total_mass_kg')
    states = {round(r['time_s'], 9): r for r in rows}
    start, end = states.get(round(start_s, 9)), states.get(round(end_s, 9))
    if (start is None or end is None or total_force is None or not isinstance(mass, (int, float))
            or not math.isfinite(mass) or mass <= 0 or abs(weights.sum() - (end_s - start_s)) > 1e-7):
        return missing
    boundaries = [start] + samples
    velocity = _array(boundaries, 'whole_body_com_velocity', 3)
    if velocity is None or abs(samples[-1]['time_s'] - end_s) > EPS:
        return missing
    intervals = np.diff([r['time_s'] for r in boundaries])
    if np.any(abs(intervals - weights) > 1e-7):
        return missing
    residuals = mass * np.diff(velocity, axis=0) - total_force * weights[:, None] - mass * gravity_w * weights[:, None]
    k = int(np.argmax(np.linalg.norm(residuals, axis=1)))
    bins = [dict(interval_s=[boundaries[i]['time_s'], r['time_s']],
                 unaccounted_impulse_w_Ns=residuals[i].tolist(), residual_norm_Ns=float(np.linalg.norm(residuals[i])))
            for i, r in enumerate(samples)]
    impulse = total_force.T @ weights
    momentum = mass * (velocity[-1] - velocity[0])
    gravity = mass * gravity_w * weights.sum()
    return dict(status='available', robot_total_mass_kg=mass, assumed_gravity_w_mps2=gravity_w.tolist(),
                measured_com_velocity_start_w_mps=velocity[0].tolist(), measured_com_velocity_end_w_mps=velocity[-1].tolist(),
                measured_com_momentum_change_w_Ns=momentum.tolist(), recorded_feet_ground_impulse_w_Ns=impulse.tolist(),
                gravity_impulse_w_Ns=gravity.tolist(), unaccounted_impulse_w_Ns=(momentum - impulse - gravity).tolist(),
                residual_norm_Ns=float(np.linalg.norm(momentum - impulse - gravity)),
                max_bin_residual_norm_Ns=bins[k]['residual_norm_Ns'], max_residual_bin_s=bins[k]['interval_s'], bins=bins,
                interpretation='Diagnostic mΔv_COM minus recorded bilateral foot-ground and assumed gravity impulses; unobserved external or constraint effects are not attributed or corrected')


def landing_window_dynamics(rows, metadata, start_s, end_s, direction, *, joint_names=None,
                            joint_velocity_limits=None, gravity_w=(0., 0., -9.81)):
    """Analyze an absolute landing interval without changing any physical state.

    Forces use overlap of post-step rectangular bins with [start, end]. Segment
    and velocity changes use actual observed timestamps, never interpolated poses.
    Momentum comparison is available only at exact observed window endpoints.
    """
    rows, direction = validated_rows(rows), _direction(direction)
    gravity = np.asarray(gravity_w, float)
    if (not all(math.isfinite(v) for v in (start_s, end_s)) or start_s < 0 or end_s <= start_s
            or gravity.shape != (3,) or not np.isfinite(gravity).all()):
        raise ValueError('Finite increasing nonnegative window and world gravity required')
    weighted = [(r, max(0., min(r['time_s'], end_s) - max(r['time_s'] - r['physics_dt_s'], start_s))) for r in rows]
    samples = [r for r, w in weighted if w > EPS]
    weights = np.asarray([w for _, w in weighted if w > EPS])
    if not samples:
        return unavailable('No physical bins overlap the requested window', 'physical observations covering the landing interval')
    observed = [r for r in rows if start_s - EPS <= r['time_s'] <= end_s + EPS]
    forces, total_force = _landing_forces(samples, weights, direction, metadata, gravity)
    endpoint_peak = unavailable('Incomplete certified force observations at window sample times',
                                 'bilateral normal, friction and total force at endpoint-inclusive sample times')
    if forces.get('complete_bilateral_evidence') and observed and not any(r.get('support_contact_capacity_exhausted') for r in observed):
        try:
            arrays = {part: np.asarray([np.sum([r[f'support_foot_{field}_force_w_N'][side] for side in ('left', 'right')], axis=0)
                                       for r in observed], float)
                      for part, field in [('normal', 'normal'), ('friction', 'friction'), ('total', 'ground')]}
            valid = all(v.shape == (len(observed), 3) and np.isfinite(v).all() for v in arrays.values())
            if valid and np.allclose(arrays['total'], arrays['normal'] + arrays['friction'], atol=1e-6, rtol=1e-6):
                k = int(np.argmax(np.linalg.norm(arrays['total'], axis=1)))
                norm = float(np.linalg.norm(arrays['total'][k]))
                endpoint_peak = dict(status='available', time_s=observed[k]['time_s'], peak_norm_force_N=norm,
                    force_w_N=arrays['total'][k].tolist(), normal_w_N=arrays['normal'][k].tolist(),
                    friction_w_N=arrays['friction'][k].tolist(),
                    peak_norm_body_weights=norm / forces['body_weight_N'] if forces['body_weight_N'] else None,
                    definition='Peak over observed samples including both window endpoints; the start sample represents the preceding bin, which is excluded from interval impulse')
        except (KeyError, TypeError, ValueError):
            pass
    segments, relative = _landing_segments(observed)
    foot = unavailable('Incomplete foot reference-point velocity observations', 'right_foot_point_velocity at the window timestamps')
    v = _array(observed, 'right_foot_point_velocity', 3)
    if v is not None and len(observed) >= 2:
        norms, jumps = np.linalg.norm(v, axis=1), np.linalg.norm(np.diff(v, axis=0), axis=1)
        k, j = int(np.argmax(norms)), int(np.argmax(jumps))
        foot = dict(status='available', peak_speed_mps=float(norms[k]), peak_speed_time_s=observed[k]['time_s'],
                    velocity_w_mps_at_peak=v[k].tolist(), largest_velocity_jump_mps=float(jumps[j]),
                    largest_jump_interval_s=[observed[j]['time_s'], observed[j + 1]['time_s']],
                    velocity_before_largest_jump_w_mps=v[j].tolist(), velocity_after_largest_jump_w_mps=v[j + 1].tolist())
    return dict(status='available', requested_window_start_s=start_s, requested_window_end_s=end_s,
                measured_duration_s=float(weights.sum()), coverage_complete=bool(abs(weights.sum() - (end_s - start_s)) < 1e-7),
                coverage_fraction=float(weights.sum() / (end_s - start_s)), direction_w=direction.tolist(),
                force_bin_sample_times_s=[r['time_s'] for r in samples], ground_reaction=forces,
                endpoint_inclusive_total_force_peak=endpoint_peak,
                segments=segments, pelvis_thorax_relative_heading=relative, foot_point_motion=foot,
                joint_speed_limits=_joint_limit_samples(observed, joint_names, joint_velocity_limits),
                momentum_balance=_landing_momentum(rows, samples, weights, total_force, metadata, start_s, end_s, gravity),
                measured_foot_ball_normal_impulse_Ns=float(np.asarray([r['contact_force_N'] for r in samples]) @ weights),
                definition='Absolute-time diagnostic window; original task outcome and acceptance gates remain separate')


def joint_velocity_scaling_hypothesis(row, names, limits, mass_kg, pelvis_inertial_origin_xyz,
                                      observed_residual_w_Ns):
    """Evaluate a published pre-solve scaling mechanism as an unconfirmed hypothesis.

    This predicts a residual only. It never edits contact forces, state, limits,
    momentum measurements, or acceptance. The local solver binary is not checked.
    """
    missing = unavailable('Scaling hypothesis lacks measured root/COM twist or joint-limit evidence',
        'joint_vel, limits, world root velocity, pelvis orientation/angular velocity and whole-body COM pose/velocity')
    if not names or len(set(names)) != len(names) or not math.isfinite(mass_kg) or mass_kg <= 0:
        return missing
    try:
        speed, limits = np.asarray(row['joint_vel'], float), np.asarray(limits, float)
        if limits.shape == (1, len(names)):
            limits = limits[0]
        vectors = np.asarray([row['root_xyz'], row['root_velocity_w'], row['whole_body_com_xyz'],
            row['whole_body_com_velocity'], row['segment_angular_velocity_w_rad_s']['pelvis'],
            pelvis_inertial_origin_xyz, observed_residual_w_Ns], float)
        q = np.asarray(row['segment_quaternions_wxyz']['pelvis'], float)
    except (KeyError, TypeError, ValueError):
        return missing
    if (speed.shape != (len(names),) or limits.shape != speed.shape or np.any(limits <= 0)
            or vectors.shape != (7, 3) or q.shape != (4,) or np.linalg.norm(q) < EPS
            or not all(np.isfinite(a).all() for a in (speed, limits, vectors, q))):
        return missing
    root_pos, root_velocity, com_pos, com_velocity, omega, offset, residual = vectors
    root_com = root_pos + Rotation.from_quat(q[[1, 2, 3, 0]]).apply(offset)
    internal = com_velocity - root_velocity - np.cross(omega, com_pos - root_com)
    ratios = abs(speed) / limits
    gamma = min(1., 1. / float(ratios.max())) if ratios.max() > 0 else 1.
    prediction = mass_kg * (gamma - 1.) * internal
    return dict(status='available', observation_time_s=row['time_s'], gamma=gamma,
        limiting_joint=names[int(np.argmax(ratios))], max_speed_to_limit_ratio=float(ratios.max()),
        internal_com_velocity_w_mps=internal.tolist(), inferred_root_com_position_w_m=root_com.tolist(),
        pelvis_inertial_origin_xyz_m=offset.tolist(), predicted_impulse_w_Ns=prediction.tolist(),
        original_observed_residual_w_Ns=residual.tolist(),
        prediction_residual_error_norm_Ns=float(np.linalg.norm(prediction - residual)),
        local_solver_mechanism_verified=False,
        source_url='https://github.com/NVIDIA-Omniverse/PhysX/pull/534',
        formula='gamma=min(1,min_j(limit_j/abs(qdot_j))); v_internal=v_COM-v_rootCOM-omega_root cross (x_COM-x_rootCOM); predicted J=m*(gamma-1)*v_internal',
        limitations=['Published PR mechanism is not verified in the installed solver binary',
                     'Pelvis local COM is inferred from the digest-matched URDF, not directly queried from PhysX',
                     'root_velocity_w is treated as root-COM linear velocity; matching a residual supports but does not prove causation',
                     'No measured forces, residuals, limits, controller behavior or task outcomes are corrected'])


def summarize_explosive_strike(result, *, expected_duration_s=14., window_s=.30, follow_window_s=.30):
    """Summarize one frozen physical episode; optional evidence never becomes zero."""
    rows = validated_episode(result, expected_duration_s)
    force_index = next((i for i, r in enumerate(rows) if r['contact_force_N'] > 0), None)
    reference = rows[max(0, force_index - 1)] if force_index is not None else rows[0]
    goal, ball = np.asarray(result['goals'][0], float), np.asarray(reference['ball_xyz'], float)
    direction = _direction(np.r_[goal[:2] - ball[:2], 0.])
    contact = contact_metrics(rows, direction)
    sequence = result.get('sequence_records', [{}])[0]
    recovery = sequence.get('recovery')
    value = dict(schema_version=1, outcome=task_outcome(result), contact=contact,
                 direction_w=direction.tolist(), physical_steps=len(rows), completed_physics_seconds=rows[-1]['time_s'],
                 terminal_status_markers=len(result['traces'][0]) - len(rows),
                 foot_reference_point_local=result.get('foot_velocity_point_local'),
                 torque_evidence=result.get('torque_evidence'),
                 precontact_joint_dynamics=None, post_plant_joint_dynamics=None, kinematic_foot_contributions=None,
                 support_braking=None, segment_sequencing=None, follow_through=None,
                 recovery=unavailable('Original recovery evidence is missing', 'sequence_records[0].recovery'),
                 unavailable_attributions=dict(
                     intersegment_energy_transfer=unavailable('Joint power and timing do not identify intersegment energy transfer',
                         'validated inverse dynamics with segment inertias, full kinematics and external wrenches'),
                     human_naturalness=unavailable('No validated cross-species naturalness construct is defined',
                         'a separately validated, task-matched human/robot comparison protocol')))
    if recovery is not None:
        duration = recovery.get('final_standing_seconds')
        stable_start = (rows[-1]['time_s'] - duration
                        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0 else None)
        value['recovery'] = dict(status='available', original_recovery=recovery,
            original_shot_recovery={key: result['records'][0].get(key)
                                    for key in ('recovered', 'standing_seconds', 'longest_standing_seconds', 'failed')},
            final_stable_standing_start_s=stable_start,
            time_from_contact_to_final_stable_standing_s=(stable_start - contact['first_force_s']
                if stable_start is not None and contact['status'] == 'available' else None),
            definition='Original frozen recovery criterion retained; start time derived from original final-standing duration, not a new gate')
    if contact['status'] != 'available':
        missing = unavailable('No measured contact to align the window', 'actual foot-ball contact')
        value.update(precontact_joint_dynamics=missing, post_plant_joint_dynamics=missing,
                     kinematic_foot_contributions=missing, support_braking=missing, segment_sequencing=missing, follow_through=missing)
        return value
    contact_s = contact['first_force_s']
    value['precontact_joint_dynamics'] = joint_dynamics(rows, result['joint_names'], contact_s, window_s)
    value['support_braking'] = support_braking(rows, contact_s, direction, window_s,
                                              metadata=result.get('dynamics_telemetry'))
    support = value['support_braking']
    value['post_plant_joint_dynamics'] = (joint_dynamics(rows, result['joint_names'], contact_s,
                                         contact_s - support['window_start_s'])
        if support['status'] == 'available' else unavailable('No loaded plant interval to align joint dynamics',
                                                            'contiguous precontact support reaching the last precontact sample'))
    value['kinematic_foot_contributions'] = kinematic_contributions(rows, contact_s, direction, window_s)
    value['segment_sequencing'] = segment_sequencing(result, rows, contact_s, window_s)
    value['follow_through'] = follow_through(rows, contact, direction, follow_window_s)
    return value
