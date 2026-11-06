#!/usr/bin/env python3
"""Offline P0 audit of saved v28/v27 physics. Never runs a simulator or GPU.

Original task acceptance and retrospective follow-through diagnostics are
reported separately. Sampled geometry and finite differences do not certify
continuous collision freedom, human naturalness, or hardware performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from follow_through import FOLLOW_THROUGH_CRITERIA, audit_follow_through
from free_shooting_metrics import ShotTracker, paid_contact_patch
from strike_baseline_metrics import summarize_strike, validated_rows
from targeted_shooting import controller_provenance

ROOT = Path(__file__).resolve().parents[1]
INVARIANTS = (
    'initial_robot_state', 'asset_sha256', 'joint_names', 'joint_effort_limits',
    'joint_velocity_limits', 'goals', 'goal_width_m', 'goal_height_m',
    'controller_sources_sha256', 'motion_sources_sha256',
    'sequence_sources_sha256', 'termination_policy', 'contact_geometry',
    'config_sha256', 'ground_collision_paths',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def unit(vector):
    value = np.asarray(vector, float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-10:
        raise ValueError('Finite nonzero three-dimensional direction required')
    return value / np.linalg.norm(value)


def sustained_start(times, flags, duration=.04):
    """Start of the first consecutive interval run lasting at least duration."""
    start = None
    for i, flag in enumerate(flags):
        if flag:
            if start is None:
                start = i
            if times[i + 1] - times[start] >= duration - 1e-9:
                return float(times[start])
        else:
            start = None
    return None


def measure_path(times, feet, roots, contact_time, direction):
    """Measure 20 ms observed paths; a constant world offset cancels exactly."""
    times, feet, roots = (np.asarray(x, float) for x in (times, feet, roots))
    if (times.ndim != 1 or len(times) < 3 or feet.shape != (len(times), 3)
            or roots.shape != feet.shape or not all(np.isfinite(x).all() for x in (times, feet, roots))
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Finite ordered measured foot/root poses required')
    begin, end = contact_time - .04, contact_time + .75
    if times[0] > begin + 1e-8 or times[-1] < end - 1e-8:
        raise ValueError('Missing bracketing follow-through samples')
    lo = max(0, np.searchsorted(times, begin + 1e-8, side='right') - 1)
    hi = np.searchsorted(times, end - 1e-8)
    times, feet, roots = times[lo:hi + 1] - contact_time, feet[lo:hi + 1], roots[lo:hi + 1]
    if np.any(np.diff(times) > .021):
        raise ValueError('Missing 20 ms measured pose samples')
    direction = unit(direction)
    forward = (feet - feet[0]) @ direction
    carry = (roots - roots[0]) @ direction
    forward -= np.interp(0., times, forward)
    carry -= np.interp(0., times, carry)
    knots = np.r_[0., times[(times > 0) & (times < .25)], .25]
    progress, body = np.interp(knots, times, forward), np.interp(knots, times, carry)
    grid = np.r_[np.arange(0., .75, .02), .75]
    world = np.interp(grid, times, forward)
    relative = world - np.interp(grid, times, carry)
    velocity, rel_velocity = np.diff(world) / np.diff(grid), np.diff(relative) / np.diff(grid)
    centers = (grid[:-1] + grid[1:]) / 2
    returns = (centers >= .22) & (centers <= .45)
    recovery = centers >= .22
    speed120 = (np.interp(.13, times, forward) - np.interp(.11, times, forward)) / .02
    rel120 = speed120 - (np.interp(.13, times, carry) - np.interp(.11, times, carry)) / .02
    reversal = sustained_start(grid, velocity < -.15)
    result = dict(
        forward_advance_m=float(progress.max()), pelvis_relative_advance_m=float((progress - body).max()),
        forward_displacement_at_250ms_m=float(progress[-1]),
        pelvis_relative_displacement_at_250ms_m=float(progress[-1] - body[-1]),
        pelvis_carry_at_250ms_m=float(body[-1]),
        forward_speed_at_120ms_mps=float(speed120), pelvis_relative_speed_at_120ms_mps=float(rel120),
        reversal_after_s=reversal,
        pelvis_relative_reversal_after_s=sustained_start(grid, rel_velocity < -.15),
        first_sustained_stop_after_s=sustained_start(grid, abs(velocity) <= .15),
        first_relative_sustained_stop_after_s=sustained_start(grid, abs(rel_velocity) <= .15),
        peak_return_speed_mps=float(max(0., -velocity[returns].min())),
        peak_recovery_return_speed_mps=float(max(0., -velocity[recovery].min())),
        peak_relative_velocity_step_mps=float(np.max(abs(np.diff(rel_velocity)))),
        peak_relative_finite_difference_acceleration_mps2=float(np.max(abs(np.diff(rel_velocity) / np.diff(centers)))),
        interpretation='20 ms position finite differences; stop/reversal requires 40 ms, speed threshold 0.15 m/s; not a continuous-time or naturalness certificate',
    )
    c = FOLLOW_THROUGH_CRITERIA
    failures = []
    for key, bound, reason in [
        ('forward_advance_m', c['forward_advance_min_m'], 'short_forward_swing'),
        ('pelvis_relative_advance_m', c['pelvis_relative_advance_min_m'], 'short_relative_swing'),
        ('forward_speed_at_120ms_mps', c['forward_speed_min_mps'], 'early_deceleration'),
    ]:
        if result[key] < bound:
            failures.append(reason)
    if reversal is not None and reversal < c['earliest_sustained_reversal_s'] - 1e-8:
        failures.append('early_reversal')
    if result['peak_return_speed_mps'] > c['peak_return_speed_max_mps']:
        failures.append('fast_retraction')
    if result['peak_recovery_return_speed_mps'] > c['recovery_return_speed_max_mps']:
        failures.append('fast_recovery_retraction')
    result['diagnostic_failures'] = failures
    return result


def contact_geometry(rows):
    """Rebuild local contacts using recorded pre-integration poses, not labels."""
    point_errors, normal_errors, impulses, normals, missing = [], [], {}, [], 0
    pose_errors, surface_residuals, top_goal_inputs = [], [], []
    for row in rows:
        for event in row.get('contacts', []):
            if event['force_N'] <= 0:
                continue
            weight = event['force_N'] * row['physics_dt_s']
            surface = paid_contact_patch(event['point_foot'], event['normal_foot'], foot=event['foot'])
            impulses[surface] = impulses.get(surface, 0.) + weight
            try:
                q = np.asarray(event['contact_foot_quat_w'], float)
                rotation = Rotation.from_quat(q[[1, 2, 3, 0]])
                n = np.asarray(event['raw_outward_normal_world'], float)
                point = (np.asarray(event['raw_sphere_point_world']) - event['raw_separation_m'] * n
                         - np.asarray(event['contact_foot_position_w']))
                point_errors.append(float(np.linalg.norm(rotation.inv().apply(point) - event['point_foot'])))
                normal_errors.append(float(np.linalg.norm(rotation.inv().apply(n) - event['normal_foot'])))
                pose_errors.append(abs(event['foot_pose_time_s'] - (row['time_s'] - row['physics_dt_s'])))
                surface_residuals.append(event['capsule_surface_residual_m'])
                if event.get('point_frame') != 'pre_step_capsule_surface':
                    missing += 1
                normals.append((row['time_s'], weight, n))
                top_goal_inputs.append((weight, rotation.apply([0., 0., 1.]), rotation.apply([1., 0., 0.])))
            except (KeyError, TypeError, ValueError):
                missing += 1
    total = sum(impulses.values())
    finite = all(np.isfinite(x) for x in point_errors + normal_errors + pose_errors + surface_residuals)
    return dict(
        frame_reconstruction_passed=bool(point_errors and not missing and finite
            and max(point_errors) <= 2e-5 and max(normal_errors) <= 2e-5 and max(pose_errors) <= 1e-7),
        missing_contact_geometry_events=missing, contact_point_count=len(point_errors),
        maximum_local_point_reconstruction_error_m=max(point_errors, default=None),
        maximum_local_normal_reconstruction_error=max(normal_errors, default=None),
        maximum_saved_capsule_surface_residual_m=max(surface_residuals, default=None),
        impulse_by_surface_Ns=impulses, instep_impulse_fraction=impulses.get('instep', 0.) / total if total else 0.,
        _normals=normals, _axes=top_goal_inputs,
    )


def segment_distance(a, b, c, d):
    """Exact Euclidean distance between finite segments, vectorized over rows."""
    a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d))
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = (u*u).sum(-1), (u*v).sum(-1), (v*v).sum(-1)
    dd, ee = (u*w).sum(-1), (v*w).sum(-1)
    den = aa*cc - bb*bb
    s, t = (bb*ee - cc*dd) / np.maximum(den, 1e-15), (aa*ee - bb*dd) / np.maximum(den, 1e-15)
    interior = (den > 1e-15) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    middle = ((w + s[..., None]*u - t[..., None]*v)**2).sum(-1)
    def point(p, x, y):
        line = y - x
        f = np.clip(((p - x)*line).sum(-1) / np.maximum((line*line).sum(-1), 1e-15), 0., 1.)
        return ((p - x - f[..., None]*line)**2).sum(-1)
    return np.sqrt(np.maximum(0., np.min(np.stack([
        point(a, c, d), point(b, c, d), point(c, a, b), point(d, a, b),
        np.where(interior, middle, np.inf)], axis=-1), axis=-1)))


class CapsuleGeometry:
    """Independent double-precision URDF forward kinematics; CPU NumPy only."""
    def __init__(self, path, names):
        xml = ET.parse(path).getroot()
        self.links = {x.attrib['name']: x for x in xml.findall('link')}
        self.joints = {x.attrib['name']: x for x in xml.findall('joint')}
        self.names = names

    @staticmethod
    def origin(node):
        e = node.find('origin')
        values = {} if e is None else e.attrib
        return (np.fromstring(values.get('xyz', '0 0 0'), sep=' '),
                Rotation.from_euler('xyz', np.fromstring(values.get('rpy', '0 0 0'), sep=' ')).as_matrix())

    def pose(self, q, side, parts):
        p, r = np.zeros((len(q), 3)), np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
        for part in parts:
            name = side + '_' + part + '_joint'
            node = self.joints[name]
            origin, rotation = self.origin(node)
            axis = np.fromstring(node.find('axis').attrib['xyz'], sep=' ')
            p += np.einsum('nij,j->ni', r, origin)
            r = r @ rotation @ Rotation.from_rotvec(q[:, self.names.index(name), None] * axis).as_matrix()
        return p, r

    def capsule(self, q, side, link, parts):
        p, r = self.pose(q, side, parts)
        collision = self.links[side + '_' + link + '_link'].find('collision')
        center, orientation = self.origin(collision)
        shape = collision.find('geometry/cylinder')
        if shape is None:
            raise ValueError('Original capsule-converted cylinder is required')
        axis = orientation[:, 2] * float(shape.attrib['length']) / 2
        a = p + np.einsum('nij,j->ni', r, center - axis)
        b = p + np.einsum('nij,j->ni', r, center + axis)
        return a, b, float(shape.attrib['radius'])

    def gaps(self, q):
        q = np.asarray(q, float)
        if q.shape != (len(q), len(self.names)) or not np.isfinite(q).all():
            raise ValueError('Finite joint positions in declared order required')
        pelvis = self.links['pelvis'].find('collision')
        center, _ = self.origin(pelvis)
        radius = float(pelvis.find('geometry/sphere').attrib['radius'])
        gaps = {}
        for side in ('left', 'right'):
            a, b, rr = self.capsule(q, side, 'hip_roll', ['hip_pitch', 'hip_roll'])
            gaps[side + '_pelvis_thigh'] = segment_distance(center, center, a, b) - radius - rr
        left = self.capsule(q, 'left', 'knee', ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee'])
        right = self.capsule(q, 'right', 'knee', ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee'])
        gaps['shank_to_shank'] = segment_distance(*left[:2], *right[:2]) - left[2] - right[2]
        return gaps


def verify_invariants(run, baseline):
    failures = [key for key in INVARIANTS if key not in run or key not in baseline or run[key] != baseline[key]]
    failures += [key for key in ('root_fixed', 'post_initialization_state_writes') if run.get(key) is not False]
    return dict(recorded_invariants_match=not failures, failures=failures,
        recorded_no_post_initialization_state_writes=run.get('post_initialization_state_writes') is False,
        limitation='Source hashes plus recorded flags support the unchanged contract; no per-call state-write/force instrumentation or runtime collision-filter snapshot was saved')


def performance_delta(old, new):
    if old is None:
        return None
    def difference(key):
        a, b = (x.get('contact') for x in (old, new))
        return b[key] - a[key] if a and b and a.get(key) is not None and b.get(key) is not None else None
    return dict(original_task_regression=bool(old['outcome']['success'] and not new['outcome']['success']),
        outgoing_speed_xy_delta_mps=difference('outgoing_speed_xy_mps'),
        outgoing_heading_error_delta_deg=difference('outgoing_heading_error_deg'),
        final_standing_delta_s=new['final_standing_seconds'] - old['final_standing_seconds'])


def limit_observations(rows, run):
    torque = abs(np.asarray([r['joint_torque_estimate'] for r in rows]))
    velocities = abs(np.asarray([r['joint_vel'] for r in rows]))
    targets = abs(np.asarray([r['joint_velocity_target'] for r in rows]))
    limit = np.asarray(run['joint_velocity_limits'][0])
    ratios = velocities / limit
    i, j = np.unravel_index(np.argmax(ratios), ratios.shape)
    return dict(maximum_effort_ratio=float(np.max(torque / run['joint_effort_limits'][0])),
        maximum_actual_speed_ratio=float(ratios[i, j]),
        maximum_velocity_target_ratio=float(np.max(targets / limit)),
        actual_speed_exceedance_sample_joint_count=int(np.sum(velocities > limit + 1e-6)),
        worst_actual_speed=dict(time_s=rows[i]['time_s'], joint=run['joint_names'][j],
            absolute_speed_rad_s=float(velocities[i, j]), limit_rad_s=float(limit[j])),
        note='Actual velocity can transiently exceed a solver limit; effort is saved clipped-PD estimate, not hardware torque')


def impact_window(run, event_time=1.855, half_width=.1):
    """Preserve the observed regression window without assigning a cause."""
    rows = validated_rows(run['traces'][0])
    index = min(range(len(rows)), key=lambda i: abs(rows[i]['time_s'] - event_time))
    previous, event = rows[index - 1], rows[index]
    fields = ['time_s', 'joint_pos', 'joint_vel', 'joint_target', 'joint_velocity_target',
              'joint_torque_estimate', 'support_foot_force_N', 'right_foot_point_velocity',
              'nonfoot_contact_N', 'contact_force_N']
    selected = lambda r: event_time - half_width - 1e-8 <= r['time_s'] <= event_time + half_width + 1e-8
    feedback_fields = ['time_s', 'phase_s', 'follow_through_active', 'follow_through_correction_rad',
                       'shank_active', 'shank_correction_rad', 'shank_ankle_correction_rad',
                       'swing_reach_fraction', 'approach_correction_rad']
    return dict(event_time_s=event['time_s'], window_half_width_s=half_width,
        joint_names=run['joint_names'],
        position_targets_unchanged_across_event=previous['joint_target'] == event['joint_target'],
        velocity_targets_unchanged_across_event=previous['joint_velocity_target'] == event['joint_velocity_target'],
        measured_point_velocity_step_mps=float(np.linalg.norm(np.asarray(event['right_foot_point_velocity'])
                                                             - previous['right_foot_point_velocity'])),
        before_right_ground_force_N=previous['support_foot_force_N']['right'],
        event_right_ground_force_N=event['support_foot_force_N']['right'],
        physical=[{k: r[k] for k in fields} for r in rows if selected(r)],
        foot_poses=[dict(time_s=r['time_s'], right=r['feet']['right']) for r in run['kinematics'][0] if selected(r)],
        controller=[{k: r.get(k) for k in feedback_fields} for r in run['joint_strike']['feedback'] if selected(r)],
        interpretation='Coincidence and command continuity evidence only; ground-force scalar lacks contact-normal/tangential decomposition and cannot establish root cause')


def audit_case(run, baseline_contract, urdf):
    rows = validated_rows(run['traces'][0])
    summary = summarize_strike(run)
    trial = run['trials'][0]
    tracker = ShotTracker(initial_root=run['initial_robot_state']['root_position'][0],
        initial_ball=[*run['points'][0], .113], goal=run['goals'][0],
        forward=trial['goal_forward'], style=trial['style'],
        width=run['goal_width_m'], height=run['goal_height_m'])
    for row in run['traces'][0]:
        tracker.update(row)
    replay = tracker.summary()
    contact = summary['contact']
    geometry = contact_geometry(rows)
    normals, axes = geometry.pop('_normals'), geometry.pop('_axes')
    q = np.asarray([r['joint_pos'] for r in rows])
    gaps = CapsuleGeometry(urdf, run['joint_names']).gaps(q)
    clearances = {}
    for name, values in gaps.items():
        index = int(np.argmin(values))
        clearances[name] = dict(minimum_sampled_gap_m=float(values[index]), time_s=rows[index]['time_s'],
                               negative_sample_count=int(np.sum(values < 0)))
    illegal = [dict(time_s=r['time_s'], body=name, force_N=force)
        for r in rows for name, force in r['robot_ball_contact_force_N'].items()
        if name not in ('right_ankle_roll_link', 'left_ankle_roll_link') and force > 2.]
    result = dict(original=summary, original_shot_replay_matches=replay == run['records'][0],
        invariants=verify_invariants(run, baseline_contract), contact_geometry=geometry,
        illegal_body_ball_contacts=illegal, maximum_nonfoot_force_N=max(r['nonfoot_contact_N'] for r in rows),
        joint_limit_observations=limit_observations(rows, run),
        sampled_clearance=clearances, dynamic_follow_through=None)
    if contact is None:
        return result
    ct, last = contact['first_force_s'], contact['last_force_s']
    poses = run['kinematics'][0]
    times = np.asarray([x['time_s'] for x in poses])
    roots = np.asarray([x['root_pose'][:3] for x in poses])
    ankle = np.asarray([x['feet']['right']['position'] for x in poses])
    quats = np.asarray([x['feet']['right']['quaternion'] for x in poses])
    points = ankle + Rotation.from_quat(quats[:, [1, 2, 3, 0]]).apply(run['foot_velocity_point_local'])
    goal = unit([*(np.asarray(run['goals'][0]) - np.asarray(run['points'][0])), 0.])
    normal_sum = sum((weight * n for t, weight, n in normals if ct - 1e-8 <= t <= last + 1e-8), np.zeros(3))
    normal = unit(normal_sum)
    normal_xy = unit([*normal[:2], 0.])
    directions = dict(goal_horizontal=goal, contact_normal_horizontal=normal_xy)
    paths = {}
    for name, direction in directions.items():
        paths[name] = dict(direction_world=direction.tolist(),
            ankle_origin=measure_path(times, ankle, roots, ct, direction),
            tracked_point=measure_path(times, points, roots, ct, direction),
            after_separation_tracked_point=measure_path(times, points, roots, contact['outgoing_time_s'], direction))
    selected = [r for r in rows if last < r['time_s'] <= ct + .75 + 1e-9]
    vt = np.asarray([r['time_s'] for r in selected])
    vv = np.asarray([r['right_foot_point_velocity'] for r in selected])
    jumps = np.linalg.norm(np.diff(vv, axis=0), axis=1)
    peak = int(np.argmax(jumps))
    strict = audit_follow_through(run, 0)
    axis_weight = sum(x[0] for x in axes)
    result['dynamic_follow_through'] = dict(
        contact_time_s=ct, first_separated_sample_s=contact['outgoing_time_s'], paths=paths,
        first_episode_weighted_contact_normal_world=normal.tolist(),
        contact_normal_goal_angle_deg=float(np.degrees(np.arccos(np.clip(normal_xy @ goal, -1., 1.)))),
        preintegration_contact_pose=dict(
            impulse_weighted_top_axis_toward_goal=sum(w * float(top @ goal) for w, top, toe in axes) / axis_weight,
            impulse_weighted_toe_world_z=sum(w * float(toe[2]) for w, top, toe in axes) / axis_weight),
        measured_point_velocity=dict(
            maximum_post_separation_vector_step_mps=float(jumps[peak]),
            maximum_post_separation_finite_difference_acceleration_mps2=float(np.max(jumps / np.diff(vt))),
            maximum_step_interval_s=[float(vt[peak]), float(vt[peak + 1])],
            point_local_m=run['foot_velocity_point_local'], sampling_s=.005,
            note='Measured body-link point velocity; finite-difference spikes include legitimate ground/impact dynamics and are not human-naturalness thresholds'),
        legacy_additional_diagnostic=strict,
        diagnostic_status='Retrospective additional diagnostic; thresholds are not frozen v28 task acceptance or a human-naturalness standard')
    return result


def build_audit(root, comparison_path):
    comparison = json.loads(comparison_path.read_text())
    reference = json.loads((root / 'reports/v027_final_center.json').read_text())
    config_path = root / 'configs/state_conditioned_strike_v5.json'
    config = json.loads(config_path.read_text())
    runtime_sources = controller_provenance(root)['controller_sources_sha256']
    frozen = [dict(path=name, expected_sha256=digest, actual_sha256=sha(root / name))
              for name, digest in config['frozen_files_sha256'].items()]
    sources = [dict(path='tools/' + name, expected_sha256=digest, actual_sha256=sha(root / 'tools' / name))
               for name, digest in comparison['sources'].items()]
    results = []
    urdf = root / 'configs/paid_original/paid_main_resolved.urdf'
    for entry in comparison['cases']:
        name = entry['case']['id']
        candidate_path = root / entry['candidate_report']
        if sha(candidate_path) != entry['candidate_sha256']:
            raise ValueError('Changed published candidate report: ' + name)
        candidate = audit_case(json.loads(candidate_path.read_text()), reference, urdf)
        if candidate['original'] != entry['candidate']:
            raise ValueError('Recomputed original candidate summary differs: ' + name)
        old = None
        if entry['baseline_report'] is not None:
            old_path = root / entry['baseline_report']
            if sha(old_path) != entry['baseline_sha256']:
                raise ValueError('Changed published baseline report: ' + name)
            old = audit_case(json.loads(old_path.read_text()), reference, urdf)
            if old['original'] != entry['baseline']:
                raise ValueError('Recomputed original baseline summary differs: ' + name)
        results.append(dict(case=entry['case'], case_group=entry['case_group'],
            candidate_report=entry['candidate_report'], candidate_sha256=entry['candidate_sha256'],
            baseline_report=entry['baseline_report'], baseline_sha256=entry['baseline_sha256'],
            candidate=candidate, baseline=old,
            delta=performance_delta(None if old is None else old['original'], candidate['original'])))
        print(name, 'original_success=' + str(candidate['original']['outcome']['success']), flush=True)
    matched = [x for x in results if x['baseline'] is not None]
    evaluation = [x for x in results if x['case_group'] == 'evaluation']
    successful = lambda x, key: x[key]['original']['outcome']['success']
    impact = {version: impact_window(json.loads((root / f'reports/{version}_final_initial_slower_02.json').read_text()))
              for version in ('v027', 'v028')}
    return dict(schema_version=1, audit='P0 saved v28 physical acceptance and additional motion diagnostics',
        comparison_source=dict(path=str(comparison_path.relative_to(root)), sha256=sha(comparison_path)),
        auditor_sha256=sha(__file__), config_sha256=sha(config_path),
        frozen_files=frozen, execution_sources=sources,
        frozen_runtime_sources=runtime_sources,
        frozen_runtime_sources_verified=runtime_sources == reference['controller_sources_sha256'],
        frozen_files_verified=all(x['actual_sha256'] == x['expected_sha256'] for x in frozen),
        execution_sources_verified=all(x['actual_sha256'] == x['expected_sha256'] for x in sources),
        all_saved_original_summaries_recomputed=True,
        totals=dict(existing=len(matched), v27_existing_successes=sum(successful(x, 'baseline') for x in matched),
            v28_existing_successes=sum(successful(x, 'candidate') for x in matched),
            new_cases=len(evaluation), v28_new_successes=sum(successful(x, 'candidate') for x in evaluation),
            v28_task_successes=sum(successful(x, 'candidate') for x in results),
            original_task_regressions=sum(x['delta']['original_task_regression'] for x in matched),
            additional_follow_through_diagnostic_passes=sum(x['candidate']['dynamic_follow_through']['legacy_additional_diagnostic']['follow_through_ok'] for x in results)),
        additional_follow_through_criteria=FOLLOW_THROUGH_CRITERIA,
        initial_slower_02_impact_comparison=impact,
        limitations=[
            'Deterministic case counts are not population success rates; v27 reports for the three new cases do not exist.',
            'The stricter follow-through/posture diagnostic is additional and retrospective, not frozen v28 acceptance or a naturalness label.',
            '5 ms joint-position geometry is sampled URDF capsule separation; it cannot establish between-sample collision clearance.',
            'Runtime collision filtering/self-collision flags and a post-reset write/force call log were not serialized. Unchanged hashes and metadata support the source contract only.',
            'No calibrated human reference, human evaluation, hardware torque, or hardware power evidence is available.',
        ], cases=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--comparison', type=Path, default=Path('reports/v028_final_comparison.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/p0_v028_acceptance.json'))
    args = parser.parse_args()
    result = build_audit(args.root, args.root / args.comparison)
    output = args.root / args.output
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['totals'], indent=2))


if __name__ == '__main__':
    main()
