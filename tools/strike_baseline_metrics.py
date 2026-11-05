"""Contact-aligned diagnostics; no naturalness score, hardware power or energy attribution."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def validated_rows(rows):
    if not rows:
        raise ValueError('Nonempty physical telemetry required')
    previous = None
    physical = []
    for index, row in enumerate(rows):
        t, dt = row['time_s'], row['physics_dt_s']
        force = row['contact_force_N']
        if not all(math.isfinite(v) for v in (t, dt, force)) or t < 0 or not 0 < dt <= .0075 or force < 0:
            raise ValueError('Finite 5ms physical samples required')
        if previous is None and abs(t-dt) > 1e-7:
            raise ValueError('Physical trace must start at the first simulation sample')
        if (previous is not None and index == len(rows)-1 and row.get('terminated')
                and abs(t-previous-1e-6) < 1e-9):
            # Original producer appends an identical terminal status at t+1us.
            # It is not another physical integration bin.
            without_status = lambda r: {k:v for k,v in r.items() if k not in ('time_s','terminated')}
            if without_status(row) != without_status(physical[-1]):
                raise ValueError('Terminal status is not a copy of the preceding physical sample')
            continue
        if previous is not None and abs(t-previous-dt) > 1e-7:
            raise ValueError('Missing or overlapping physics samples')
        previous = t
        physical.append(row)
    return physical


def first_force_episode(rows):
    rows = validated_rows(rows)
    start = next((i for i, r in enumerate(rows) if r['contact_force_N'] > 0), None)
    if start is None:
        return None
    end = start
    while end < len(rows) and rows[end]['contact_force_N'] > 0:
        end += 1
    contact = rows[start:end]
    value = dict(first_force_s=rows[start]['time_s'], last_force_s=contact[-1]['time_s'],
        force_duration_s=sum(r['physics_dt_s'] for r in contact),
        normal_impulse_Ns=sum(r['contact_force_N']*r['physics_dt_s'] for r in contact),
        peak_normal_force_N=max(r['contact_force_N'] for r in contact),
        outgoing_time_s=None, outgoing_speed_xy_mps=None, outgoing_speed_3d_mps=None,
        outgoing_velocity_mps=None, pre_contact_reference_point_speed_mps=None)
    if end < len(rows):
        v = np.asarray(rows[end]['ball_velocity'], float)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError('Finite outgoing ball velocity required')
        value.update(outgoing_time_s=rows[end]['time_s'], outgoing_velocity_mps=v.tolist(),
            outgoing_speed_xy_mps=float(np.linalg.norm(v[:2])), outgoing_speed_3d_mps=float(np.linalg.norm(v)))
    if start:
        v = np.asarray(rows[start-1]['right_foot_point_velocity'], float)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError('Finite pre-contact foot reference-point velocity required')
        value['pre_contact_reference_point_speed_mps'] = float(np.linalg.norm(v))
        value['pre_contact_reference_point_time_s'] = rows[start-1]['time_s']
    return value


def preparation_metrics(rows, names, contact_s, window_s=.30):
    rows = validated_rows(rows)
    if len(set(names)) != len(names) or not names or not math.isfinite(contact_s) or not math.isfinite(window_s) or window_s <= 0:
        raise ValueError('Unique joints and a positive finite contact window required')
    start = max(0., contact_s-window_s)
    samples = [r for r in rows if start < r['time_s'] < contact_s-1e-9]
    if not samples:
        raise ValueError('No preparatory samples before measured contact')
    weights = np.array([max(0., r['time_s']-max(start, r['time_s']-r['physics_dt_s'])) for r in samples])
    try:
        velocity = np.array([r['joint_vel'] for r in samples], float)
        torque = np.array([r['joint_torque_estimate'] for r in samples], float)
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing joint velocity/estimated torque evidence') from exc
    if (velocity.shape != (len(samples), len(names)) or torque.shape != velocity.shape
            or not np.isfinite(velocity).all() or not np.isfinite(torque).all()):
        raise ValueError('Finite correctly ordered joint telemetry required')
    expected = max(0., min(window_s, contact_s)-samples[-1]['physics_dt_s'])
    if weights.sum() < expected-1e-7:
        raise ValueError('Preparatory measurements do not cover the declared window')
    power = velocity*torque
    times = np.array([r['time_s'] for r in samples])
    joints = {}
    for j, name in enumerate(names):
        k = int(np.argmax(abs(velocity[:, j])))
        joints[name] = dict(peak_abs_velocity_rad_s=float(abs(velocity[k, j])),
            signed_velocity_at_peak_rad_s=float(velocity[k, j]),
            peak_velocity_time_to_contact_s=float(times[k]-contact_s),
            positive_work_J=float(np.maximum(power[:, j], 0)@weights),
            negative_work_J=float(np.minimum(power[:, j], 0)@weights),
            peak_positive_power_W=float(np.maximum(power[:, j], 0).max()),
            peak_negative_power_W=float(np.minimum(power[:, j], 0).min()))
    return dict(window_start_s=start, window_end_s=contact_s, measured_duration_s=float(weights.sum()),
        coverage_fraction=float(weights.sum()/min(window_s, contact_s)), joints=joints,
        whole_body_positive_work_J=float(np.maximum(power, 0).sum(axis=1)@weights),
        whole_body_negative_work_J=float(np.minimum(power, 0).sum(axis=1)@weights),
        definition='Estimated clipped PD torque times measured joint speed; post-step rectangular bins; excludes collision bin')


def yaw_excursion(quaternions_wxyz):
    q = np.asarray(quaternions_wxyz, float)
    if q.ndim != 2 or q.shape[1] != 4 or not len(q) or not np.isfinite(q).all():
        raise ValueError('Finite observed wxyz orientations required')
    heading = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()[:, :2, 0]
    yaw = np.unwrap(np.arctan2(heading[:, 1], heading[:, 0]))
    return dict(net_deg=float(np.degrees(yaw[-1]-yaw[0])), range_deg=float(np.degrees(np.ptp(yaw))))


def task_outcome(result):
    shot = result['records'][0]
    reasons = []
    if shot['contact_time_s'] is None: reasons.append('no_foot_contact')
    if not shot['surface_match']: reasons.append('wrong_contact_surface')
    if not shot['goal']: reasons.append('missed_goal')
    if not shot['recovered']: reasons.append('recovery_failed')
    if shot['failed']: reasons.append('physical_failure')
    if not result['running_evidence']['running_success']: reasons.append('running_failed')
    if not result['stride_shot_evidence']['continuous_running_contact']: reasons.append('running_strike_transition_failed')
    if result['sequence_records'][0]['illegal_body_ball_contact']: reasons.append('illegal_body_ball_contact')
    if result['root_fixed'] is not False or result['post_initialization_state_writes'] is not False:
        reasons.append('physical_integrity_failed')
    if not result['stride_shot_success'] and not reasons: reasons.append('sequence_gate_failed')
    return dict(success=bool(result['stride_shot_success'] and not reasons), failure_reasons=reasons)


def validated_episode(result, expected_duration_s=14.):
    raw = result['traces'][0]
    rows = validated_rows(raw)
    terminal = len(raw) != len(rows)
    if terminal:
        if (not result['records'][0]['failed'] or not result.get('termination_reasons', [[]])[0]
                or rows[-1]['time_s'] > expected_duration_s+1e-7):
            raise ValueError('Early ending requires a genuine recorded physical termination')
    elif abs(rows[-1]['time_s']-expected_duration_s) > 1e-7:
        raise ValueError('Trace ends before/after the declared experiment horizon')
    saved = result['records'][0]['contact_time_s']
    if saved is not None and not any(abs(r['time_s']-saved) < 1e-7 and r['contact_force_N'] > 0 for r in rows):
        raise ValueError('Stored contact outcome is absent from physical evidence')
    if result['records'][0]['success'] and not any(r['contact_force_N'] > 0 for r in rows):
        raise ValueError('Successful shot cannot have no measured foot-ball force')
    return rows


def summarize_strike(result, mass_kg=.43, window_s=.30):
    rows = validated_episode(result)
    episode = first_force_episode(rows)
    shot = result['records'][0]
    summary = dict(schema_version=1, outcome=task_outcome(result), contact=episode,
        original_shot=shot, final_standing_seconds=shot['standing_seconds'],
        completed_physics_seconds=rows[-1]['time_s'], physical_steps=len(rows),
        terminal_status_markers=len(result['traces'][0])-len(rows),
        termination_reasons=result.get('termination_reasons'),
        torque_definition=result['torque_evidence'], foot_reference_point_local=result['foot_velocity_point_local'],
        precontact=None, rotation=None, support=None)
    if episode is None:
        return summary
    contact = episode['first_force_s']
    prep = preparation_metrics(rows, result['joint_names'], contact, window_s)
    summary['precontact'] = prep
    before = [r for r in rows if contact-window_s < r['time_s'] < contact-1e-9]
    direction = np.r_[np.asarray(result['goals'][0])-np.asarray(before[-1]['ball_xyz'][:2]), 0.]
    direction /= np.linalg.norm(direction)
    if episode['outgoing_velocity_mps'] is not None:
        v = np.asarray(episode['outgoing_velocity_mps'])
        u = np.asarray(before[-1]['ball_velocity'])
        episode['ball_translational_energy_gain_J'] = float(.5*mass_kg*(v@v-u@u))
        episode['ball_momentum_change_Ns'] = (mass_kg*(v-u)).tolist()
        episode['outgoing_heading_error_deg'] = (float(np.degrees(np.arctan2(abs(np.cross(direction[:2], v[:2])), direction[:2]@v[:2])))
            if np.linalg.norm(v[:2]) > 1e-8 else None)
    prep['foot_velocity_integrals_m'] = {}
    for part in ('base', 'hip', 'knee', 'ankle'):
        values = np.array([np.asarray(r['foot_velocity_contributions'][part])@direction for r in before])
        weights = np.array([min(r['physics_dt_s'], r['time_s']-(contact-window_s)) for r in before])
        prep['foot_velocity_integrals_m'][part] = dict(signed=float(values@weights), positive=float(np.maximum(values, 0)@weights))
    poses = [k for k in result['kinematics'][0] if contact-window_s <= k['time_s'] < contact]
    if len(poses) >= 2:
        waist = result['joint_names'].index('waist_yaw_joint')
        q = np.array([k['joint_pos'][waist] for k in poses])
        summary['rotation'] = dict(pelvis_yaw=yaw_excursion([k['root_pose'][3:] for k in poses]),
            relative_waist_yaw_range_deg=float(np.degrees(np.ptp(q))),
            observed_sample_count=len(poses), sampling_dt_s=.02,
            note='Pelvis projected heading and relative waist joint rotation, not human pelvis/thorax ground truth')
    support = np.array([r['support_foot_force_N']['left'] for r in before])
    start_index = next(i for i, r in enumerate(rows) if r['time_s'] >= contact)
    plant = start_index
    while plant >= 0 and rows[plant]['support_foot_force_N']['left'] >= 80.:
        plant -= 1
    supported = []
    for a, b in zip(poses, poses[1:]):
        ra = min(rows, key=lambda r: abs(r['time_s']-a['time_s']))
        rb = min(rows, key=lambda r: abs(r['time_s']-b['time_s']))
        if ra['support_foot_force_N']['left'] >= 80 and rb['support_foot_force_N']['left'] >= 80:
            p = np.asarray(b['feet']['left']['position'])-np.asarray(a['feet']['left']['position'])
            supported.append(float(np.linalg.norm(p[:2])))
    plant_window = contact-rows[plant+1]['time_s'] if plant < start_index else 0.
    summary['post_plant_precontact'] = (preparation_metrics(rows, result['joint_names'], contact, plant_window)
        if plant_window > .01 else None)
    summary['support'] = dict(left_force_min_N=float(support.min()), left_force_peak_N=float(support.max()),
        left_force_mean_N=float(support.mean()),
        last_continuous_left_support_before_contact_s=contact-rows[plant+1]['time_s'] if plant < start_index else 0.,
        loaded_left_ankle_link_path_xy_m=sum(supported) if supported else None,
        loaded_kinematic_intervals=len(supported),
        note='80N force gate; 20ms ankle-link path is a motion proxy and includes pivoting, not a sole-slip measurement')
    return summary
