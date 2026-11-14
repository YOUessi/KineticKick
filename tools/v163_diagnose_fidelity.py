"""Compare a matched physical pair to the same full human sequence and a held start pose."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from v163_diagnose_motion import rotation, sha


GROUPS = {
    'left_leg': ['left_knee_link', 'left_ankle_roll_link'],
    'right_leg': ['right_knee_link', 'right_ankle_roll_link'],
    'arms': [s + '_' + joint + '_link' for s in ('left', 'right') for joint in ('shoulder_pitch', 'elbow', 'wrist_yaw')],
    'trunk': ['torso_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link'],
}


def pelvis_relative(position, r):
    return np.einsum('tji,tkj->tki', r.as_matrix(), position-position[:, :1])


def series_metrics(p, reference, q, rq, rot, ref_rot, names):
    reference_local = pelvis_relative(reference, ref_rot)
    local = pelvis_relative(p, rot)
    root_delta = p[:, 0] - reference[:, 0]
    # Heading error has an explicit world vertical; the geodesic error includes the other two axes.
    heading = rot.as_euler('xyz')[:, 2] - ref_rot.as_euler('xyz')[:, 2]
    heading = np.rad2deg(np.arctan2(np.sin(heading), np.cos(heading)))
    out = dict(root_translation_xyz_rmse_m=float(np.sqrt(np.mean(np.sum(root_delta**2, axis=1)))),
               root_translation_axis_rmse_m=np.sqrt(np.mean(root_delta**2, axis=0)).tolist(),
               root_heading_absolute_mean_deg=float(np.mean(abs(heading))), root_heading_rmse_deg=float(np.sqrt(np.mean(heading**2))),
               root_orientation_geodesic_mean_deg=float(np.rad2deg((rot.inv()*ref_rot).magnitude()).mean()),
               all29_joint_rmse_rad=float(np.sqrt(np.mean((q-rq)**2))),
               all30_pelvis_relative_mpjpe_m=float(np.linalg.norm(local-reference_local, axis=-1).mean()), groups={})
    for name, links in GROUPS.items():
        ids = [names.index(link) for link in links]
        actual, source = local[:, ids], reference_local[:, ids]
        v, target = np.diff(actual, axis=0), np.diff(source, axis=0)
        den = float(np.linalg.norm(v) * np.linalg.norm(target))
        out['groups'][name] = dict(links=links, pelvis_relative_mpjpe_m=float(np.linalg.norm(actual-source,axis=-1).mean()),
                                   source_velocity_cosine=float(np.sum(v*target)/den) if den>1e-12 else None)
    return out


def last_swing_contact(folder, time_s, mask, body, names, forward):
    hits = np.flatnonzero(mask)
    if not len(hits):
        return None
    first, last = hits[[0, -1]]
    actors = set()
    with gzip.open(folder/'native_contacts.jsonl.gz', 'rt') as file:
        for line in file:
            tick, rows = json.loads(line)
            if tick == last + 1:
                for row in rows:
                    pair = set(row['actors'])
                    if 'SoccerBall' in pair and 'ground' not in pair:
                        net = np.array(row['points'])[:, 6:9].sum(axis=0)
                        if np.linalg.norm(net)/.005 > 2:
                            actors |= pair - {'SoccerBall'}
                break
    release = last + 1
    if not actors:
        raise ValueError('Inconsistent native contact log')
    out = dict(first_robot_ball_contact_s=float(time_s[first]), last_loaded_ball_sample_s=float(time_s[last]),
               first_separated_sample_s=float(time_s[release]) if release<len(time_s) else None, last_loaded_actors=sorted(actors))
    if 'right_ankle_roll_link' not in actors or release+60 >= len(time_s):
        return out
    foot = body[:, names.index('right_ankle_roll_link'), :2]
    relative = foot - body[:, 0, :2]
    out['foot_origin_follow_through'] = {}
    # A release is first separated 5ms sample, so all here are strictly post-contact.
    for name, data in [('world', foot), ('minus_pelvis_translation', relative)]:
        delta = (data[release:release+61] - data[release]) @ forward
        out['foot_origin_follow_through'][name] = {
            'forward_excursion_120ms_m': float(delta[24]), 'forward_excursion_250ms_m': float(delta[50]),
            'peak_forward_excursion_first_300ms_m': float(np.max(delta)),
            'time_of_peak_first_300ms_after_release_s': float(np.argmax(delta) * .005)}
    return out


def evaluate(folder, refpath, ref, clock_file):
    with np.load(folder/'rollout.npz', allow_pickle=False) as z:
        t = z['time_s']
        body = z['body_position_w_m']
        actual_q = z['joint_q_rad']
        quat = z['root_pose_wxyz'][:, 3:]
        contact = z['robot_ball_contact']
    report = json.loads((folder/'tracking_result.json').read_text())
    trial_path = folder/'trial.json'
    trial = json.loads(trial_path.read_text()) if trial_path.exists() else report
    for name, digest in report['local_artifact_sha256'].items():
        if sha(folder/name) != digest:
            raise ValueError(f'Different rollout evidence: {folder/name}')
    target_sha = sha(refpath)
    declared = trial.get('reference_sha256') or trial.get('external_reference_sha256') or report.get('reference_sha256')
    if declared is None:
        declared = sha(Path(report['source_clip']))
    if target_sha != declared:
        raise ValueError('Source path SHA differs from executed source')
    if not np.array_equal(t, np.arange(1, 2801)*.005) or report['additional_state_writes_after_initialization']:
        raise ValueError('Not a complete uninterrupted 14s physical episode')
    names = ref['body_names'].tolist()
    if report['source_body_names'] != names or report['source_joint_names'] != ref['joint_names'].tolist():
        raise ValueError('Body/joint order differs')
    rt = ref['time_s']
    rf = rt[1:]
    if clock_file:
        with np.load(clock_file) as clock:
            ct, source = clock['time_s'], clock['source_time_s']
        if not np.all(np.diff(ct) > 0) or not np.all(np.diff(source) >= 0) or source[0] > 0 or source[-1] < rt[-1]:
            raise ValueError('Reference clock does not cover full source')
        # If a phase is held, use its first arrival, never terminal stand samples.
        values, first_index = np.unique(source, return_index=True)
        phys_time = np.interp(rf, values, ct[first_index])
        if phys_time[0] < t[0] or phys_time[-1] > t[-1]:
            raise ValueError('Clock outside physical observation interval')
    else:
        phys_time = rf
    indices = np.arange(len(t))
    # Interpolate only when the physical sample falls between 5 ms endpoints.
    idx = np.interp(phys_time, t, indices)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo+1, len(t)-1)
    w = idx-lo
    actual = body[lo]*(1-w[:,None,None]) + body[hi]*w[:,None,None]
    joints = actual_q[lo]*(1-w[:,None]) + actual_q[hi]*w[:,None]
    orientations = Slerp(t, rotation(quat))(phys_time)
    base = ref['body_pos_w']
    q = ref['joint_pos']
    root_q = ref['root_quat_wxyz']
    measured = series_metrics(actual, base[1:], joints, q[1:], orientations, rotation(root_q[1:]), names)
    static = series_metrics(np.repeat(base[:1],len(rf),axis=0), base[1:], np.repeat(q[:1],len(rf),axis=0),
                            q[1:], rotation(np.repeat(root_q[:1],len(rf),axis=0)), rotation(root_q[1:]), names)
    forward = np.asarray(trial['goal_forward_xy'], dtype=float)
    forward /= np.linalg.norm(forward)
    return dict(rollout_sha256=sha(folder/'rollout.npz'), reference_sha256=target_sha, clock_sha256=sha(clock_file) if clock_file else None,
                native_contacts_sha256=sha(folder/'native_contacts.jsonl.gz'), source_first_and_last_evaluated_s=rf[[0,-1]].tolist(),
                source_intervals=len(rf), last_physical_time_compared_s=float(phys_time[-1]), measured=measured, held_source_first_pose=static,
                real_ball_events_and_follow=last_swing_contact(folder,t,contact,body,names,forward),
                scene=dict(ball_xyz=trial['ball_initial_xyz_m'],goal_xyz=trial['goal_xyz_m'],forward=trial['goal_forward_xy']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--after-clock',type=Path)
    a = parser.parse_args()
    with np.load(a.reference,allow_pickle=False) as r:
        ref = {k:r[k] for k in ('body_pos_w','joint_pos','root_quat_wxyz','time_s','body_names','joint_names')}
    old = evaluate(a.before, a.reference, ref, None)
    new = evaluate(a.after, a.reference, ref, a.after_clock)
    if any(not np.allclose(old['scene'][k],new['scene'][k],atol=1.e-5,rtol=0) for k in old['scene']):
        raise ValueError('Paired camera scenario differs')
    audit = dict(before=old, after=new, units='physical metres, radians or explicitly degrees; positions from original link origins',
                 definitions='50 Hz source frame ends at 0.02...4.90s; no phase alignment fitted to see outcomes. '
                             'The static negative replays the initial reference body placement and joint angles. '
                             'The relative pose removes each body root orientation and translation. '
                             'Velocity cosine pools the successive 20ms vector increments. A zero-motion signal has undefined cosine. '
                             'Follow-through compares the right ankle origin strictly after release in the fixed goal direction, also relative to root translation. '
                             'Ankle-origin movement is separate from the contact surface and the ball impulse.')
    a.output.parent.mkdir(exist_ok=True, parents=True)
    with a.output.open('x') as file:
        json.dump(audit, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
