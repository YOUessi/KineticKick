"""Localize CMU 10_02 balance loss and reconstruct a source-aligned ball scenario."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path('/home/you/projects/kinetickick')
FEET = ['left_ankle_roll_link', 'right_ankle_roll_link']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(q):
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def xyz(element, key):
    return np.fromstring(element.get(key, '0 0 0') if element is not None else '0 0 0', sep=' ')


def aggregate_mass(urdf, names):
    """Fold unrecorded fixed child inertias into their closest recorded parent."""
    fixed = {j.find('child').get('link'): j for j in urdf.findall("joint[@type='fixed']")}
    masses = np.zeros(len(names))
    moments = np.zeros((len(names), 3))
    for link in urdf.findall('link'):
        inertia = link.find('inertial')
        if inertia is None:
            continue
        name = link.get('name')
        mass = float(inertia.find('mass').get('value'))
        point = xyz(inertia.find('origin'), 'xyz')
        while name not in names:
            joint = fixed[name]
            origin = joint.find('origin')
            point = xyz(origin, 'xyz') + Rotation.from_euler('xyz', xyz(origin, 'rpy')).apply(point)
            name = joint.find('parent').get('link')
        index = names.index(name)
        masses[index] += mass
        moments[index] += mass * point
    return masses, moments


def center_of_mass(position, quat, masses, moments):
    r = rotation(quat.reshape(-1, 4)).as_matrix().reshape(*quat.shape[:-1], 3, 3)
    return ((position * masses[None, :, None]).sum(1) + np.einsum('tbij,bj->ti', r, moments)) / masses.sum()


def loaded_y_edges(rows, dt):
    ys = []
    for pair in rows:
        if 'ground' not in pair['actors'] or not any(foot in pair['actors'] for foot in FEET):
            continue
        for p in pair['points']:
            if abs(np.dot(p[3:6], p[6:9]) * p[5]) / dt > 2.0:
                ys.append(p[1])
    return (min(ys), max(ys)) if ys else (None, None)


def events(t, mask, minimum):
    bins = np.flatnonzero(mask)
    runs = np.split(bins, np.flatnonzero(np.diff(bins) > 1) + 1)
    return [(float(t[r[0]]), float(t[r[-1]])) for r in runs if len(r) >= minimum]


def summarize(urdf, base, kind, reference):
    folder = base / f'physics_{kind}_10_02'
    with np.load(reference, allow_pickle=False) as ref, np.load(folder / 'rollout.npz', allow_pickle=False) as measured:
        report = json.loads((folder / 'tracking_result.json').read_text())
        trial = json.loads((folder / 'trial.json').read_text())
        if sha(reference) != trial['external_reference_sha256']:
            raise ValueError('Reference does not match rollout provenance')
        hashes = report['local_artifact_sha256']
        for name, digest in hashes.items():
            if sha(folder / name) != digest:
                raise ValueError(f'Stale rollout artifact {name}')
        t = measured['time_s']
        dt = float(t[0])
        rf = ref['time_s']
        if not np.array_equal(t, np.arange(1, 2801) * .005):
            raise ValueError('A full identical 200 Hz episode is required')
        names = ref['body_names'].tolist()
        mass, moment = aggregate_mass(urdf, names)
        com = center_of_mass(measured['body_position_w_m'], measured['body_quat_wxyz'], mass, moment)
        reference_com = center_of_mass(ref['body_pos_w'], ref['body_quat_w'], mass, moment)
        load = measured['foot_ground_normal_load_N']
        with gzip.open(folder/'native_contacts.jsonl.gz', 'rt') as stream:
            edges = {}
            for line in stream:
                tick, rows = json.loads(line)
                if tick > 400:
                    break
                edges[tick] = loaded_y_edges(rows, dt)
        root = measured['root_pose_wxyz']
        src_xy = np.stack([np.interp(t, rf, ref['root_pos_w'][:, j]) for j in (0, 1)], axis=1)
        dist = np.linalg.norm(root[:, :2] - src_xy, axis=1)
        snapshots = []
        for sec in (.10, .20, .30, .40, .50, .60, .70, 1.0, 1.4):
            ri = round(sec / .02)
            tick = round(sec / dt)
            idx = tick - 1
            rerr = rotation(ref['root_quat_wxyz'][ri]).inv() * rotation(root[idx, 3:])
            ymin, ymax = edges[tick]
            # Lateral-only signed margin to native loaded points. Does not imply a dynamic fall threshold.
            margin = min(com[idx, 1] - ymin, ymax - com[idx, 1]) if ymin is not None else None
            snapshots.append(dict(t_s=sec, load_LR_N=load[idx].tolist(), ref_low_quiet_LR=ref['source_low_quiet_foot_proxy'][ri].tolist(),
                                  root_minus_ref_xyz_m=(root[idx, :3] - ref['root_pos_w'][ri]).tolist(),
                                  root_geodesic_error_deg=float(np.rad2deg(rerr.magnitude())),
                                  actual_com_xyz_m=com[idx].tolist(), reference_com_xyz_m=reference_com[ri].tolist(),
                                  actual_native_support_y_range_m=[ymin, ymax], lateral_com_margin_to_loaded_points_m=margin))
        err = measured['policy_commands_time_frame_action_target'][0, -29:] - ref['joint_pos'][0]
        return dict(reference_sha256=sha(reference), rollout_sha256=hashes['rollout.npz'], native_contacts_sha256=hashes['native_contacts.jsonl.gz'],
                    total_urdf_mass_kg=float(mass.sum()), source_first_nondouble_quiet_s=float(rf[np.flatnonzero(~ref['source_low_quiet_foot_proxy'].all(axis=1))[0]]),
                    first_sustained_0p1s_left_loaded_right_unloaded_first_last_sample_s=events(t, (load[:, 0] > 30) & (load[:, 1] < 2), 20)[0],
                    root_xy_err_10cm_first_s=float(t[np.flatnonzero(dist > .10)[0]]),
                    first_low_pelvis_s=report['first_height_below_0p45m_s'], first_tilt_60deg_s=report['first_tilt_over_60deg_s'],
                    reference_start_command_delta_rad=dict(zip(ref['joint_names'].tolist(), err.tolist())), snapshots=snapshots)


def ball_gap(urdf, ref, center, radius):
    names = ref['body_names'].tolist()
    result = {}
    # The source-informed interval covers the rapid foot swing surrounding the optical launch cue.
    phase = (ref['time_s'] >= 2.70) & (ref['time_s'] <= 3.04)
    for foot in FEET:
        index = names.index(foot)
        into_foot = rotation(ref['body_quat_w'][:, index]).inv()
        local = into_foot.apply(center - ref['body_pos_w'][:, index])
        gaps = []
        for shape in urdf.find(f"link[@name='{foot}']").findall('collision'):
            cyl = shape.find('geometry/cylinder')
            origin = shape.find('origin')
            direction = Rotation.from_euler('xyz', xyz(origin, 'rpy')).as_matrix()[:, 2]
            middle = xyz(origin, 'xyz')
            half = .5 * float(cyl.get('length'))
            offset = np.clip((local - middle) @ direction, -half, half)
            gaps.append(np.linalg.norm(local - middle - offset[:, None] * direction, axis=1) - radius - float(cyl.get('radius')))
        gap = np.stack(gaps).min(axis=0)
        best = np.flatnonzero(phase)[np.argmin(gap[phase])]
        intersect = np.flatnonzero(phase & (gap <= 0))
        result[foot] = dict(min_signed_capsule_sphere_gap_m=float(gap[best]), closest_source_time_s=float(ref['time_s'][best]),
                            first_intersection_frame_s=float(ref['time_s'][intersect[0]]) if intersect.size else None)
    return result


def source_scenario(urdf, repo, assets):
    protocol = json.loads((repo/'configs/v160_tracking_protocol.json').read_text())
    directory = assets/'data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050'
    source = directory / '10_02.npz'
    manifest = json.loads((repo/'experiments/v160_human/manifest.json').read_text())['clips']['10_02']
    audit = next(row for row in json.loads((directory/'quality_audit.json').read_text())['motions'] if row['id'] == '10_02')
    if sha(source) != manifest['source_processed_sha256'] or sha(source) != audit['derived_npz_sha256']:
        raise ValueError('Source is not the qualified optical marker file')
    with np.load(source) as human, np.load(repo/'experiments/v160_human/10_02.npz') as original, np.load(repo/'experiments/v162_human/armclear/10_02.npz') as ref:
        names = original['body_names'].tolist()
        landmark = human['primary_landmark_names'].tolist()
        source_hips = human['primary_human_3d_m'][0, [landmark.index(k) for k in ('LHJ0', 'RHJ0')]].mean(0)
        robot_hips = original['body_pos_w'][0, [names.index(n) for n in ('left_hip_pitch_link', 'right_hip_pitch_link')]].mean(0)
        scale = manifest['geometry']['map_scale']
        markers = scale * (human['uncalibrated_ball_marker_cluster_centroid_m'] - source_hips) + robot_hips
        impact = audit['inferred_event']['inferred_ball_launch_onset_frame']
        pre = slice(impact - 53, impact - 5)
        if not human['three_ball_marker_valid'][pre].all() or not human['three_ball_marker_valid'][impact:impact + 7].all():
            raise ValueError('Ball mapping cannot use occluded markers')
        centroid = np.median(markers[pre], axis=0)
        forward = markers[impact + 6, :2] - markers[impact, :2]
        forward /= np.linalg.norm(forward)
        center = np.r_[centroid[:2], protocol['ball_radius_m']]
        old = np.array(protocol['ball_initial_xyz_m'])
        return dict(source_processed_sha256=sha(source), marker_launch_source_s=float(human['time_s'][impact]),
                    marker_launch_reference_time_s=float(np.interp(human['time_s'][impact], ref['source_time_s'], ref['time_s'])),
                    nearest_foot_proxy=audit['inferred_event']['nearest_foot_side'], marker_stationary_window_frames=[pre.start, pre.stop - 1],
                    marker_centroid_world_m=centroid.tolist(), source_scale=scale, source_ball_world_xy_unit=forward.tolist(),
                    registered_ground_ball_xyz_m=center.tolist(), legacy_ball_xyz_m=old.tolist(), old_to_registered_ball_xy_distance_m=float(np.linalg.norm(center[:2]-old[:2])),
                    old_goal_world_xy_unit=protocol['goal_forward_xy'], reference_sampled_swept_gap_registered=ball_gap(urdf, ref, center, protocol['ball_radius_m']),
                    reference_sampled_swept_gap_legacy=ball_gap(urdf, ref, old, protocol['ball_radius_m']),
                    note='Ball cluster is uncalibrated and used as a planar placement proxy. Capsule overlaps at 50 Hz are reference geometry and cannot certify physical contact, surface or impact time.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--assets', type=Path, default=ASSETS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    urdf = ET.parse(args.assets/'configs/paid_original/paid_main_resolved.urdf')
    base = args.repo/'experiments/v162_human'
    refs = {'old': args.repo/'experiments/v160_human/10_02.npz', 'support': base/'references/10_02.npz', 'armclear': base/'armclear/10_02.npz'}
    result = dict(episodes={k:summarize(urdf, base, k, ref) for k, ref in refs.items()}, source_scenario=source_scenario(urdf, args.repo, args.assets))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
