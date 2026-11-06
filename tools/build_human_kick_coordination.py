"""Build explicit surface-marker coordination proxies from original human C3D.

No robot data, learned pose estimates, anatomical joint-center fits, force claims,
or professional/instep/contact labels are manufactured. All phase events remain
rule-based candidates; strike/contact truth is always unknown in this dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from inspect_human_kick_assets import read_cmu_c3d

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / 'configs/human_coordination_schema_v1.json'
SOCCER_CLIPS = ('10_01', '10_02', '10_03', '10_05', '10_06')


def _runs(mask):
    indices = np.flatnonzero(mask)
    return [] if not len(indices) else np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


def _fill_short_gaps(xyz, valid, times, maximum):
    filled = xyz.copy()
    imputed = np.zeros(valid.shape, dtype=bool)
    if not isinstance(maximum, int) or not 0 <= maximum <= 12:
        raise ValueError('max_gap_frames must be an integer from 0 to 12')
    for marker in range(xyz.shape[1]):
        for run in _runs(~valid[:, marker]):
            a, b = int(run[0]), int(run[-1])
            if len(run) > maximum or a == 0 or b == len(times) - 1:
                continue
            weight = (times[run] - times[a - 1]) / (times[b + 1] - times[a - 1])
            filled[run, marker] = ((1 - weight[:, None]) * xyz[a - 1, marker]
                                   + weight[:, None] * xyz[b + 1, marker])
            imputed[run, marker] = True
    return filled, valid | imputed, imputed


def _landmarks(xyz, marker_names, valid, imputed, definitions):
    index = {name: i for i, name in enumerate(marker_names)}
    names = list(definitions)
    points = np.full((len(xyz), len(names), 3), np.nan)
    okay = np.zeros(points.shape[:2], dtype=bool)
    filled = np.zeros_like(okay)
    for j, name in enumerate(names):
        dependencies = definitions[name]
        if any(n not in index for n in dependencies):
            continue
        ids = [index[n] for n in dependencies]
        mask = valid[:, ids].all(axis=1)
        points[mask, j] = xyz[mask][:, ids].mean(axis=1)
        okay[:, j] = mask
        filled[:, j] = mask & imputed[:, ids].any(axis=1)
    return np.array(names), points, okay, filled


def _pelvis_frame(xyz, marker_names, valid, imputed, definitions):
    names, p, okay, filled = _landmarks(xyz, marker_names, valid, imputed, {
        'origin': definitions['origin_markers'], 'left': definitions['left_markers'],
        'right': definitions['right_markers'], 'front': definitions['front_markers'],
        'back': definitions['back_markers']})
    y = p[:, 1] - p[:, 2]
    y_length = np.linalg.norm(y, axis=1)
    good = okay.all(axis=1) & (y_length > definitions['degenerate_epsilon_m'])
    y_unit = np.divide(y, y_length[:, None], out=np.full_like(y, np.nan), where=good[:, None])
    front = p[:, 3] - p[:, 4]
    x = front - np.einsum('ti,ti->t', front, y_unit)[:, None] * y_unit
    x_length = np.linalg.norm(x, axis=1)
    good &= x_length > definitions['degenerate_epsilon_m']
    x_unit = np.divide(x, x_length[:, None], out=np.full_like(x, np.nan), where=good[:, None])
    z_unit = np.cross(x_unit, y_unit)
    axes = np.stack([x_unit, y_unit, z_unit], axis=2)
    axes[~good] = np.nan
    return p[:, 0], axes, good, good & filled.any(axis=1)


def _differentiate(values, valid, times, imputed=None):
    derivative = np.full_like(values, np.nan, dtype=float)
    okay = np.zeros(valid.shape, dtype=bool)
    flagged = np.zeros_like(okay)
    imputed = np.zeros_like(valid) if imputed is None else imputed
    for i in range(len(times)):
        a, b = max(0, i - 1), min(len(times) - 1, i + 1)
        mask = valid[a] & valid[i] & valid[b]
        derivative[i, mask] = (values[b, mask] - values[a, mask]) / (times[b] - times[a])
        okay[i] = mask
        flagged[i] = mask & (imputed[a] | imputed[i] | imputed[b])
    return derivative, okay, flagged


def _ball_centroid(points, valid, names):
    """A fixed marker subset avoids centroid jumps when a marker disappears."""
    centroid = np.full((len(points), 3), np.nan)
    okay = np.zeros(len(points), dtype=bool)
    if points.shape[1] < 2:
        return centroid, okay, []
    order = np.argsort(-valid.sum(axis=0), kind='stable')[:2]
    if not valid[:, order].all(axis=1).any():
        return centroid, okay, []
    okay = valid[:, order].all(axis=1)
    centroid[okay] = points[okay][:, order].mean(axis=1)
    return centroid, okay, [str(names[i]) for i in order]


def _candidate(index, times, frames, method, **details):
    return {'time_s': float(times[index]), 'array_frame': int(index),
            'source_frame_number': int(frames[index]), 'confidence': 'low',
            'confidence_is_calibrated_probability': False, 'method': method,
            'is_contact_truth': False, **details}


def _first_run(mask, minimum):
    return next((run for run in _runs(mask) if len(run) >= minimum), None)


def _events(arrays):
    times, frames = arrays['time_s'], arrays['source_frame_numbers']
    names = list(arrays['landmark_names'])
    world = arrays['landmarks_world_m']
    source_valid = arrays['landmark_source_valid']
    velocity, velocity_valid, _ = _differentiate(world, source_valid, times)
    speed = np.linalg.norm(velocity, axis=2)
    normalized_velocity, normalized_valid, _ = _differentiate(
        arrays['landmarks_normalized'], arrays['landmark_normalized_source_valid'], times)
    phases = dict.fromkeys(['approach', 'support', 'backswing', 'forward_swing', 'strike', 'follow_through'])
    result = {'foot_speed_peak': None, 'candidate_kicking_side': None,
              'ball_marker_centroid_motion_onset': None, 'phases': phases,
              'unavailable_reasons': {
                  'strike': 'No calibrated ball center, source impact event or force measurement; toe speed peak is not contact.'},
              'confidence_note': 'Qualitative heuristic confidence, not a validated probability.'}
    peaks = []
    for side in ('left', 'right'):
        j = names.index(side + '_toe')
        indices = np.flatnonzero(velocity_valid[:, j])
        if len(indices):
            i = indices[np.argmax(speed[indices, j])]
            if speed[i, j] > 1.0:
                peaks.append((float(speed[i, j]), int(i), side, j))
    if peaks:
        peak_speed, peak, side, toe = max(peaks)
        result['candidate_kicking_side'] = {'side': side, 'confidence': 'low',
            'method': 'Side of fastest source-valid toe-marker peak; not source-provided kicking-side truth.'}
        result['foot_speed_peak'] = _candidate(peak, times, frames,
            'Fastest source-valid toe marker speed above 1 m/s; no contact inference.',
            side=side, toe_speed_mps=peak_speed)
        pelvis = names.index('pelvis_center')
        run = _first_run(velocity_valid[:, pelvis] & (speed[:, pelvis] > .15)
                         & (times < times[peak]), 3)
        if run is not None:
            phases['approach'] = _candidate(run[0], times, frames,
                'Pelvis surface-marker centroid translation >0.15 m/s for >=3 frames before toe peak; not certified running.',
                end_time_s=float(times[run[-1]]))
        other = 'left' if side == 'right' else 'right'
        foot_ids = [names.index(other + '_heel'), names.index(other + '_toe')]
        foot = world[:, foot_ids].mean(axis=1)
        foot_valid = source_valid[:, foot_ids].all(axis=1)
        foot_velocity, foot_velocity_valid, _ = _differentiate(foot[:, None], foot_valid[:, None], times)
        support = foot_velocity_valid[:, 0] & (np.linalg.norm(foot_velocity[:, 0], axis=1) <= .2)
        support &= (times >= times[peak] - .6) & (times <= times[peak] + .1)
        options = [run for run in _runs(support) if len(run) >= 5]
        if options:
            run = min(options, key=lambda r: float(np.min(abs(times[r] - times[peak]))))
            phases['support'] = _candidate(run[0], times, frames,
                'Contralateral heel/toe midpoint speed <=0.2 m/s for >=5 frames; no verified ground contact or force.',
                side=other, end_time_s=float(times[run[-1]]))
        local_valid = arrays['landmark_normalized_source_valid'][:, toe]
        candidates = np.flatnonzero(local_valid & (times >= times[peak] - 1.) & (times < times[peak]))
        if len(candidates) >= 3:
            forward = arrays['landmarks_normalized'][:, toe, 0]
            minimum = int(candidates[np.argmin(forward[candidates])])
            interior = minimum > candidates[0] and minimum < candidates[-1]
            neighborhood = local_valid[max(0, minimum - 1):minimum + 2].all()
            if interior and neighborhood and np.ptp(forward[candidates]) >= .03:
                phases['backswing'] = _candidate(minimum, times, frames,
                    'Interior minimum of toe pelvis-forward normalized coordinate before peak; pelvis motion can affect this proxy.', side=side)
                run = _first_run(normalized_valid[:, toe] & (normalized_velocity[:, toe, 0] > .4)
                    & (times > times[minimum]) & (times <= times[peak]), 3)
                if run is not None:
                    phases['forward_swing'] = _candidate(run[0], times, frames,
                        'Toe pelvis-forward normalized coordinate rate >0.4 /s for >=3 source-valid frames after rearward extremum.', side=side)
        decay = velocity_valid[:, toe] & (speed[:, toe] <= max(.3, .3 * peak_speed))
        decay &= (times > times[peak] + .05) & (times <= times[peak] + 1.5)
        run = _first_run(decay, 5)
        if run is not None:
            phases['follow_through'] = _candidate(run[0], times, frames,
                'Toe speed decay below max(0.3 m/s, 30% of peak) for >=5 frames; recovery candidate, not impact-normalized follow-through truth.', side=side)
    centroid, centroid_valid = arrays['ball_marker_centroid_m'], arrays['ball_centroid_valid']
    cv, cv_valid, _ = _differentiate(centroid[:, None], centroid_valid[:, None], times)
    cspeed = np.linalg.norm(cv[:, 0], axis=1)
    baseline = (times <= times[0] + .5) & centroid_valid & cv_valid[:, 0]
    if baseline.sum() >= 5:
        med_speed = float(np.median(cspeed[baseline]))
        mad_speed = float(np.median(abs(cspeed[baseline] - med_speed)))
        baseline_xyz = np.median(centroid[baseline], axis=0)
        threshold = max(.5, med_speed + 6 * mad_speed)
        displacement = np.linalg.norm(centroid - baseline_xyz, axis=1)
        if med_speed < .15:
            moving = cv_valid[:, 0] & (cspeed > threshold) & (displacement > .03)
            moving &= times > times[baseline][-1]
            run = _first_run(moving, 3)
            if run is not None:
                i = int(run[0])
                stationary = np.flatnonzero(centroid_valid & (displacement <= .01) & (times < times[i]))
                lower = float(times[stationary[-1]]) if len(stationary) else None
                result['ball_marker_centroid_motion_onset'] = _candidate(i, times, frames,
                    'Fixed two-marker centroid first sustained observed motion; onset can be delayed by missing samples.',
                    is_ball_center=False, selected_marker_names=arrays['ball_centroid_marker_names'].tolist(),
                    speed_threshold_mps=threshold, centroid_speed_mps=float(cspeed[i]),
                    uncertainty_interval_s=[lower, float(times[i])],
                    interval_is_statistical_confidence_interval=False)
    for phase, value in phases.items():
        if value is None and phase not in result['unavailable_reasons']:
            result['unavailable_reasons'][phase] = 'No source-valid sequence satisfied this conservative candidate rule.'
    return result


def build_coordination(markers_m, marker_names, marker_valid, time_s, *, source,
                       max_gap_frames=2, source_frame_numbers=None,
                       source_zero_residual_flag=None, ball_markers_m=None,
                       ball_marker_names=None, ball_marker_valid=None,
                       ball_source_zero_residual_flag=None):
    """Return arrays and metadata without treating proxy geometry as joint truth."""
    schema_bytes = SCHEMA_PATH.read_bytes()
    schema = json.loads(schema_bytes)
    if source.get('source_class') not in schema['allowed_source_classes']:
        raise ValueError('Only original optical human capture or explicit synthetic test fixtures are accepted')
    xyz = np.asarray(markers_m, dtype=float).copy()
    names = np.asarray(marker_names, dtype=str)
    valid = np.asarray(marker_valid, dtype=bool).copy()
    times = np.asarray(time_s, dtype=float)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or len(xyz) < 3:
        raise ValueError('Expected at least three frames of [T,M,3] marker positions')
    if names.shape != (xyz.shape[1],) or valid.shape != xyz.shape[:2]:
        raise ValueError('Marker names/validity shapes do not match coordinates')
    if len(set(names)) != len(names):
        raise ValueError('Marker names must be unique')
    if times.shape != (len(xyz),) or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('Time must be finite and strictly increasing')
    valid &= np.isfinite(xyz).all(axis=2)
    xyz[~valid] = np.nan
    frame_numbers = np.arange(len(xyz)) if source_frame_numbers is None else np.asarray(source_frame_numbers)
    if frame_numbers.shape != times.shape:
        raise ValueError('Source frame numbers must match times')
    zero_residual = (np.zeros_like(valid) if source_zero_residual_flag is None
                     else np.asarray(source_zero_residual_flag, dtype=bool))
    if zero_residual.shape != valid.shape:
        raise ValueError('Source residual flag shape does not match marker samples')
    if ball_markers_m is None:
        ball = np.empty((len(xyz), 0, 3))
        ball_names = np.array([], dtype=str)
        ball_valid = np.zeros((len(xyz), 0), dtype=bool)
    else:
        ball = np.asarray(ball_markers_m, dtype=float).copy()
        ball_names = np.asarray(ball_marker_names, dtype=str)
        ball_valid = np.asarray(ball_marker_valid, dtype=bool).copy()
        if ball.ndim != 3 or ball.shape[0] != len(xyz) or ball.shape[2] != 3:
            raise ValueError('Expected [T,B,3] ball-marker positions')
        if ball_valid.shape != ball.shape[:2] or ball_names.shape != (ball.shape[1],):
            raise ValueError('Ball names/validity shapes do not match coordinates')
        if len(set(ball_names)) != len(ball_names):
            raise ValueError('Ball marker names must be unique')
        ball_valid &= np.isfinite(ball).all(axis=2)
        ball[~ball_valid] = np.nan
    ball_zero = (np.zeros_like(ball_valid) if ball_source_zero_residual_flag is None
                 else np.asarray(ball_source_zero_residual_flag, dtype=bool))
    if ball_zero.shape != ball_valid.shape:
        raise ValueError('Ball residual flag shape does not match samples')
    filled, filled_valid, imputed = _fill_short_gaps(xyz, valid, times, max_gap_frames)
    landmark_names, landmarks, landmark_valid, landmark_imputed = _landmarks(
        filled, names, filled_valid, imputed, schema['landmarks'])
    _, raw_landmarks, landmark_source_valid, _ = _landmarks(
        xyz, names, valid, np.zeros_like(valid), schema['landmarks'])
    origin, axes, frame_valid, frame_imputed = _pelvis_frame(
        filled, names, filled_valid, imputed, schema['pelvis_frame'])
    _, _, frame_source_valid, _ = _pelvis_frame(
        xyz, names, valid, np.zeros_like(valid), schema['pelvis_frame'])
    local = np.einsum('tji,tnj->tni', axes, landmarks - origin[:, None])
    local_valid = landmark_valid & frame_valid[:, None]
    local_source_valid = landmark_source_valid & frame_source_valid[:, None]
    local_imputed = local_valid & (landmark_imputed | frame_imputed[:, None])
    local[~local_valid] = np.nan
    landmark_index = {name: i for i, name in enumerate(landmark_names)}
    scale_samples = []
    for side in ('left', 'right'):
        a, b, c = [landmark_index[side + '_' + n] for n in ('pelvis_side', 'knee', 'ankle')]
        scale_valid = landmark_source_valid[:, [a, b, c]].all(axis=1)
        lengths = (np.linalg.norm(raw_landmarks[:, b] - raw_landmarks[:, a], axis=1)
                   + np.linalg.norm(raw_landmarks[:, c] - raw_landmarks[:, b], axis=1))
        scale_samples.extend(lengths[scale_valid & (lengths > 1e-8)].tolist())
    scale = float(np.median(scale_samples)) if scale_samples else float('nan')
    scale_valid = np.isfinite(scale) and scale > 1e-8
    normalized = local / scale if scale_valid else np.full_like(local, np.nan)
    normalized_valid = local_valid & scale_valid
    normalized_source_valid = local_source_valid & scale_valid
    normalized_imputed = local_imputed & normalized_valid
    world_velocity, world_velocity_valid, world_velocity_imputed = _differentiate(
        landmarks, landmark_valid, times, landmark_imputed)
    normalized_velocity, normalized_velocity_valid, normalized_velocity_imputed = _differentiate(
        normalized, normalized_valid, times, normalized_imputed)
    segment_names = list(schema['segments'])
    segment_vectors = np.full((len(xyz), len(segment_names), 3), np.nan)
    segment_lengths = np.full(segment_vectors.shape[:2], np.nan)
    segment_directions = np.full_like(segment_vectors, np.nan)
    segment_valid = np.zeros(segment_vectors.shape[:2], dtype=bool)
    segment_imputed = np.zeros_like(segment_valid)
    for j, name in enumerate(segment_names):
        a, b = [landmark_index[n] for n in schema['segments'][name]]
        delta = local[:, b] - local[:, a]
        length = np.linalg.norm(delta, axis=1)
        okay = local_valid[:, a] & local_valid[:, b] & (length > 1e-8)
        segment_vectors[okay, j] = delta[okay]
        segment_lengths[okay, j] = length[okay]
        segment_directions[okay, j] = delta[okay] / length[okay, None]
        segment_valid[:, j] = okay
        segment_imputed[:, j] = okay & (local_imputed[:, a] | local_imputed[:, b])
    relations = schema['relation_pairs']
    relation_cosines = np.full((len(xyz), len(relations)), np.nan)
    relation_valid = np.zeros(relation_cosines.shape, dtype=bool)
    relation_imputed = np.zeros_like(relation_valid)
    for j, (left, right) in enumerate(relations):
        a, b = segment_names.index(left), segment_names.index(right)
        okay = segment_valid[:, a] & segment_valid[:, b]
        relation_cosines[okay, j] = np.clip(np.einsum('ti,ti->t',
            segment_directions[okay, a], segment_directions[okay, b]), -1., 1.)
        relation_valid[:, j] = okay
        relation_imputed[:, j] = okay & (segment_imputed[:, a] | segment_imputed[:, b])
    centroid, centroid_valid, centroid_names = _ball_centroid(ball, ball_valid, ball_names)
    arrays = {
        'schema_version': np.array(1), 'sourcequalification': np.array(False),
        'source_class': np.array(source['source_class']),
        'source_sha256': np.array(source.get('source_sha256', '')),
        'time_s': times, 'fps': np.array(1. / np.median(np.diff(times))),
        'source_frame_numbers': frame_numbers,
        'source_marker_names': names, 'source_markers_m': xyz,
        'source_marker_valid': valid, 'source_zero_residual_flag': zero_residual,
        'markers_m': filled, 'marker_valid': filled_valid, 'marker_interpolated': imputed,
        'landmark_names': landmark_names, 'landmarks_world_m': landmarks,
        'landmark_valid': landmark_valid, 'landmark_source_valid': landmark_source_valid,
        'landmark_interpolated': landmark_imputed,
        'pelvis_origin_m': origin, 'pelvis_axes_world': axes,
        'pelvis_frame_valid': frame_valid, 'pelvis_frame_source_valid': frame_source_valid,
        'pelvis_frame_interpolated': frame_imputed,
        'landmarks_pelvis_m': local, 'landmarks_normalized': normalized,
        'landmark_local_valid': local_valid, 'landmark_local_source_valid': local_source_valid,
        'landmark_local_interpolated': local_imputed,
        'landmark_normalized_valid': normalized_valid,
        'landmark_normalized_source_valid': normalized_source_valid,
        'landmark_normalized_interpolated': normalized_imputed,
        'landmark_velocity_world_mps': world_velocity,
        'landmark_velocity_world_valid': world_velocity_valid,
        'landmark_velocity_world_interpolated': world_velocity_imputed,
        'landmark_normalized_velocity_per_s': normalized_velocity,
        'landmark_normalized_velocity_valid': normalized_velocity_valid,
        'landmark_normalized_velocity_interpolated': normalized_velocity_imputed,
        'length_scale_m': np.array(scale), 'length_scale_valid': np.array(scale_valid),
        'segment_names': np.array(segment_names), 'segment_vectors_pelvis_m': segment_vectors,
        'segment_lengths_m': segment_lengths, 'segment_directions_pelvis': segment_directions,
        'segment_valid': segment_valid, 'segment_interpolated': segment_imputed,
        'relation_names': np.array(['__'.join(pair) for pair in relations]),
        'relation_cosines': relation_cosines, 'relation_valid': relation_valid,
        'relation_interpolated': relation_imputed,
        'ball_marker_names': ball_names, 'ball_markers_m': ball,
        'ball_marker_valid': ball_valid, 'ball_zero_residual_flag': ball_zero,
        'ball_marker_centroid_m': centroid, 'ball_centroid_valid': centroid_valid,
        'ball_centroid_marker_names': np.array(centroid_names, dtype=str),
    }
    metadata = {
        'schema_version': 1, 'sourcequalification': False,
        'representation': 'Original human surface-marker geometry proxies; not anatomical joint-center or joint-angle ground truth.',
        'source': dict(source), 'schema_path': str(SCHEMA_PATH),
        'schema_sha256': hashlib.sha256(schema_bytes).hexdigest(),
        'frames': len(xyz), 'fps_nominal': float(arrays['fps']),
        'time_steps_uniform': bool(np.allclose(np.diff(times), np.median(np.diff(times)))),
        'parameters': {'max_gap_frames': max_gap_frames, 'ball_interpolation': False,
                       'derivative': 'Central differences and one-sided endpoints; every stencil sample must be valid.'},
        'truth_labels': {'skill_level': None, 'kick_subtype': None,
                         'contact_time_s': None, 'kicking_side': None,
                         'ball_center_trajectory': None, 'ground_contact_force': None},
        'normalization': {'length_scale_m': scale if scale_valid else None,
            'valid_source_frame_side_measurements': len(scale_samples),
            'method': schema['normalization']['length_scale'],
            'anatomical_leg_length_truth': False},
        'quality': {
            'source_marker_valid_fraction': float(valid.mean()),
            'pipeline_interpolated_marker_samples': int(imputed.sum()),
            'source_zero_residual_samples': int((zero_residual & valid).sum()),
            'source_zero_residual_provenance_available': source_zero_residual_flag is not None,
            'ball_source_zero_residual_provenance_available': ball_source_zero_residual_flag is not None,
            'source_zero_residual_interpretation': schema['missing_data']['source_zero_residual'],
            'pelvis_frame_valid_fraction': float(frame_valid.mean()),
            'feature_valid_fraction': {str(name): float(relation_valid[:, i].mean())
                                      for i, name in enumerate(arrays['relation_names'])},
            'ball_centroid_valid_fraction': float(centroid_valid.mean()),
            'ball_centroid_is_ball_center': False,
        },
        'limits': [
            'Same ungraded CMU participant/session, not five independent skilled-player subjects.',
            'No instep, professional skill, contact surface, impact time or force truth is supplied.',
            'Source-present markers can include upstream modelled/filtered/gap-filled samples.',
            'New interpolation is flagged; derivatives retain stencil-level interpolation flags.',
            'Normalization uses a surface-marker chain and pelvis coordinates, not calibrated anthropometry.',
            'All phase events are heuristic candidates; unavailable phases remain null.',
        ],
    }
    metadata['events'] = _events(arrays)
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/research_human_kicks/cmu_subject10')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/research_human_kicks/coordination_v1')
    parser.add_argument('--clips', nargs='+', default=list(SOCCER_CLIPS))
    parser.add_argument('--max-gap-frames', type=int, default=2)
    args = parser.parse_args()
    if len(set(args.clips)) != len(args.clips) or any(c not in SOCCER_CLIPS for c in args.clips):
        parser.error('Only verified soccer trials 10_01, 10_02, 10_03, 10_05, 10_06 are accepted; 10_04 is walking.')
    manifest = json.loads((args.input_dir / 'download_manifest.json').read_text())
    hashes = {item['path']: item['sha256'] for item in manifest['files']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for clip in args.clips:
        path = args.input_dir / (clip + '.c3d')
        info, raw = read_cmu_c3d(path)
        if hashes.get(path.name) != info['source_sha256']:
            raise ValueError('Original source hash does not match acquisition manifest: ' + str(path))
        arrays, metadata = build_coordination(raw['human_markers_m'], raw['human_marker_names'],
            raw['human_marker_valid'], raw['time_s'],
            source={'source_class': info['source_class'], 'source_sha256': info['source_sha256'],
                    'source_path': str(path.resolve()), 'clip_id': clip, 'source_id': 'cmu_subject10_original'},
            source_frame_numbers=raw['source_frame_numbers'],
            source_zero_residual_flag=raw['human_zero_residual_flag'],
            ball_markers_m=raw['ball_markers_m'], ball_marker_names=raw['ball_marker_names'],
            ball_marker_valid=raw['ball_marker_valid'],
            ball_source_zero_residual_flag=raw['ball_zero_residual_flag'],
            max_gap_frames=args.max_gap_frames)
        target = args.output_dir / (clip + '_coordination_v1.npz')
        np.savez_compressed(target, **arrays)
        metadata['output_npz'] = {'path': str(target.resolve()),
                                 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        metadata['builder_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report = target.with_suffix('.json')
        report.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        summary = {'clip_id': clip, 'frames': len(arrays['time_s']),
            'source_sha256': info['source_sha256'], 'sourcequalification': False,
            'npz': str(target.resolve()), 'npz_sha256': metadata['output_npz']['sha256'],
            'json': str(report.resolve()), 'json_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
            'length_scale_m': metadata['normalization']['length_scale_m'],
            'source_valid_fraction': metadata['quality']['source_marker_valid_fraction'],
            'interpolated_marker_samples': metadata['quality']['pipeline_interpolated_marker_samples'],
            'source_zero_residual_samples': metadata['quality']['source_zero_residual_samples'],
            'candidate_kicking_side': metadata['events']['candidate_kicking_side'],
            'candidate_phase_names': [k for k, v in metadata['events']['phases'].items() if v is not None],
            'ball_motion_candidate': metadata['events']['ball_marker_centroid_motion_onset'],
            'contact_truth': None}
        summaries.append(summary)
        print(json.dumps({'clip_id': clip, 'frames': summary['frames'],
                          'length_scale_m': summary['length_scale_m'],
                          'candidate_phase_names': summary['candidate_phase_names'],
                          'contact_truth': None}), flush=True)
    aggregate = {'schema_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'sourcequalification': False, 'clip_count': len(summaries),
        'total_frames': sum(x['frames'] for x in summaries),
        'source_subject_count': 1, 'professional_population_statistics': None,
        'schema_sha256': hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
        'clips': summaries}
    (args.output_dir / 'manifest.json').write_text(json.dumps(aggregate, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
