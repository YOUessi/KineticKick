"""Export one pinned professional-match reconstructed3D shot for coarse geometry.

SkillCorner's original publication files contain model estimates, not raw motion
capture. Error budgets below are conditional sensitivity calculations using the
vendor's predicted errors as radii, never calibrated confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/skillcorner_coordination_v1.json'


def _config():
    return json.loads(CONFIG_PATH.read_text())


def _time(value):
    pieces = str(value).split(':')
    if len(pieces) not in (2, 3):
        raise ValueError('Expected mm:ss or hh:mm:ss timestamp')
    try:
        parts = [float(p) for p in pieces]
    except ValueError as exc:
        raise ValueError('Invalid timestamp') from exc
    if not np.isfinite(parts).all() or any(p < 0 for p in parts) or parts[-1] >= 60:
        raise ValueError('Invalid timestamp')
    if len(parts) == 3 and parts[1] >= 60:
        raise ValueError('Invalid timestamp minutes')
    return sum(p * 60**i for i, p in enumerate(reversed(parts)))


def _number(value):
    return np.nan if value is None else float(value)


def _summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return dict(zip(('min', 'median', 'p95', 'max'), np.quantile(values, [0, .5, .95, 1]).tolist()))


def verify_sources(source_dir, config):
    """Check tracked pins rather than trusting a mutable local download manifest."""
    verified = {}
    for name, expected in config['input_files'].items():
        path = Path(source_dir) / name
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != expected['sha256']:
            raise ValueError(f'Source hash mismatch: {name}')
        if 'size_bytes' in expected and len(blob) != expected['size_bytes']:
            raise ValueError(f'Source size mismatch: {name}')
        verified[name] = {**expected, 'sha256': digest, 'size_bytes': len(blob)}
    return verified


def load_inputs(source_dir):
    """Load the single pinned sample, its matched event and identity metadata."""
    source_dir = Path(source_dir)
    config = _config()
    verified = verify_sources(source_dir, config)
    files = config['sample']
    with gzip.open(source_dir / files['pose_file'], 'rt') as handle:
        frames = []
        for line in handle:
            if len(frames) >= config['max_frames']:
                raise ValueError('Bounded sample frame limit exceeded')
            frames.append(json.loads(line))
    match = json.loads((source_dir / files['match_file']).read_text())
    with (source_dir / files['events_file']).open() as handle:
        selected = [row for row in csv.DictReader(handle) if row['event_id'] == files['event_id']]
    if len(selected) != 1:
        raise ValueError('Expected exactly one pinned event')
    event = selected[0]
    if int(match['id']) != files['match_id'] or int(event['player_id']) != files['player_id']:
        raise ValueError('Pinned match/player identity mismatch')
    with (source_dir / files['phases_file']).open() as handle:
        phases = [row for row in csv.DictReader(handle) if row['index'] == event['phase_index']]
    if len(phases) != 1 or not int(phases[0]['frame_start']) <= int(event['frame_end']) <= int(phases[0]['frame_end']):
        raise ValueError('Event does not resolve to the pinned current phase file')
    source = {
        'source_class': 'broadcast_video_pose_estimate', 'coordinate_units': 'm',
        'source_sha256': verified[files['pose_file']]['sha256'], 'verified_input_files': verified,
        'github_revision': config['github_revision'], 'hf_revision': config['hf_revision'],
        'professional_participation_verified': True,
        'professional_participation_evidence': config['professional_participation_evidence'],
        'data_license': config['data_license'],
        'sample_filename_phase_index': 406, 'matched_phase_index': int(phases[0]['index']),
        'known_conflicts': config['known_conflicts'],
    }
    return frames, match, event, source


def _align_event(frames, times, periods, event, match):
    if str(event['match_id']) != str(match['id']):
        raise ValueError('Event match does not match metadata')
    if event.get('event_type') != 'player_possession' or event.get('end_type') != 'shot':
        raise ValueError('Expected an explicit vendor player-possession shot event')
    if not np.all(periods == int(event['period'])):
        raise ValueError('Event period does not match this single-period sample')
    fractional = float(event['frame_end']) * 2.5
    if fractional < frames[0] or fractional > frames[-1]:
        raise ValueError('Event anchor lies outside the pose sample')
    event_time = _time(event['time_end'])
    interpolated_time = float(np.interp(fractional, frames, times))
    residual = interpolated_time - event_time
    if abs(residual) > 1e-6:
        raise ValueError('Event frame/time association disagrees with pose timestamps')
    return {
        'pose_frame_fractional': fractional,
        'bracketing_pose_frames': [int(np.floor(fractional)), int(np.ceil(fractional))],
        'event_match_time_s': event_time, 'event_clip_time_s': event_time - float(times[0]),
        'time_residual_s': residual, 'pose_grid_s': .04, 'event_grid_s': .1,
        'formula': 'pose_frame = 2.5 * tracking_frame; checked against match-time timestamps',
        'is_measured_impact': False,
        'timing_resolution_is_accuracy_bound': False,
    }


def _geometry(xyz, error_m, valid, error_valid, quality, names, config):
    index = {name: i for i, name in enumerate(names)}
    origin = index['midHip']
    relative_valid = valid & valid[:, origin, None]
    relative = xyz - xyz[:, origin:origin+1]
    relative[~relative_valid] = np.nan
    relative_error_valid = relative_valid & error_valid & error_valid[:, origin, None]
    relative_budget = error_m + error_m[:, origin, None]
    relative_budget[relative_error_valid[:, origin], origin] = 0.0  # same variable cancels exactly
    relative_budget[~relative_error_valid] = np.nan

    seg_names = list(config['segments'])
    endpoints = np.array([[index[a], index[b]] for a, b in config['segments'].values()])
    a, b = endpoints[:, 0], endpoints[:, 1]
    vectors = xyz[:, b] - xyz[:, a]
    lengths = np.linalg.norm(vectors, axis=2)
    seg_valid = valid[:, a] & valid[:, b] & (lengths > 1e-8)
    vectors[~seg_valid] = np.nan
    lengths[~seg_valid] = np.nan
    directions = np.divide(vectors, lengths[..., None], out=np.full_like(vectors, np.nan), where=seg_valid[..., None])
    seg_error_valid = seg_valid & error_valid[:, a] & error_valid[:, b]
    vector_budget = error_m[:, a] + error_m[:, b]
    vector_budget[~seg_error_valid] = np.nan
    # If an estimated vector of length L can move within radius e<L, its
    # direction turns by at most asin(e/L). e>=L permits arbitrary direction.
    ratio = np.divide(vector_budget, lengths, out=np.full_like(lengths, np.nan), where=seg_error_valid)
    direction_budget = np.where(ratio < 1, 2*np.sin(.5*np.arcsin(np.minimum(ratio, 1))), 2.)
    direction_budget[~seg_error_valid] = np.nan
    segment_quality = seg_error_valid & quality[:, a] & quality[:, b]

    relation_names = list(config['relations'])
    seg_index = {name: i for i, name in enumerate(seg_names)}
    pairs = np.array([[seg_index[a], seg_index[b]] for a, b in config['relations'].values()])
    a, b = pairs[:, 0], pairs[:, 1]
    relation_valid = seg_valid[:, a] & seg_valid[:, b]
    cosines = np.sum(directions[:, a] * directions[:, b], axis=2)
    cosines = np.clip(cosines, -1, 1)
    cosines[~relation_valid] = np.nan
    relation_error_valid = seg_error_valid[:, a] & seg_error_valid[:, b]
    cosine_budget = np.minimum(2, direction_budget[:, a] + direction_budget[:, b])
    cosine_budget[~relation_error_valid] = np.nan
    relation_quality = segment_quality[:, a] & segment_quality[:, b]
    relation_quality &= cosine_budget <= config['quality_policy']['max_relation_cosine_error_budget']

    # A robust sample normalizer, never a measured anatomical leg length.
    leg = [seg_index[n] for n in ('left_thigh','left_shank','right_thigh','right_shank')]
    scale_frames = segment_quality[:, leg].all(axis=1)
    scale_valid = int(scale_frames.sum()) >= config['quality_policy']['min_scale_frames']
    scale = scale_budget = np.nan
    if scale_valid:
        per_frame = lengths[:, leg].sum(axis=1)/2
        per_frame_budget = vector_budget[:, leg].sum(axis=1)/2
        scale = float(np.median(per_frame[scale_frames]))
        lower = float(np.median((per_frame-per_frame_budget)[scale_frames]))
        upper = float(np.median((per_frame+per_frame_budget)[scale_frames]))
        scale_budget = max(scale-lower, upper-scale)
        scale_valid = bool(np.isfinite(scale) and scale > 1e-8)
    normalized = np.full_like(relative, np.nan)
    normalized_valid = relative_valid & scale_valid
    if scale_valid:
        normalized[normalized_valid] = relative[normalized_valid]/scale
    normalized_budget = np.full_like(relative_budget, np.nan)
    normalized_error_valid = relative_error_valid & scale_valid & (scale > scale_budget)
    if scale_valid and scale > scale_budget:
        numerator = relative_budget + np.linalg.norm(relative, axis=2)*scale_budget/scale
        normalized_budget[normalized_error_valid] = numerator[normalized_error_valid]/(scale-scale_budget)
    return {
        'joints_pelvis_relative_m': relative, 'pelvis_relative_valid': relative_valid,
        'pelvis_relative_error_budget_m': relative_budget, 'pelvis_relative_error_valid': relative_error_valid,
        'joints_normalized': normalized, 'normalized_valid': normalized_valid,
        'normalized_position_error_budget': normalized_budget, 'normalized_error_valid': normalized_error_valid,
        'length_scale_m': np.array(scale), 'length_scale_error_budget_m': np.array(scale_budget),
        'length_scale_valid': np.array(scale_valid), 'length_scale_frame_eligible': scale_frames,
        'segment_names': np.array(seg_names), 'segment_vectors_source_axes_m': vectors,
        'segment_lengths_m': lengths, 'segment_directions_source_axes': directions,
        'segment_valid': seg_valid, 'segment_vector_error_budget_m': vector_budget,
        'segment_direction_l2_error_budget': direction_budget, 'segment_error_valid': seg_error_valid,
        'segment_quality_pass': segment_quality, 'relation_names': np.array(relation_names),
        'relation_cosines': cosines, 'relation_valid': relation_valid,
        'relation_cosine_error_budget': cosine_budget, 'relation_error_valid': relation_error_valid,
        'relation_quality_pass': relation_quality,
    }


def build_coordination(records, match, event, *, source):
    """Return array+metadata contract for one continuous shot, without gap filling."""
    config = _config()
    if source.get('source_class') not in ('broadcast_video_pose_estimate', 'synthetic_test_fixture'):
        raise ValueError('Source must be reconstructed3D or an explicit synthetic test fixture')
    if source.get('coordinate_units', 'm') != 'm':
        raise ValueError('This adapter requires source coordinates in metres')
    if not 3 <= len(records) <= config['max_frames']:
        raise ValueError('Expected a bounded continuous sample of at least three frames')
    frames = np.array([r['frame'] for r in records])
    if frames.dtype.kind not in 'iu' or not np.all(np.diff(frames) == 1):
        raise ValueError('Pose frames must be consecutive integer frame numbers')
    times = np.array([_time(r['timestamp']) for r in records])
    if not np.allclose(np.diff(times), .04, atol=1e-7, rtol=0):
        raise ValueError('Pose timestamp spacing must match 25 Hz')
    periods = np.array([r['period'] for r in records], dtype=int)
    alignment = _align_event(frames, times, periods, event, match)
    player_id = int(event['player_id'])
    matches = [p for p in match['players'] if int(p['id']) == player_id]
    if len(matches) != 1:
        raise ValueError('Expected a unique matching player identity')
    player = matches[0]
    names = config['joint_names']
    xyz = np.full((len(records), len(names), 3), np.nan)
    error_cm = np.full(xyz.shape[:2], np.nan)
    present = np.zeros(xyz.shape[:2], dtype=bool)
    detected = np.zeros(len(records), dtype=bool)
    detection_available = np.zeros(len(records), dtype=bool)
    xy = np.full((len(records), 2), np.nan)
    ball = np.full((len(records), 3), np.nan)
    ball_detected = np.zeros(len(records), dtype=bool)
    ball_detection_available = np.zeros(len(records), dtype=bool)
    possession_available = np.zeros(len(records), dtype=bool)
    possession_shooter = np.zeros(len(records), dtype=bool)
    for i, frame in enumerate(records):
        rows = [p for p in frame['player_data'] if int(p['player_id']) == player_id]
        if len(rows) > 1:
            raise ValueError('A frame has duplicate player ids')
        if rows:
            row = rows[0]
            xy[i] = [_number(row.get(k)) for k in ('x', 'y')]
            detection_available[i] = isinstance(row.get('is_detected'), bool)
            detected[i] = row.get('is_detected') is True
            joints = row.get('joints') or {}
            for j, name in enumerate(names):
                point = joints.get(name)
                if point is None:
                    continue
                present[i, j] = True
                coordinates = point.get('xyz')
                if coordinates is not None:
                    if len(coordinates) != 3:
                        raise ValueError('Joint xyz must contain exactly three coordinates')
                    xyz[i, j] = [_number(v) for v in coordinates]
                error_cm[i, j] = _number(point.get('p90_mae_cm'))
        b = frame.get('ball_data') or {}
        ball[i] = [_number(b.get(k)) for k in ('x','y','z')]
        ball_detection_available[i] = isinstance(b.get('is_detected'), bool)
        ball_detected[i] = b.get('is_detected') is True
        possession = (frame.get('possession') or {}).get('player_id')
        possession_available[i] = possession is not None
        possession_shooter[i] = possession == player_id
    valid = present & np.isfinite(xyz).all(axis=2)
    error_valid = np.isfinite(error_cm) & (error_cm >= 0)
    error_m = error_cm/100
    quality = valid & error_valid & detected[:, None]
    quality &= error_m <= config['quality_policy']['max_joint_predicted_p90_m']
    arrays = {
        'source_frame_numbers': frames.astype(np.int64), 'match_time_s': times,
        'clip_time_s': times-times[0], 'period': periods, 'pose_fps': np.array(25.),
        'tracking_fps': np.array(10.), 'joint_names': np.array(names),
        'joints_reconstructed_m': xyz, 'joint_predicted_p90_cm': error_cm,
        'joint_predicted_p90_m': error_m, 'joint_present': present,
        'joint_geometry_valid': valid, 'joint_error_valid': error_valid, 'joint_quality_pass': quality,
        'player_is_detected': detected, 'player_detection_flag_available': detection_available,
        'player_xy_resampled_m': xy, 'player_xy_valid': np.isfinite(xy).all(axis=1),
        'tracking_xy_resampled_10_to_25hz': np.ones(len(records), dtype=bool),
        'exact_even_tracking_frame_alignment': frames % 5 == 0,
        'pipeline_joint_interpolated': np.zeros_like(valid),
        'ball_reconstructed_xyz_m': ball, 'ball_geometry_valid': np.isfinite(ball).all(axis=1),
        'ball_is_detected': ball_detected, 'ball_detection_flag_available': ball_detection_available,
        'possession_player_id_available': possession_available, 'possession_is_shooter': possession_shooter,
    }
    arrays.update(_geometry(xyz,error_m,valid,error_valid,quality,names,config))
    shot_window = np.abs(times-alignment['event_match_time_s']) <= 1.
    quality_by_relation = {str(n): int(arrays['relation_quality_pass'][:,i].sum()) for i,n in enumerate(arrays['relation_names'])}
    metadata = {
        'schema_version': 1, 'schema_name': config['name'], 'source': source,
        'source_class': source['source_class'], 'source_sha256': source['source_sha256'],
        'config_sha256': hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'coordinate_frame': 'midHip-relative translation; source axes retained (pitch x/y, player-relative z); no anatomical pelvis rotation or floor.',
        'representation': 'Vendor-estimated human keypoint geometry; no measured anatomical joint-center or joint-angle truth.',
        'qualification': {
            'professional_participation': {'verified': bool(source.get('professional_participation_verified', False)),
                                           'evidence': source.get('professional_participation_evidence'), 'scope': 'Match participation, not optimal technique or maximal effort.'},
            'shot_event': {'verified': True, 'basis': 'Explicit vendor player-possession end_type=shot; not independently video-adjudicated by this adapter.'},
            'non_header': {'verified': event.get('is_header') == 'False', 'basis': 'Vendor is_header annotation; does not establish instep.'},
            'instep': {'verified': False, 'status': 'unknown'},
            'kick_strength': {'verified': False, 'status': 'unknown'},
            'precise_contact': {'verified': False, 'status': 'unknown'},
            'independent_metric_ground_truth': {'verified': False, 'status': 'not available'},
            'p2_complete': False,
        },
        'truth_labels': {'instep': None, 'kick_strength': None, 'kick_side': None,
                         'contact_time_s': None, 'support_contact': None, 'ball_center_calibrated': None},
        'match_id': match['id'], 'player': player, 'event': event, 'event_alignment': alignment,
        'field_semantics': config['field_semantics'], 'uncertainty': config['uncertainty'],
        'quality_policy': config['quality_policy'],
        'normalization': {'method': 'Median of per-frame mean left/right estimated hip-knee-ankle chains, using endpoint quality-pass frames.',
                          'is_measured_anthropometry': False,
                          'scale_error': 'Radius from medians of per-frame lower/upper chain envelopes; eligibility held fixed for sensitivity calculation.'},
        'counts': {'frames': len(records), 'joints': len(names), 'geometry_valid_joint_frames': int(valid.sum()),
                   'quality_pass_joint_frames': int(quality.sum()), 'player_detected_frames': int(detected.sum()),
                   'ball_detected_frames': int(ball_detected.sum()), 'pipeline_interpolated_joint_frames': 0,
                   'relation_valid_frames_total': int(arrays['relation_valid'].sum()),
                   'relation_quality_pass_frames_total': int(arrays['relation_quality_pass'].sum()),
                   'relation_quality_pass_frames_by_name': quality_by_relation},
        'quality_summary': {'all_joint_predicted_p90_cm': _summary(error_cm),
                            'shot_plus_minus_1s_joint_predicted_p90_cm': _summary(error_cm[shot_window]),
                            'shot_plus_minus_1s_frames': int(shot_window.sum()),
                            'shot_plus_minus_1s_relation_quality_pass_by_name': {
                                str(n): int(arrays['relation_quality_pass'][shot_window,i].sum())
                                for i,n in enumerate(arrays['relation_names'])},
                            'scale_m': _summary(arrays['length_scale_m']),
                            'scale_error_budget_m': _summary(arrays['length_scale_error_budget_m'])},
        'limitations': config['limitations'],
    }
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT/'data/research_human_kicks/skillcorner_probe_20261107')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/research_human_kicks/skillcorner_coordination_v1')
    args = parser.parse_args()
    records, match, event, source = load_inputs(args.source_dir)
    arrays, metadata = build_coordination(records,match,event,source=source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"skillcorner_{match['id']}_{event['event_id']}_{event['player_id']}"
    npz = args.output_dir/(stem+'.npz')
    info = args.output_dir/(stem+'.json')
    np.savez_compressed(npz, **arrays)
    metadata['output_npz_sha256'] = hashlib.sha256(npz.read_bytes()).hexdigest()
    info.write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    manifest = {'schema_version':1, 'outputs':[{'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size} for p in [npz,info]],
                'source_sha256': source['source_sha256'], 'source_hashes_verified': len(source['verified_input_files']),
                'frames':len(records), 'qualification': metadata['qualification']}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'npz':str(npz),'json':str(info),'counts':metadata['counts'],'quality_summary':metadata['quality_summary']},ensure_ascii=False))


if __name__ == '__main__':
    main()
