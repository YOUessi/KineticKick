#!/usr/bin/env python3
"""Independent CPU ball momentum audit. A closed balance is not task acceptance.

The fixed 0.02 N s tolerance is the prior probe's engineering impulse bound,
not a hardware accuracy statement. Unexplained impulse is never called friction.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from strike_baseline_metrics import first_force_episode, task_outcome, validated_rows
from strike_dynamics_telemetry import validated_force_probe

ROOT = Path(__file__).resolve().parents[1]
DT = .005
WINDOW = (1.45, 1.72)
TOLERANCE_NS = .02
PAIRS = ('right_foot', 'left_foot', 'other_robot', 'ground')
RECONSTRUCTION_ATOL_N = 1e-3  # Raw float32 reductions only, not physical closure.


def vector(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Finite world-coordinate vector of length 3 required')
    return result


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite scalar required')
    return float(value)


def heading(value):
    v = vector(value)
    return math.degrees(math.atan2(v[0], -v[1])) if np.linalg.norm(v[:2]) > 1e-12 else None


def heading_change(a, b):
    a, b = heading(a), heading(b)
    return None if a is None or b is None else (b-a+180.) % 360.-180.


def _balance(rows, mass, gravity):
    pair_impulses = {name: {kind: sum((vector(r['pairs'][name][kind]) *
        (r['step_end_s']-r['step_start_s']) for r in rows), np.zeros(3))
        for kind in ('normal', 'tangent')} for name in PAIRS}
    normal = sum((p['normal'] for p in pair_impulses.values()), np.zeros(3))
    tangent = sum((p['tangent'] for p in pair_impulses.values()), np.zeros(3))
    duration = rows[-1]['step_end_s']-rows[0]['step_start_s']
    gravity_impulse = mass*gravity*duration
    before, after = vector(rows[0]['velocity_before']), vector(rows[-1]['velocity_after'])
    momentum = mass*(after-before)
    predicted = normal+tangent+gravity_impulse
    residual = momentum-predicted
    foot_normal = pair_impulses['right_foot']['normal']+pair_impulses['left_foot']['normal']
    foot_tangent = pair_impulses['right_foot']['tangent']+pair_impulses['left_foot']['tangent']
    normal_velocity = before+(normal+gravity_impulse)/mass
    complete_velocity = before+predicted/mass
    return dict(step_start_s=rows[0]['step_start_s'], step_end_s=rows[-1]['step_end_s'],
        bin_count=len(rows), duration_s=duration, velocity_before_w_mps=before.tolist(),
        velocity_after_w_mps=after.tolist(), momentum_change_Ns=momentum.tolist(),
        pair_impulses_Ns={name: {kind: value.tolist() for kind, value in p.items()} for name, p in pair_impulses.items()},
        normal_impulse_Ns=normal.tolist(), tangent_impulse_Ns=tangent.tolist(),
        gravity_impulse_Ns=gravity_impulse.tolist(), accounted_impulse_Ns=predicted.tolist(),
        residual_Ns=residual.tolist(), residual_norm_Ns=float(np.linalg.norm(residual)),
        passed=bool(np.linalg.norm(residual) <= TOLERANCE_NS),
        normal_and_gravity_only_end_velocity_w_mps=normal_velocity.tolist(),
        measured_contact_and_gravity_end_velocity_w_mps=complete_velocity.tolist(),
        heading=dict(normal_impulse_deg=heading(normal), tangent_impulse_deg=heading(tangent),
            contact_total_impulse_deg=heading(normal+tangent),
            normal_to_contact_total_change_deg=heading_change(normal, normal+tangent),
            foot_normal_impulse_deg=heading(foot_normal), foot_tangent_impulse_deg=heading(foot_tangent),
            foot_normal_to_total_change_deg=heading_change(foot_normal, foot_normal+foot_tangent),
            measured_outgoing_deg=heading(after), normal_and_gravity_only_outgoing_deg=heading(normal_velocity),
            all_measured_impulses_outgoing_deg=heading(complete_velocity),
            tangent_addition_outgoing_change_deg=heading_change(normal_velocity, complete_velocity)))


def balance_bins(rows, mass_kg, gravity_w):
    """Balance already validated, separately measured normal/tangent pairs.

    The counterfactual sums are descriptive vector arithmetic; they do not
    predict the changed trajectory of a hypothetical friction-free simulation.
    """
    mass, gravity = finite_number(mass_kg), vector(gravity_w)
    if mass <= 0 or not rows:
        raise ValueError('Positive actual mass and nonempty bins required')
    previous = None
    for row in rows:
        start, end = finite_number(row['step_start_s']), finite_number(row['step_end_s'])
        if abs(end-start-DT) > 1e-8 or start < 0:
            raise ValueError('Every momentum sample must span exactly 5 ms')
        before, after = vector(row['velocity_before']), vector(row['velocity_after'])
        if previous is not None and (abs(start-previous['step_end_s']) > 1e-8 or
                not np.array_equal(before, vector(previous['velocity_after']))):
            raise ValueError('Contiguous time and identical shared boundary state required')
        if set(row['pairs']) != set(PAIRS):
            raise ValueError('Four disjoint ball-side contact groups required')
        for pair in row['pairs'].values():
            vector(pair['normal']); vector(pair['tangent'])
        previous = row
    bins = [_balance([r], mass, gravity) for r in rows]
    aggregate = _balance(rows, mass, gravity)
    worst = max(bins, key=lambda b: b['residual_norm_Ns'])
    failed = [b['step_end_s'] for b in bins if not b['passed']]
    return dict(passed=aggregate['passed'] and not failed, bins=bins, aggregate=aggregate,
        failed_bin_end_s=failed, maximum_bin_residual_Ns=worst['residual_norm_Ns'],
        maximum_bin_residual_end_s=worst['step_end_s'],
        maximum_absolute_bin_residual_by_coordinate_Ns=np.max(np.abs([b['residual_Ns'] for b in bins]), axis=0).tolist(),
        tolerance_Ns=TOLERANCE_NS,
        coordinates='World x,y,z; heading atan2(x,-y) degrees, positive toward +x from -y. Undefined for zero horizontal vector.',
        interpretation='Tangential impulse is measured from separate friction anchors with the original probe sign. '
            'Residual is unexplained, never an estimated friction impulse. Normal-only and tangent-added headings '
            'are ordered sums of measured vectors, not independently simulated interventions.')


def compare_original(reference, candidate, root=ROOT):
    """Require every original field, with only declared observation exclusions."""
    changed, missing, path_exceptions = [], [], []
    streams = {}
    for name in ('traces', 'contacts', 'kinematics'):
        a, b = reference.get(name), candidate.get(name)
        if a is None or b is None:
            missing.append(name)
            continue
        if name == 'traces':
            b = [[{k: v for k, v in row.items() if k != 'ball_contact_detail'} for row in episode]
                 for episode in b]
        streams[name] = dict(equal=a == b, reference_count=len(a[0]) if a else 0,
                             candidate_count=len(b[0]) if b else 0)
        if a != b:
            changed.append(name)
    for key, value in reference.items():
        if key in ('traces', 'contacts', 'kinematics', 'dynamics_telemetry'):
            continue
        if key not in candidate:
            missing.append(key)
        elif value != candidate[key]:
            changed.append(key)
    old_meta, new_meta = reference.get('dynamics_telemetry', {}), candidate.get('dynamics_telemetry', {})
    for key, value in old_meta.items():
        if key == 'runner_source_sha256':
            continue
        if key == 'probe_path' and key in new_meta and value != new_meta[key]:
            a, b = (Path(root)/p for p in (value, new_meta[key]))
            expected = old_meta.get('probe_sha256')
            if (a.resolve() == b.resolve() and expected == new_meta.get('probe_sha256')
                    and a.is_file() and sha(a) == expected):
                path_exceptions.append(dict(field='dynamics_telemetry.probe_path', reference=value,
                    candidate=new_meta[key], resolved_path=str(a.resolve()), verified_sha256=expected))
                continue
        if key not in new_meta:
            missing.append('dynamics_telemetry.'+key)
        elif value != new_meta[key]:
            changed.append('dynamics_telemetry.'+key)
    try:
        rows = [validated_rows(r['traces'][0]) for r in (reference, candidate)]
        complete = all(len(r) == 2800 and abs(r[-1]['time_s']-14.) < 1e-7
                       and all(abs(s['physics_dt_s']-DT) < 1e-9 for s in r) for r in rows)
    except (KeyError, ValueError, TypeError, IndexError):
        complete = False
    return dict(status='available', passed=complete and not changed and not missing,
        complete_original_2800_samples=complete, streams=streams, changed_fields=changed,
        missing_fields=missing, metadata_path_exceptions=path_exceptions, ignored=['traces[].ball_contact_detail',
            'dynamics_telemetry.runner_source_sha256', 'new acquisition metadata'],
        definition='Every original trace/contact/kinematic/feedback/record and configuration field is exact. '
                   'Old failure labels are retained; successful observation is not successful shooting.')


def _buffer_entries(pair, kind, matrix_shape, sign, allow_signed_normal=False):
    counts = np.asarray(pair[kind+'_counts'], dtype=float)
    starts = np.asarray(pair[kind+'_starts'], dtype=float)
    capacity = finite_number(pair[kind+'_buffer_capacity'])
    if (counts.shape != matrix_shape or starts.shape != counts.shape or capacity <= 0 or int(capacity) != capacity
            or not np.isfinite(counts).all() or not np.isfinite(starts).all()
            or np.any(counts < 0) or np.any(counts != np.floor(counts)) or np.any(starts != np.floor(starts))):
        raise ValueError('Invalid '+kind+' buffer metadata')
    if counts.sum() >= capacity:
        raise ValueError('Possibly exhausted '+kind+' contact buffer')
    expected = set()
    indices = set()
    for idx in np.ndindex(counts.shape):
        count, start = int(counts[idx]), int(starts[idx])
        if count and (start < 0 or start+count >= capacity):
            raise ValueError('Out-of-capacity or possibly saturated '+kind+' interval')
        for buffer_index in range(start, start+count):
            if buffer_index in indices:
                raise ValueError('Overlapping '+kind+' pair intervals')
            indices.add(buffer_index)
            expected.add((idx, buffer_index))
    rows = pair['normal_contacts' if kind == 'normal' else 'friction_anchors']
    actual = set()
    reconstructed = np.zeros((*matrix_shape, 3))
    for entry in rows:
        raw_idx = entry['pair_index']
        index = tuple(int(x) for x in raw_idx)
        raw_buffer = finite_number(entry['buffer_index'])
        token = (index, int(raw_buffer))
        if list(index) != raw_idx or raw_buffer != int(raw_buffer) or token not in expected or token in actual:
            raise ValueError('Missing, duplicated or misattributed '+kind+' entry')
        actual.add(token)
        vector(entry['point_w'])
        if kind == 'normal':
            scalar, normal = finite_number(entry['force_scalar_N']), vector(entry['normal_w'])
            finite_number(entry['separation_m'])
            if (scalar < 0 and not allow_signed_normal) or (scalar != 0 and abs(np.linalg.norm(normal)-1.) > 1e-4):
                raise ValueError('Invalid normal magnitude or direction')
            force = scalar*normal
        else:
            force = sign*vector(entry['force_raw_w_N'])
        if 'force_w_N' in entry and not np.allclose(vector(entry['force_w_N']), force, rtol=0, atol=RECONSTRUCTION_ATOL_N):
            raise ValueError('Stored '+kind+' vector does not match raw data and fixed sign')
        reconstructed[index] += force
    if actual != expected:
        raise ValueError('Incomplete '+kind+' raw entry coverage')
    return reconstructed, int(counts.sum())


def _decode_signed_pair(pair, sign):
    """Decode only the known scalar-sign rejection, preserving raw f and n.

    This is an explicit offline interpretation correction. It does not alter the
    original acquisition flags or repair malformed buffers/force matrices.
    """
    if pair.get('error') != 'Negative normal-force magnitude' or pair.get('force_on_ball_valid') is not False:
        raise ValueError('Only the recorded negative-scalar rejection is eligible for signed-raw decoding')
    raw = pair['raw_buffers']
    f, p, n, d, c, s = [np.asarray(x, dtype=float) for x in raw['normal']]
    ff, fp, fc, fs = [np.asarray(x, dtype=float) for x in raw['friction']]
    matrix = np.asarray(raw['normal_force_matrix_w_N'], dtype=float)
    if (f.ndim != 2 or f.shape[1:] != (1,) or d.shape != f.shape or p.shape != (len(f), 3)
            or n.shape != p.shape or ff.ndim != 2 or ff.shape[1:] != (3,) or fp.shape != ff.shape
            or c.ndim != 2 or c.shape[0] != 1 or s.shape != c.shape or fc.shape != c.shape
            or fs.shape != c.shape or matrix.shape != (*c.shape, 3) or not np.isfinite(matrix).all()):
        raise ValueError('Malformed signed-raw contact buffers')
    def entries(counts, starts, size, fields):
        if (not np.isfinite(counts).all() or not np.isfinite(starts).all() or np.any(counts < 0)
                or np.any(counts != np.floor(counts)) or np.any(starts != np.floor(starts))):
            raise ValueError('Invalid signed-raw counts/starts')
        result = []
        for idx in np.ndindex(counts.shape):
            count, start = int(counts[idx]), int(starts[idx])
            if count and (start < 0 or start+count >= size):
                raise ValueError('Signed-raw range exceeds or touches capacity')
            for j in range(start, start+count):
                result.append(dict(pair_index=list(idx), buffer_index=j,
                    **{name: values[j].tolist() for name, values in fields.items()}))
        return result
    normal = entries(c, s, len(f), dict(force_scalar_N=f[:, 0], point_w=p, normal_w=n,
                                      separation_m=d[:, 0], force_w_N=f*n))
    friction = entries(fc, fs, len(ff), dict(point_w=fp, force_raw_w_N=ff, force_w_N=sign*ff))
    decoded = dict(pair, normal_contacts=normal, friction_anchors=friction,
        normal_counts=c.tolist(), normal_starts=s.tolist(), friction_counts=fc.tolist(), friction_starts=fs.tolist(),
        normal_buffer_capacity=len(f), friction_buffer_capacity=len(ff), capacity_exhausted=False,
        normal_force_matrix_w_N=matrix.tolist(), normal_reconstruction_tolerance_N=RECONSTRUCTION_ATOL_N)
    # The same independent checker enforces finite active data, distinct ranges,
    # complete indices and per-filter attribution for both original and raw paths.
    normal_matrix, nc = _buffer_entries(decoded, 'normal', c.shape, sign, allow_signed_normal=True)
    tangent_matrix, tc = _buffer_entries(decoded, 'friction', c.shape, sign)
    normal_sum, tangent_sum = matrix.sum(axis=(0, 1)), tangent_matrix.sum(axis=(0, 1))
    valid = bool(np.max(np.linalg.norm(normal_matrix-matrix, axis=-1)) <= RECONSTRUCTION_ATOL_N)
    decoded.update(raw_normal_reconstruction_matrix_w_N=normal_matrix.tolist(),
        normal_force_on_ball_w_N=normal_sum.tolist(), friction_raw_matrix_w_N=(tangent_matrix/sign).tolist(),
        friction_raw_w_N=(tangent_sum/sign).tolist(), friction_force_on_ball_w_N=tangent_sum.tolist(),
        force_on_ball_w_N=(normal_sum+tangent_sum).tolist(), force_on_ball_valid=valid,
        normal_reconstruction_verified=valid, normal_contact_count=nc, friction_anchor_count=tc)
    return decoded


def check_pair(pair, friction_sign, decode_signed_raw=False):
    """Independently reconstruct each filter pair, not just a cancelling sum."""
    if friction_sign not in (-1, 1) or pair['friction_sign'] != friction_sign:
        raise ValueError('Friction sign must equal the original probe certificate')
    acquisition_valid, acquisition_error = pair['force_on_ball_valid'], pair.get('error')
    decoded = bool(decode_signed_raw and acquisition_valid is False)
    if decoded:
        pair = _decode_signed_pair(pair, friction_sign)
    matrix = np.asarray(pair['normal_force_matrix_w_N'], dtype=float)
    if matrix.ndim != 3 or matrix.shape[0] != 1 or matrix.shape[1] < 1 or matrix.shape[2] != 3 or not np.isfinite(matrix).all():
        raise ValueError('Original one-ball filter matrix shape required')
    if pair['capacity_exhausted'] is not False:
        raise ValueError('Contact buffer capacity not certified')
    normal, normal_count = _buffer_entries(pair, 'normal', matrix.shape[:-1], friction_sign, allow_signed_normal=decoded)
    tangent, tangent_count = _buffer_entries(pair, 'friction', matrix.shape[:-1], friction_sign)
    error = float(np.max(np.linalg.norm(normal-matrix, axis=-1)))
    saved = np.asarray(pair['raw_normal_reconstruction_matrix_w_N'], dtype=float)
    if (saved.shape != normal.shape or not np.isfinite(saved).all()
            or not np.allclose(saved, normal, rtol=0, atol=RECONSTRUCTION_ATOL_N)
            or error > RECONSTRUCTION_ATOL_N or pair['normal_reconstruction_verified'] is not True):
        raise ValueError('Raw normal points do not reconstruct every matrix filter')
    raw_tangent = np.asarray(pair['friction_raw_matrix_w_N'], dtype=float)
    if (raw_tangent.shape != tangent.shape or not np.isfinite(raw_tangent).all()
            or not np.allclose(raw_tangent*friction_sign, tangent, rtol=0, atol=RECONSTRUCTION_ATOL_N)
            or pair['normal_contact_count'] != normal_count or pair['friction_anchor_count'] != tangent_count
            or pair['normal_reconstruction_tolerance_N'] != RECONSTRUCTION_ATOL_N):
        raise ValueError('Recorded independent friction filters/counts or reconstruction bound differ')
    n, t = normal.reshape(-1, 3).sum(0), tangent.reshape(-1, 3).sum(0)
    for name, value in (('normal_force_on_ball_w_N', matrix.reshape(-1, 3).sum(0)),
                        ('friction_raw_w_N', t/friction_sign), ('friction_force_on_ball_w_N', t),
                        ('force_on_ball_w_N', matrix.reshape(-1, 3).sum(0)+t)):
        if not np.allclose(vector(pair[name]), value, rtol=0, atol=RECONSTRUCTION_ATOL_N):
            raise ValueError('Stored force component differs from independently reconstructed '+name)
    if pair['force_on_ball_valid'] is not True:
        raise ValueError('Producer did not certify the ball-side force')
    return dict(normal=matrix.reshape(-1, 3).sum(0).tolist(), tangent=t.tolist(),
                raw_normal_sum=n.tolist(), normal_contact_count=normal_count,
                friction_anchor_count=tangent_count, normal_reconstruction_max_error_N=error,
                offline_signed_raw_decoded=decoded, acquisition_force_on_ball_valid=acquisition_valid,
                acquisition_error=acquisition_error,
                negative_normal_scalar_count=sum(e['force_scalar_N'] < 0 for e in pair['normal_contacts']))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def _verified_file(root, path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('A complete SHA256 pin is required: '+str(path))
    path = Path(root)/path
    actual = sha(path)
    if actual != expected:
        raise ValueError('SHA256 mismatch: '+str(path))
    return dict(path=str(path), sha256=actual)


def provenance(reference, candidate, protocol, root=ROOT):
    """Rehash declared bytes and compare acquisition pins, never relabel sources."""
    from targeted_shooting import controller_provenance

    root = Path(root)
    if (protocol['case_id'] != 'heldout_v028_left_near_slow'
            or tuple(protocol['bin_end_window_s']) != WINDOW
            or protocol['closure_tolerance_Ns'] != TOLERANCE_NS
            or protocol['maximum_complete_physical_replays'] != 1):
        raise ValueError('The single old case, window and closure tolerance are fixed before acquisition')
    verified_reference = _verified_file(root, protocol['reference_report'], protocol['reference_report_sha256'])
    if read_json(verified_reference['path']) != reference:
        raise ValueError('Supplied reference differs from pinned v32 report')
    frozen = protocol['frozen_files_sha256']
    if len(frozen) != 19 or frozen != read_json(root/'configs/strike_baseline_v1.json')['frozen_files_sha256']:
        raise ValueError('Exactly the original 19-file mapping is required')
    control, acquisition = protocol['control_sources_sha256'], protocol['acquisition_sources_sha256']
    if not control or not acquisition:
        raise KeyError('Control and pre-acquisition source fingerprints are not frozen')
    required_acquisition = {'tools/run_instrumented_strike.py', 'tools/ball_contact_observer.py',
                            'tools/ground_contact_observer.py', 'tools/strike_dynamics_telemetry.py'}
    if set(acquisition) != required_acquisition:
        raise ValueError('Exactly the four declared acquisition sources must be pinned')
    for name in control.keys() & acquisition.keys():
        if control[name] != acquisition[name]:
            raise ValueError('Unchanged helper has conflicting source pins: '+name)
    files = [_verified_file(root, name, digest) for mapping in (frozen, control, acquisition)
             for name, digest in mapping.items()]
    cfg_pin, model_pin, probe_pin = (protocol[k] for k in ('candidate_config', 'checkpoint', 'probe'))
    for pin in (cfg_pin, model_pin, probe_pin):
        files.append(_verified_file(root, pin['path'], pin['sha256']))
    config = read_json(root/cfg_pin['path'])
    expected_model = config['post_shot_recovery']
    if (expected_model['checkpoint'] != model_pin['path']
            or expected_model['checkpoint_sha256'] != model_pin['sha256']):
        raise ValueError('Pinned checkpoint is not the exact original v32 actor')
    current_controller = controller_provenance(root)['controller_sources_sha256']
    for report in (reference, candidate):
        joint = report['joint_strike']
        if (joint['config'] != config or joint['config_sha256'] != cfg_pin['sha256']
                or report['policy_sha256'] != cfg_pin['sha256'] or joint['config']['frozen_files_sha256'] != frozen
                or joint['checkpoint_sha256'] != model_pin['sha256'] or joint['checkpoint'] != model_pin['path']
                or report['controller_sources_sha256'] != current_controller):
            raise ValueError('Recorded original control/configuration/checkpoint provenance differs')
        for name, digest in joint['source_sha256'].items():
            if control.get('tools/'+name) != digest:
                raise ValueError('Recorded active controller source differs: '+name)
        for name, digest in report['sequence_sources_sha256'].items():
            if frozen.get('tools/'+name) != digest:
                raise ValueError('Recorded sequence source differs: '+name)
        metadata = report['dynamics_telemetry']
        if metadata['probe_sha256'] != probe_pin['sha256']:
            raise ValueError('Recorded force probe differs from the original certificate')
        if (report['root_fixed'] is not False or report['post_initialization_state_writes'] is not False
                or metadata['physical_writes'] is not False or metadata['controller_changed'] is not False):
            raise ValueError('Original physical-write/free-root invariants not retained')
    meta = candidate['dynamics_telemetry']
    if (meta['acquisition_sources_sha256'] != acquisition
            or meta['runner_source_sha256'] != acquisition['tools/run_instrumented_strike.py']
            or meta['observer_source_sha256'] != acquisition['tools/strike_dynamics_telemetry.py']
            or meta['ball_contact_detail']['source_sha256'] != acquisition['tools/ball_contact_observer.py']
            or meta['ball_contact_detail']['probe_sha256'] != probe_pin['sha256']
            or meta['ball_contact_detail']['indexed_entries_source_sha256'] != acquisition['tools/ground_contact_observer.py']):
        raise ValueError('Actual acquisition fingerprints do not match the predeclared sources')
    certificate = validated_force_probe(read_json(root/probe_pin['path']))
    return dict(status='available', passed=True, reference_report=verified_reference,
        original_frozen_file_count=19, verified_files=files, acquisition_sources_sha256=acquisition,
        original_probe_certificate=certificate,
        interpretation='Recorded acquisition hashes and current declared bytes agree with the pinned v32 report, '
            'configuration, checkpoint, original 19 assets and control sources. Recorded no-write invariants '
            'and exact replay are evidence; a trace alone cannot prove absence of unlogged writes.')


def _flat_paths(value):
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError('Recorded actor/filter path list required')
    return [path for item in value for path in _flat_paths(item)]


def observer_parameters(candidate, decode_signed_raw=False):
    meta = candidate['dynamics_telemetry']['ball_contact_detail']
    if (tuple(meta['window_s']) != WINDOW or meta['samples'] != 55 or meta['expected_samples'] != 55
            or meta['window_complete'] is not True
            or (meta['force_coverage_complete'] is not True and not decode_signed_raw)
            or meta['physical_writes'] is not False
            or meta['normal_reconstruction_tolerance_N'] != RECONSTRUCTION_ATOL_N
            or meta['reaction_tolerance_N'] != RECONSTRUCTION_ATOL_N):
        raise ValueError('Fixed observation window, sample count, read-only and reconstruction bounds required')
    mass = finite_number(meta['ball_mass_kg'])
    gravity, effective = vector(meta['gravity_w_mps2']), vector(meta['effective_ball_gravity_w_mps2'])
    disabled = meta['ball_disable_gravity']
    if mass <= 0 or type(disabled) is not bool or not np.array_equal(effective, np.zeros(3) if disabled else gravity):
        raise ValueError('Actual mass and explicit effective ball gravity required')
    if (meta['ball_mass_source'] != 'soccer_ball.root_physx_view.get_masses()'
            or meta['gravity_source'] != 'SimulationView.get_gravity()'
            or meta['disable_gravity_source'] != 'soccer_ball.root_physx_view.get_disable_gravities()'
            or meta['ball_actor_paths'] != ['/World/envs/env_0/SoccerBall']):
        raise ValueError('Recorded actual mass/gravity getter provenance required')
    linear, angular = (finite_number(meta['damping'][key]) for key in ('linear_per_s', 'angular_per_s'))
    if min(linear, angular) < 0 or not meta['damping']['source']:
        raise ValueError('Actual finite nonnegative damping and API source required')
    groups, bodies = meta['pair_groups'], meta['robot_body_names']
    if set(groups) != set(PAIRS) or not bodies or len(set(bodies)) != len(bodies):
        raise ValueError('Four contact groups and unique complete robot body inventory required')
    feet = {side: side+'_ankle_roll_link' for side in ('right', 'left')}
    if not set(feet.values()).issubset(bodies):
        raise ValueError('Original foot actor names absent')
    expected = {side+'_foot': ['/World/envs/env_*/Robot/'+body] for side, body in feet.items()}
    expected['other_robot'] = ['/World/envs/env_*/Robot/'+body for body in bodies if body not in feet.values()]
    expected['ground'] = candidate['ground_collision_paths']
    if meta['ground_collision_paths'] != candidate['ground_collision_paths']:
        raise ValueError('Ball ground paths differ from original physical ground')
    seen = set()
    for name in PAIRS:
        group = groups[name]
        actual_paths = _flat_paths(group['filter_paths'])
        sensors = _flat_paths(group['sensor_paths'])
        exact_filters = [p.replace('env_*', 'env_0') for p in expected[name]]
        if (group['source_actor_pattern'] != '/World/envs/env_*/SoccerBall'
                or set(group['filter_patterns']) != set(expected[name])
                or len(group['filter_patterns']) != len(expected[name])
                or group['sensor_count'] != 1 or group['filter_count'] != len(expected[name])
                or sensors != ['/World/envs/env_0/SoccerBall']
                or len(actual_paths) != len(exact_filters) or set(actual_paths) != set(exact_filters)
                or seen.intersection(actual_paths) or group['max_contact_data_count'] <= 0):
            raise ValueError('Nonexclusive, missing or misattributed ball contact filters: '+name)
        seen.update(actual_paths)
    return dict(status='available' if linear == 0 else 'unknown', passed=linear == 0,
        mass_kg=mass, gravity_w_mps2=gravity.tolist(), effective_gravity_w_mps2=effective.tolist(),
        ball_disable_gravity=disabled, damping=meta['damping'], pair_groups=groups,
        source_metadata=meta, acquisition_force_coverage_complete=meta['force_coverage_complete'],
        offline_signed_raw_requested=decode_signed_raw,
        damping_boundary='Linear damping is zero: no omitted linear damping impulse.' if linear == 0 else
            'Nonzero linear damping: its per-bin impulse was not measured. No closure certification or friction attribution.',
        interpretation='Groups cover the recorded robot actors and original ground, with unique actual filter paths. '
            'This does not prove absence of unrecorded external actors. Angular damping affects spin/contact evolution '
            'but is not an omitted direct force in the translational balance.')


def check_reaction(measured, ball, side, decode_signed_raw=False):
    matrix = np.asarray(measured['normal_force_matrix_w_N'], dtype=float)
    if matrix.shape != (1, 1, 3) or not np.isfinite(matrix).all():
        raise ValueError('Original foot-on-ball reaction sensor matrix required')
    foot, ball = matrix.reshape(-1, 3).sum(0), vector(ball)
    error = float(np.linalg.norm(foot+ball))
    if (not np.allclose(vector(measured['normal_force_on_foot_w_N']), foot, rtol=0, atol=RECONSTRUCTION_ATOL_N)
            or measured['force_on'] != side+'_foot'
            or _flat_paths(measured['sensor_paths']) != ['/World/envs/env_0/Robot/'+side+'_ankle_roll_link']
            or _flat_paths(measured['filter_paths']) != ['/World/envs/env_0/SoccerBall']
            or (measured['verified'] is not True and not decode_signed_raw)
            or measured['tolerance_N'] != RECONSTRUCTION_ATOL_N
            or error > RECONSTRUCTION_ATOL_N):
        raise ValueError('Ball-side normal lacks opposing original foot-sensor evidence')
    return dict(side=side, force_on_foot_w_N=foot.tolist(), force_on_ball_w_N=ball.tolist(), opposed_error_N=error,
                acquisition_verified=measured['verified'], independently_verified=True)


def collect_bins(candidate, friction_sign, decode_signed_raw=False):
    rows = validated_rows(candidate['traces'][0])
    meta = candidate['dynamics_telemetry']['ball_contact_detail']
    selected = [(i, r) for i, r in enumerate(rows) if WINDOW[0]-1e-9 <= r['time_s'] <= WINDOW[1]+1e-9]
    if len(selected) != 55:
        raise ValueError('All 55 predeclared bin-end samples required')
    if any('ball_contact_detail' in r for r in rows if not WINDOW[0]-1e-9 <= r['time_s'] <= WINDOW[1]+1e-9):
        raise ValueError('Detailed acquisition outside the predeclared window')
    normalized, reaction, decoded_evidence, max_reconstruction = [], [], [], 0.
    if meta['friction_sign'] != friction_sign:
        raise ValueError('Metadata friction sign differs from original probe')
    for i, row in selected:
        detail = row['ball_contact_detail']
        start, end = finite_number(detail['step_start_s']), finite_number(detail['step_end_s'])
        if abs(end-row['time_s']) > 1e-8 or abs(start-rows[i-1]['time_s']) > 1e-8 or abs(end-start-DT) > 1e-8:
            raise ValueError('Observer pre/end time does not bracket the original physics bin')
        before, after = detail['ball_before'], detail['ball_after']
        vb, va = vector(before['com_linear_velocity_w_mps']), vector(after['com_linear_velocity_w_mps'])
        if (not np.array_equal(vb, vector(rows[i-1]['ball_velocity']))
                or not np.array_equal(va, vector(row['ball_velocity']))):
            raise ValueError('Observed COM velocities differ from same-step original ball telemetry')
        for state, old in ((before, rows[i-1]), (after, row)):
            for key in ('link_position_w_m', 'com_position_w_m', 'link_origin_linear_velocity_w_mps', 'angular_velocity_w_rad_s'):
                vector(state[key])
            for key in ('link_quaternion_wxyz', 'com_quaternion_wxyz'):
                q = np.asarray(state[key], dtype=float)
                if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > 1e-4:
                    raise ValueError('Finite unit quaternion state required')
            if (not np.array_equal(vector(state['link_position_w_m']), vector(old['ball_xyz']))
                    or not np.array_equal(vector(state['angular_velocity_w_rad_s']), vector(old['ball_angular_velocity']))):
                raise ValueError('Ball state frame/time differs from original physical sample')
        if set(detail['pairs']) != set(PAIRS):
            raise ValueError('Complete disjoint contact groups required in every bin')
        pairs = {}
        for name in PAIRS:
            pair = detail['pairs'][name]
            group = meta['pair_groups'][name]
            if (pair['source_actor_pattern'] != group['source_actor_pattern']
                    or pair['filter_patterns'] != group['filter_patterns'] or pair['force_on'] != 'ball'
                    or pair['frame'] != 'world'):
                raise ValueError('Per-bin ball actor/filter/frame differs from recorded acquisition')
            pairs[name] = check_pair(pair, friction_sign, decode_signed_raw=decode_signed_raw)
            matrix = pair.get('normal_force_matrix_w_N', pair.get('raw_buffers', {}).get('normal_force_matrix_w_N'))
            if np.asarray(matrix).shape != (1, group['filter_count'], 3):
                raise ValueError('Actual per-filter matrix shape differs from declared paths')
            max_reconstruction = max(max_reconstruction, pairs[name]['normal_reconstruction_max_error_N'])
            if pairs[name]['offline_signed_raw_decoded']:
                decoded_evidence.append(dict(step_end_s=end, pair=name, **pairs[name]))
        for side in ('right', 'left'):
            measured = detail['foot_reaction_normal'][side]
            reaction.append(dict(step_end_s=end, **check_reaction(measured, pairs[side+'_foot']['normal'], side,
                decode_signed_raw=pairs[side+'_foot']['offline_signed_raw_decoded'])))
        normalized.append(dict(step_start_s=start, step_end_s=end, velocity_before=vb.tolist(),
            velocity_after=va.tolist(), pairs=pairs))
    if max(np.linalg.norm(r['force_on_ball_w_N']) for r in reaction) <= 10*RECONSTRUCTION_ATOL_N:
        raise ValueError('Opposing normal-force sign has no measured nonzero contact signal')
    return normalized, dict(status='available', passed=True, bin_count=len(normalized),
        normal_reconstruction_max_error_N=max_reconstruction,
        opposing_normal_max_error_N=max(r['opposed_error_N'] for r in reaction),
        opposing_normal_samples=reaction, force_sign=friction_sign,
        offline_signed_raw_decoded_pairs=decoded_evidence,
        original_acquisition_force_coverage_complete=meta['force_coverage_complete'],
        offline_reconstruction_valid=True,
        normal_reconstruction_tolerance_N=RECONSTRUCTION_ATOL_N,
        interpretation='Normal reaction orientation is independently checked against the original foot-actor sensor. '
            'Tangential sign is inherited unchanged from the independently validated probe, not fit from this shot. '
            'Normal points and friction anchors retain independent indices; no point-to-anchor correspondence is assumed.')


def audit_ball_impulse(reference, candidate, protocol, root=ROOT, decode_signed_raw=False):
    """Keep measured arithmetic separate from prerequisite and task verdicts."""
    checks = {}
    def checked(name, function):
        try:
            value = function()
            checks[name] = value
        except (KeyError, OSError) as error:
            checks[name] = dict(status='unknown', passed=False, reason=str(error))
        except (ValueError, TypeError, IndexError, OverflowError) as error:
            checks[name] = dict(status='rejected', passed=False, reason=str(error))
        return checks[name]

    checked('exact_original_replay', lambda: compare_original(reference, candidate, root))
    checked('provenance', lambda: provenance(reference, candidate, protocol, root))
    parameters = checked('observer_parameters', lambda: observer_parameters(candidate, decode_signed_raw))
    def probe():
        pin = protocol['probe']
        file = _verified_file(root, pin['path'], pin['sha256'])
        return dict(status='available', passed=True, file=file,
                    certificate=validated_force_probe(read_json(file['path'])))
    probe_evidence = checked('original_force_probe', probe)
    bins = []
    def raw():
        nonlocal bins
        bins, value = collect_bins(candidate, probe_evidence['certificate']['friction_sign'], decode_signed_raw)
        return value
    checked('raw_pair_time_and_reaction', raw)
    windows = {}
    episode = None
    def closure():
        nonlocal episode
        mass, gravity = parameters['mass_kg'], parameters['effective_gravity_w_mps2']
        if not bins:
            raise KeyError('Validated ball impulse bins unavailable')
        episode = first_force_episode(candidate['traces'][0])
        if episode is None or episode['outgoing_time_s'] is None:
            raise KeyError('Original first force episode or outgoing separation sample unavailable')
        if episode['first_force_s'] < WINDOW[0]-1e-9 or episode['outgoing_time_s'] > WINDOW[1]+1e-9:
            raise ValueError('Detailed window does not contain the complete original first strike through outgoing sample')
        scopes = dict(full_declared_window=(WINDOW[0], WINDOW[1]),
            first_force_episode=(episode['first_force_s'], episode['last_force_s']),
            first_force_through_outgoing_sample=(episode['first_force_s'], episode['outgoing_time_s']),
            separation_outgoing_bin=(episode['last_force_s']+DT, episode['outgoing_time_s']))
        for name, (first, last) in scopes.items():
            selected = [b for b in bins if first-1e-9 <= b['step_end_s'] <= last+1e-9]
            if not selected or abs(selected[0]['step_end_s']-first) > 1e-8 or abs(selected[-1]['step_end_s']-last) > 1e-8:
                raise ValueError('Missing exact original episode boundary: '+name)
            windows[name] = balance_bins(selected, mass, gravity)
        required = ('full_declared_window', 'first_force_through_outgoing_sample')
        return dict(status='available', passed=all(windows[n]['passed'] for n in required),
                    required_scopes=['each_5ms_bin', *required],
                    additional_descriptive_scopes=['first_force_episode', 'separation_outgoing_bin'],
                    tolerance_Ns=TOLERANCE_NS,
                    interpretation='Each bin and both predeclared cumulative windows must close separately. '
                        'The impulse bound is inherited from the prior probe; it is not a per-bin hardware accuracy specification.')
    checked('translational_closure', closure)
    def outcomes():
        return dict(status='available', reference=task_outcome(reference), candidate=task_outcome(candidate),
                    original_records=candidate['records'], original_sequence_records=candidate['sequence_records'])
    try:
        original_task = outcomes()
    except (KeyError, ValueError, TypeError, IndexError) as error:
        original_task = dict(status='unknown', reason=str(error))
    failures = [name for name, item in checks.items() if item.get('passed') is not True]
    rejected = any(item.get('status') == 'rejected' or
                   (item.get('status') == 'available' and item.get('passed') is False) for item in checks.values())
    return dict(schema_version=1, diagnostic_verified=not failures,
        interpretation_mode='explicit_offline_signed_raw' if decode_signed_raw else 'strict_acquisition',
        interpretation_revision='Explicit offline repair of the invalid assumption that an SDK scalar normal is nonnegative. '
            'Use signed f*n unchanged, original probe sign, original reconstruction/closure bounds and original raw data. '
            'All ranges/finite values/filter matrices/reactions are rechecked; acquisition flags remain unmodified.'
            if decode_signed_raw else None,
        status='verified' if not failures else 'rejected' if rejected else 'unknown',
        failed_or_unknown_checks=failures, checks=checks, windows=windows,
        original_first_force_episode=episode, original_task=original_task,
        time_definition='Original force/velocity timestamps are bin ends. The first included bin starts 5 ms '
            'before its force timestamp. Outgoing is the original first zero-foot-force sample after the first contiguous episode. '
            'Observer before/after COM states must equal those same original bin boundaries.',
        scope='One old development case, original measured controller trajectory. Closure permits attribution only '
            'to measured external impulse components. It does not identify internal solver load allocation, certify a '
            'contact-point/anchor correspondence, improve the controller, repair old goal misses or establish naturalness.',
        omissions='No residual is assigned to friction. Linear damping must be zero for certification. '
            'Raw contact/anchor solver staging remains unknown; this audit is a COM linear-momentum balance, not an angular-momentum or energy proof.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'protocol', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--decode-signed-raw', action='store_true',
                        help='Explicit offline decode of the known signed-scalar collector rejection; preserve raw signs and original flags')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evidence, reports = {}, {}
    for name in ('reference', 'candidate', 'protocol'):
        path = getattr(args, name).resolve()
        try:
            reports[name] = read_json(path)
            if not isinstance(reports[name], dict):
                raise ValueError('Top-level JSON object required')
            evidence[name] = dict(path=str(path), sha256=sha(path))
        except (OSError, ValueError, TypeError) as error:
            evidence[name] = dict(path=str(path), status='unknown', reason=str(error))
            reports.pop(name, None)
    if len(reports) != 3:
        result = dict(schema_version=1, diagnostic_verified=False, status='unknown',
                      failed_or_unknown_checks=['input_reports'], checks={}, windows={})
    else:
        result = audit_ball_impulse(reports['reference'], reports['candidate'], reports['protocol'],
                                    decode_signed_raw=args.decode_signed_raw)
    result['input_reports'] = evidence
    result['analysis_sources_sha256'] = {name: sha(ROOT/name) for name in (
        'tools/audit_ball_impulse.py', 'tools/strike_baseline_metrics.py',
        'tools/strike_dynamics_telemetry.py', 'tools/targeted_shooting.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), status=result['status'],
        diagnostic_verified=result['diagnostic_verified'], failed_or_unknown_checks=result['failed_or_unknown_checks'])))
    return 0 if result['diagnostic_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
