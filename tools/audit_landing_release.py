#!/usr/bin/env python3
"""Independent CPU audit of v30 landing-envelope release against frozen v28.

Physical task acceptance, prefix/strike invariance, and retrospective dynamics
are separate. Lower impact peaks cannot turn a failed original task into a pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_explosive_strike import compare_frozen_replay, read_verified_report
from audit_p0_dynamic_strike import limit_observations, measure_path
from explosive_strike_metrics import contact_metrics, landing_window_dynamics, summarize_explosive_strike, unavailable
from strike_baseline_metrics import validated_rows
from strike_dynamics_telemetry import validated_force_probe

ROOT = Path(__file__).resolve().parents[1]
POLICY_DT = .02
EPS = 1e-9
WEIGHT_ATOL = 5e-7  # Float32 multiply in the saved controller debug.
LAG_ATOL_S = 2e-6  # Seconds: float32 policy/frame timestamps, not a phase-frame count.
LAG_TRIGGER_EPS_S = 1e-6  # Recorded v2 controller's numerical zero guard.
PREFIX_JOINT_NAMES = [side + '_' + joint + '_joint' for side in ('left', 'right')
                      for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'ankle_pitch', 'ankle_roll')]
ABSOLUTE_JOINT_NAMES = ['right_' + joint + '_joint' for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
V1_AUDIT_SHA256 = '5d0e998b7cfaa2604af5ccec902a86db7e713be91aef49bd72c51c2e24cb6f0c'
FIXED_INPUTS = ('asset_sha256', 'joint_names', 'joint_velocity_limits', 'joint_effort_limits',
                'initial_robot_state', 'points', 'goals', 'goal_width_m', 'goal_height_m',
                'motion_sources_sha256', 'termination_policy', 'contact_geometry', 'ground_collision_paths')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _policy_time(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError('Finite policy timestamp required')
    rounded = round(value / POLICY_DT) * POLICY_DT
    if abs(value - rounded) > 2e-6:
        raise ValueError('Feedback must remain on the original 20 ms policy grid')
    return rounded


def first_release_decision(result):
    feedback = result.get('joint_strike', {}).get('feedback', [])
    if not feedback or any(not isinstance(r.get('landing_release'), dict) for r in feedback):
        return unavailable('Complete per-decision release debug is absent', 'joint_strike.feedback[].landing_release.retention')
    first = None
    for i, row in enumerate(feedback):
        t = _policy_time(row['time_s'])
        if abs(t - i * POLICY_DT) > EPS:
            raise ValueError('Release feedback omits, duplicates, or reorders a policy decision')
        retention = row['landing_release'].get('retention')
        if not isinstance(retention, (int, float)) or not math.isfinite(retention) or not 0 <= retention <= 1:
            raise ValueError('Retention must be finite and lie in [0, 1]')
        if first is None and retention < 1.:
            first = dict(status='available', activation_observed=True, first_retention_below_one_s=t,
                         raw_decision_time_s=row['time_s'], decision_index=i, debug=row['landing_release'])
    return first or dict(status='available', activation_observed=False, first_retention_below_one_s=None,
                         raw_decision_time_s=None, decision_index=None, debug=None,
                         reason='Retention never fell below one; an effective release was not observed')


def release_weight_evidence(result):
    """Check observed mode/weight rules, not the controller's quintic formula."""
    feedback = result.get('joint_strike', {}).get('feedback', [])
    if not feedback:
        return unavailable('Release-weight observations absent', 'per-decision original/effective weights')
    failures, maximum_error, modes, zero_s = [], 0., set(), None
    previous = 1.
    for row in feedback:
        t = _policy_time(row['time_s'])
        d = row.get('landing_release', {})
        required = ['mode', 'retention', 'pulse_finished'] + [name + '_weight_' + suffix
            for name in ('swing', 'absolute', 'placement') for suffix in ('original', 'effective')]
        if any(k not in d for k in required):
            return unavailable('Incomplete original/effective release-weight observations',
                               'all three weight pairs, mode, retention and pulse_finished at every decision')
        mode, retention = d['mode'], d['retention']
        if mode not in ('all', 'pose') or not isinstance(retention, (int, float)) or not math.isfinite(retention) or not 0 <= retention <= 1:
            raise ValueError('Invalid release mode or retention')
        modes.add(mode)
        if retention > previous + WEIGHT_ATOL:
            failures.append(dict(time_s=t, reason='retention_increased'))
        if retention < 1 and d['pulse_finished'] is not True:
            failures.append(dict(time_s=t, reason='pulse_not_finished'))
        if zero_s is None and retention == 0.:
            zero_s = t
        previous = retention
        for name in ('swing', 'absolute', 'placement'):
            original, effective = d[name + '_weight_original'], d[name + '_weight_effective']
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not -WEIGHT_ATOL <= v <= 1 + WEIGHT_ATOL
                   for v in (original, effective)):
                raise ValueError('Invalid observed release weight')
            expected = original * (1. if mode == 'pose' and name == 'swing' else retention)
            error = abs(effective - expected)
            maximum_error = max(maximum_error, error)
            if error > WEIGHT_ATOL:
                failures.append(dict(time_s=t, weight=name, error=error, reason='effective_weight_inconsistent'))
    if len(modes) != 1:
        failures.append(dict(reason='mode_changed'))
    return dict(status='available', valid=not failures, modes=sorted(modes), decision_count=len(feedback),
                absolute_tolerance=WEIGHT_ATOL, max_absolute_weight_error=maximum_error,
                first_zero_retention_decision_s=zero_s, failures=failures,
                interpretation='Observed all mode scales three weights; pose mode preserves swing weight and scales absolute/placement. A nonincreasing retention and completed pulse are checked, without reimplementing the trigger or quintic curve.')


def phase_lag_evidence(result):
    feedback = result.get('joint_strike', {}).get('feedback', [])
    cfg = result.get('joint_strike', {}).get('config', {}).get('landing_release', {})
    expected = 'phase_lag_scale_s' in cfg or any('phase_lag_s' in r.get('landing_release', {}) for r in feedback)
    if not expected:
        return dict(**unavailable('Acquisition predates lag-conditioned release', 'v2 lag and frozen-strength observations'), expected=False)
    scale, duration = cfg.get('phase_lag_scale_s'), cfg.get('duration_s')
    if not feedback or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (scale, duration)):
        return dict(**unavailable('Missing declared lag scale or release duration', 'resolved landing_release configuration and feedback'), expected=True)
    failures, max_unit_error = [], 0.
    trigger, trigger_lag, strength, settled, zero_count = None, None, None, None, 0
    for row in feedback:
        d = row.get('landing_release', {})
        required = ('phase_lag_s', 'frame_phase_lag_s', 'release_strength', 'trigger_time_s', 'retention')
        if any(k not in d for k in required) or any(k not in row for k in ('phase_action', 'phase_s')):
            return dict(**unavailable('Incomplete lag/strength observations', 'both lag fields, phase_action, phase_s, trigger, strength and retention'), expected=True)
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in [*(d[k] for k in required), row['phase_action'], row['phase_s']]):
            raise ValueError('Nonfinite lag/strength evidence')
        t, lag, frame_lag, s, r, observed_trigger = _policy_time(row['time_s']), d['phase_lag_s'], d['frame_phase_lag_s'], d['release_strength'], d['retention'], d['trigger_time_s']
        if lag < 0 or frame_lag < 0 or not 0 <= s <= 1 or not 0 <= r <= 1:
            raise ValueError('Out-of-range lag/strength evidence')
        error = max(abs(lag - max(-row['phase_action'], 0.)), abs(frame_lag - max(t - row['phase_s'], 0.)), abs(lag-frame_lag))
        max_unit_error = max(max_unit_error, error)
        if error > LAG_ATOL_S:
            failures.append(dict(time_s=t, reason='lag_seconds_inconsistent', maximum_error_s=error))
        zero_count += lag <= LAG_TRIGGER_EPS_S
        if observed_trigger < 0:
            if trigger is not None:
                failures.append(dict(time_s=t, reason='trigger_reset'))
            if s != 0. or r != 1.:
                failures.append(dict(time_s=t, reason='untriggered_strength_or_retention_changed'))
            continue
        if trigger is None:
            trigger, trigger_lag, strength = observed_trigger, lag, s
            if abs(t-trigger) > LAG_ATOL_S:
                failures.append(dict(time_s=t, reason='first_trigger_time_inconsistent'))
            if lag <= LAG_TRIGGER_EPS_S:
                failures.append(dict(time_s=t, reason='zero_lag_triggered'))
            if abs(s-min(lag/scale, 1.)) > WEIGHT_ATOL:
                failures.append(dict(time_s=t, reason='strength_not_scaled_from_trigger_lag'))
        elif observed_trigger != trigger or s != strength:
            failures.append(dict(time_s=t, reason='trigger_or_strength_not_frozen'))
        if r < 1.-strength-WEIGHT_ATOL:
            failures.append(dict(time_s=t, reason='retention_below_partial_release_floor'))
        if row['time_s']-trigger > duration+LAG_ATOL_S:
            settled = r
            if abs(r-(1.-strength)) > WEIGHT_ATOL:
                failures.append(dict(time_s=t, reason='settled_retention_wrong'))
    return dict(status='available', expected=True, valid=not failures, decision_count=len(feedback), failures=failures,
                phase_lag_scale_s=scale, duration_s=duration, lag_absolute_tolerance_s=LAG_ATOL_S,
                numerical_zero_lag_s=LAG_TRIGGER_EPS_S, maximum_lag_unit_error_s=max_unit_error,
                trigger_time_s=trigger, trigger_phase_lag_s=trigger_lag, frozen_release_strength=strength,
                settled_retention=settled, settled_endpoint_observed=settled is not None,
                zero_lag_decisions=zero_count,
                zero_lag_preservation_observed=not failures and trigger is None and zero_count == len(feedback),
                interpretation='Lag seconds are independently compared with max(-phase_action,0) and max(policy_time-phase_s,0). Strength is checked at the first trigger and must remain exactly frozen thereafter, even if current lag returns to zero. Only settled endpoint retention is checked; no claim of whole-command C2 continuity is made.')


def absolute_window_exit_evidence(result):
    feedback = result.get('joint_strike', {}).get('feedback', [])
    partial = any(0 < row.get('landing_release', {}).get('release_strength', 0.) < 1 for row in feedback)
    if not partial:
        return dict(status='not_applicable', reason='No observed partial-strength release')
    for previous, row in zip(feedback, feedback[1:]):
        a, b = previous.get('landing_release', {}), row.get('landing_release', {})
        if a.get('absolute_weight_original', 0.) <= 0 or b.get('absolute_weight_original') != 0 or b.get('retention', 0.) <= 0 or not 0 < b.get('release_strength', 0.) < 1:
            continue
        t = _policy_time(row['time_s'])
        base, blend = (np.asarray(b.get('absolute_' + k + '_target_rad', []), float) for k in ('base', 'blended'))
        if b.get('absolute_joint_names') != ABSOLUTE_JOINT_NAMES or base.shape != (6,) or blend.shape != (6,) or not np.isfinite(base).all() or not np.isfinite(blend).all():
            return unavailable('Original window exit lacks internal targets', 'named base/blended targets at the first original absolute-weight zero decision')
        offset = blend-base
        return dict(status='available', decision_time_s=t, previous_decision_time_s=_policy_time(previous['time_s']),
                    retention=b['retention'], release_strength=b['release_strength'], joint_names=ABSOLUTE_JOINT_NAMES,
                    original_weight_before=a['absolute_weight_original'], original_weight_after=b['absolute_weight_original'],
                    bypassed_projection_offset_rad=offset.tolist(), maximum_bypassed_projection_offset_rad=float(np.max(np.abs(offset))),
                    final_target_boundary=target_release_boundary(result, t),
                    interpretation='At original absolute weight zero, the saved internal blend is bypassed by the original active-window mask. This offset exposes remaining projection when partial release leaves retention>0. The measured final target step also contains policy/residual changes; neither is a whole-command C2 certificate.')
    return unavailable('No observed original-window positive-to-zero transition with partial retention',
                       'consecutive original absolute weights and targets across the window exit')


def projection_evidence(result):
    """Audit recorded whole-operator retention; legacy raw targets stay absent."""
    feedback = result.get('joint_strike', {}).get('feedback', [])
    expected = any('projection_semantics' in row.get('landing_release', {}) for row in feedback)
    if not expected:
        return dict(**unavailable('Legacy acquisition did not record absolute-operator inputs',
                                 'base, projected and blended targets with runtime soft bounds'), expected=False)
    failures, max_error, endpoints = [], 0., 0
    for row in feedback:
        t, d = _policy_time(row['time_s']), row.get('landing_release', {})
        if d.get('projection_semantics') != 'whole_operator_retention' or d.get('absolute_joint_names') != ABSOLUTE_JOINT_NAMES:
            return dict(**unavailable('Incomplete or unsupported absolute-operator schema',
                                     'whole_operator_retention and ordered six right-leg joint names at every decision'), expected=True)
        base, projected, blended = [np.asarray(d.get('absolute_' + k + '_target_rad', []), float)
                                     for k in ('base', 'projected', 'blended')]
        bounds = np.asarray(d.get('absolute_soft_limits_rad', []), float)
        if any(v.shape != (6,) or not np.isfinite(v).all() for v in (base, projected, blended)) or bounds.shape != (6, 2) or not np.isfinite(bounds).all() or not np.all(bounds[:, 0] < bounds[:, 1]):
            return dict(**unavailable('Incomplete finite absolute-operator targets/bounds',
                                     'six measured base/projected/blended values and lower/upper soft bounds'), expected=True)
        retention = d.get('retention')
        if not isinstance(retention, (int, float)) or not math.isfinite(retention) or not 0 <= retention <= 1:
            raise ValueError('Invalid absolute-operator retention')
        error = float(np.max(np.abs(blended - (base + retention * (projected - base)))))
        max_error = max(max_error, error)
        if error > WEIGHT_ATOL:
            failures.append(dict(time_s=t, reason='whole_operator_formula_mismatch', maximum_error_rad=error))
        if np.any(projected < bounds[:, 0] - WEIGHT_ATOL) or np.any(projected > bounds[:, 1] + WEIGHT_ATOL):
            failures.append(dict(time_s=t, reason='projected_target_outside_recorded_soft_bounds'))
        if retention in (0., 1.):
            endpoints += 1
            if not np.array_equal(blended, base if retention == 0 else projected):
                failures.append(dict(time_s=t, reason='endpoint_not_exact', retention=retention))
    return dict(status='available', expected=True, valid=not failures, decision_count=len(feedback),
                semantics='whole_operator_retention', joint_names=ABSOLUTE_JOINT_NAMES,
                maximum_formula_error_rad=max_error, absolute_tolerance_rad=WEIGHT_ATOL,
                exact_endpoint_checks=endpoints, failures=failures,
                interpretation='Saved internal q_blend = q_base + r*(q_projected-q_base), with exact r=0/1 endpoints and projected soft bounds. Downstream residuals and final actuator targets remain distinct; this does not certify smoothness across changing policy targets.')


def target_release_boundary(result, decision_s):
    """Inspect the saved target on each side of a zero-retention command."""
    rows = validated_rows(result['traces'][0])
    before = next((r for r in rows if abs(r['time_s'] - decision_s) <= EPS), None)
    after = next((r for r in rows if abs(r['time_s'] - (decision_s + .005)) <= EPS), None)
    if before is None or after is None:
        return unavailable('The exact before/after command samples are absent',
                           'joint_target at the decision timestamp and first 5 ms bin after it')
    names = result.get('joint_names', [])
    a, b = (np.asarray(row.get('joint_target', []), float) for row in (before, after))
    if not names or a.shape != (len(names),) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return unavailable('Complete measured joint targets are absent', 'joint_names and both joint_target vectors')
    feedback = next((r for r in result.get('joint_strike', {}).get('feedback', [])
                     if abs(_policy_time(r['time_s']) - decision_s) <= EPS), {})
    bounds = np.asarray(feedback.get('approach_soft_limits_rad', []), float)
    known = (dict(zip(PREFIX_JOINT_NAMES, bounds.tolist())) if bounds.shape == (10, 2)
             and np.isfinite(bounds).all() and np.all(bounds[:, 0] < bounds[:, 1]) else {})
    absolute = feedback.get('landing_release', {})
    absolute_bounds = np.asarray(absolute.get('absolute_soft_limits_rad', []), float)
    if absolute.get('absolute_joint_names') == ABSOLUTE_JOINT_NAMES and absolute_bounds.shape == (6, 2) and np.isfinite(absolute_bounds).all() and np.all(absolute_bounds[:, 0] < absolute_bounds[:, 1]):
        for name, bound in zip(ABSOLUTE_JOINT_NAMES, absolute_bounds.tolist()):
            if name in known and known[name] != bound:
                raise ValueError('Runtime soft-bound observations disagree')
            known[name] = bound
    joints, unknown = [], []
    for j, name in enumerate(names):
        item = dict(joint=name, target_before_rad=float(a[j]), target_after_rad=float(b[j]), target_delta_rad=float(b[j]-a[j]))
        if name in known:
            lo, hi = known[name]
            excess_a, excess_b = max(lo-a[j], a[j]-hi, 0.), max(lo-b[j], b[j]-hi, 0.)
            item.update(soft_bounds=dict(status='available', bounds_rad=[lo, hi]),
                        soft_excess_before_rad=float(excess_a), soft_excess_after_rad=float(excess_b),
                        crossed_soft_bound=bool(excess_a <= 1e-6 and excess_b > 1e-6))
        else:
            item['soft_bounds'] = unavailable('This joint soft bound was not recorded', 'runtime soft_joint_pos_limits for this joint')
            unknown.append(name)
        joints.append(item)
    worst = max(joints, key=lambda j: abs(j['target_delta_rad']))
    return dict(status='available', decision_time_s=decision_s,
                state_sample_times_s=[before['time_s'], after['time_s']], joints=joints,
                maximum_absolute_target_step_rad=abs(worst['target_delta_rad']), maximum_step_joint=worst['joint'],
                unobserved_soft_bound_joints=unknown,
                soft_bound_source='Recorded feedback.approach_soft_limits_rad in fixed 10-joint prefix_names order, plus absolute_soft_limits_rad for the six named right-leg joints when sampled. No URDF-derived bounds substituted.',
                interpretation='The sample ending at the decision still holds the previous command; the first subsequent 5 ms sample records the new target. Soft-bound excess is a target diagnostic, not measured joint-limit violation. The final target also contains downstream residuals, so its change cannot be assigned solely to the absolute branch.')


def _compare_stream(reference, candidate):
    different = []
    for a, b in zip(reference, candidate):
        keys = [k for k, v in a.items() if k not in b or b[k] != v]
        if keys:
            different.append(dict(time_s=a['time_s'], keys=keys))
    return dict(equal=len(reference) == len(candidate) and not different,
                reference_count=len(reference), candidate_count=len(candidate),
                first_difference_s=different[0]['time_s'] if different else None,
                first_changed_keys=different[0]['keys'] if different else [],
                differing_sample_count=len(different))


def compare_prefix(reference, candidate, decision_s):
    if not math.isfinite(decision_s) or decision_s < 0:
        raise ValueError('Finite nonnegative effective release time required')
    streams = {}
    for name, a, b, policy in [
        ('physical', validated_rows(reference['traces'][0]), validated_rows(candidate['traces'][0]), False),
        ('contacts', (reference.get('contacts') or [[]])[0], (candidate.get('contacts') or [[]])[0], False),
        ('kinematics', (reference.get('kinematics') or [[]])[0], (candidate.get('kinematics') or [[]])[0], True),
        ('feedback', reference.get('joint_strike', {}).get('feedback', []), candidate.get('joint_strike', {}).get('feedback', []), True),
    ]:
        before = lambda r: (_policy_time(r['time_s']) if policy else r['time_s']) < decision_s - EPS
        streams[name] = _compare_stream([r for r in a if before(r)], [r for r in b if before(r)])
    return dict(equal=all(s['equal'] for s in streams.values()), decision_time_s=decision_s, streams=streams,
                definition='Every original value strictly before the first effective retention<1 decision; new observation/debug keys are ignored. Policy timestamps are checked against the 20 ms grid.')


def compare_full_legacy(reference, candidate):
    cutoff = max(r['time_s'] for report in (reference, candidate) for r in report['traces'][0]) + POLICY_DT
    compared = compare_prefix(reference, candidate, cutoff)
    identity = compare_frozen_replay(candidate, reference)
    compared.update(equal=compared['equal'] and identity['records_equal'] and identity['legacy_trace_fields_equal'],
                    decision_time_s=None, full_run_identity=identity,
                    definition='Deliberate zero-lag preservation: every original field over the entire physical/contact/kinematic/feedback streams and original records must remain exact. Changed controller/config hashes are metadata, not physical-output differences.')
    return compared


def compare_primary_strike(reference_rows, candidate_rows, direction):
    a, b = (contact_metrics(rows, direction) for rows in (reference_rows, candidate_rows))
    fields = ('first_force_s', 'last_force_s', 'force_duration_s', 'normal_impulse_Ns', 'peak_normal_force_N',
              'separation_sample_s', 'preimpact_foot', 'outgoing_ball', 'observed_contact_feet')
    if a['status'] != 'available' or b['status'] != 'available':
        return dict(equal=False, reason='Both runs require an actual primary foot-ball contact', reference=a, candidate=b)
    changes = [key for key in fields if a[key] != b[key]]
    complete = a['separation_sample_s'] is not None and b['separation_sample_s'] is not None
    return dict(equal=not changes and complete, changed_fields=changes, primary_episode_complete=complete,
                reference={key: a[key] for key in fields}, candidate={key: b[key] for key in fields},
                definition='Exact first contiguous foot-ball episode, fixed-point preimpact state and first zero-force outgoing state; later contacts are not merged into this episode')


def first_right_touchdown(rows, contact_s):
    rows = validated_rows(rows)
    if contact_s is None or not math.isfinite(contact_s):
        return unavailable('No measured ball contact anchors touchdown search', 'actual first foot-ball contact')
    forces, source = [], 'ground-filtered right vertical normal-force scalar'
    use_vector = all('support_foot_normal_force_w_N' in r for r in rows)
    for row in rows:
        try:
            if use_vector:
                v = np.asarray(row['support_foot_normal_force_w_N']['right'], float)
                if v.shape != (3,) or not np.isfinite(v).all():
                    raise ValueError('Invalid right normal-force vector')
                force = float(np.linalg.norm(v))
                source = 'norm of recorded ground-filtered right normal-force vector'
            else:
                force = float(row['support_foot_force_N']['right'])
            if not math.isfinite(force) or force < 0:
                raise ValueError('Invalid right support force')
        except (KeyError, TypeError, ValueError):
            return unavailable('Incomplete right ground-force observations', 'right normal ground force across the search interval')
        forces.append(force)
    for i in range(1, len(rows)):
        if rows[i]['time_s'] > contact_s + EPS and forces[i - 1] <= EPS and forces[i] > EPS:
            return dict(status='available', time_s=rows[i]['time_s'], index=i,
                        onset_bracket_s=[rows[i - 1]['time_s'], rows[i]['time_s']],
                        first_sample_normal_force_N=forces[i], zero_tolerance_N=EPS, source=source,
                        definition='First post-ball-contact right ground-normal transition from zero to positive; not the largest force sample')
    return unavailable('No zero-to-positive right ground-contact transition after ball contact',
                       'an observed unloaded-to-loaded right-foot transition')


def release_verdict(prefix, strike, outcome, inputs, release):
    failures = list(outcome.get('failure_reasons', []))
    if not outcome.get('success') and not failures:
        failures.append('original_full_task_failed')
    if not prefix.get('equal'):
        failures.append('protected_prefix_changed_or_unavailable')
    if not strike.get('equal'):
        failures.append('primary_contact_or_outgoing_changed')
    if not inputs.get('match'):
        failures.append('frozen_physical_inputs_changed')
    preserved = release.get('verification_mode') == 'inactive_zero_lag_preservation'
    if not release.get('activation_observed') and not preserved:
        failures.append('no_effective_release_observed')
    if not release.get('after_observed_separation') and not preserved:
        failures.append('release_not_after_measured_separation')
    if not release.get('weight_evidence', {}).get('valid'):
        failures.append('release_weight_debug_inconsistent_or_unavailable')
    projection = release.get('projection_evidence', {})
    if projection.get('expected') and not projection.get('valid'):
        failures.append('whole_operator_evidence_inconsistent_or_unavailable')
    lag = release.get('phase_lag_evidence', {})
    if lag.get('expected') and not lag.get('valid'):
        failures.append('phase_lag_evidence_inconsistent_or_unavailable')
    if preserved and (not lag.get('zero_lag_preservation_observed') or 'full_run_identity' not in prefix):
        failures.append('inactive_zero_lag_full_run_not_verified')
    return dict(audit_pass=not failures, failure_reasons=failures,
                verification_mode=release.get('verification_mode', 'active_release'),
                interpretation='Pass means original task plus protected prefix/strike and measured release timing, or an explicitly verified exact full-run zero-lag preservation. It does not certify improved dynamics, command continuity, matrix robustness, or human naturalness')


def _probe(result):
    metadata = result.get('dynamics_telemetry')
    if not isinstance(metadata, dict):
        return None, unavailable('No instrumented force/segment evidence', 'dynamics_telemetry and independent force-probe source')
    path = Path(metadata['probe_path'])
    if not path.is_absolute():
        path = ROOT / path
    probe, digest = read_verified_report(ROOT, path, metadata['probe_sha256'])
    certificate = validated_force_probe(probe)
    return probe, dict(path=str(path.relative_to(ROOT)), sha256=digest, certificate=certificate,
                       acquisition_metadata=metadata)


def describe_run(result):
    metrics = summarize_explosive_strike(result)
    rows = validated_rows(result['traces'][0])
    probe, provenance = _probe(result)
    contact_s = metrics['contact'].get('first_force_s')
    touchdown = first_right_touchdown(rows, contact_s)
    windows = {}
    for name, start, end in ([('event_aligned', touchdown['time_s'] - .05, touchdown['time_s'] + .10)]
                              if touchdown['status'] == 'available' else []) + [('legacy_absolute', 1.80, 1.90)]:
        windows[name] = landing_window_dynamics(rows, result.get('dynamics_telemetry', {}), max(0., start), end,
            metrics['direction_w'], joint_names=result['joint_names'], joint_velocity_limits=result['joint_velocity_limits'],
            gravity_w=probe['gravity_w'] if probe else [0., 0., -9.81])
    if 'event_aligned' not in windows:
        windows['event_aligned'] = unavailable('No measured touchdown to align the window', 'right touchdown after ball contact')
    follow = unavailable('No complete measured follow-through path', 'bracketing 20 ms right-foot/root poses through contact+0.75 s')
    if contact_s is not None:
        try:
            poses = result['kinematics'][0]
            times = np.asarray([p['time_s'] for p in poses])
            roots = np.asarray([p['root_pose'][:3] for p in poses])
            ankles = np.asarray([p['feet']['right']['position'] for p in poses])
            q = np.asarray([p['feet']['right']['quaternion'] for p in poses])
            point = ankles + Rotation.from_quat(q[:, [1, 2, 3, 0]]).apply(result['foot_velocity_point_local'])
            follow = dict(status='available', point_local_m=result['foot_velocity_point_local'],
                measured=measure_path(times, point, roots, contact_s, metrics['direction_w']),
                interpretation='Existing P0 measured fixed-point path relative to pelvis translation; retrospective diagnostic, not a replacement task gate')
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    return dict(original_metrics=metrics, original_task_evidence=dict(records=result.get('records'),
                    termination_reasons=result.get('termination_reasons'),
                    stride_shot_evidence=result.get('stride_shot_evidence'), running_evidence=result.get('running_evidence')),
                controller_provenance=dict(joint_strike_config_sha256=result.get('joint_strike', {}).get('config_sha256'),
                    joint_strike_sources_sha256=result.get('joint_strike', {}).get('source_sha256'),
                    controller_sources_sha256=result.get('controller_sources_sha256'),
                    sequence_sources_sha256=result.get('sequence_sources_sha256'),
                    landing_release_config=result.get('joint_strike', {}).get('config', {}).get('landing_release'),
                    interpretation='Hashes recorded at acquisition, retained separately from current analysis source hashes. Raw report SHA anchors this metadata; historical source contents are not substituted with current files.'),
                joint_limit_observations=limit_observations(rows, result),
                right_touchdown=touchdown, windows=windows, pelvis_relative_follow_through=follow,
                probe_provenance=provenance)


def load_baseline(case):
    comparison_path = ROOT / 'reports/v028_final_comparison.json'
    comparison = json.loads(comparison_path.read_text())
    entry = next(c for c in comparison['cases'] if c['case']['id'] == case)
    reference, reference_digest = read_verified_report(ROOT, entry['candidate_report'], entry['candidate_sha256'])
    instrumented_manifest_path = ROOT / 'reports/explosive_instrumented_v029.json'
    manifest = json.loads(instrumented_manifest_path.read_text())
    observed_entry = next((c for c in manifest['cases'] if c['case'] == case), None)
    if observed_entry is None:
        return reference, reference, dict(case=case, frozen_path=entry['candidate_report'], frozen_sha256=reference_digest,
            comparison_sha256=sha(comparison_path), instrumented_manifest_sha256=sha(instrumented_manifest_path),
            baseline_observation='Frozen original v28 only; vector ground force/friction and high-rate segment telemetry were not acquired for this case')
    observed, observed_digest = read_verified_report(ROOT, observed_entry['report'], observed_entry['sha256'])
    identity = compare_frozen_replay(observed, reference)
    if not all(identity[k] for k in ('records_equal', 'kinematics_equal', 'legacy_trace_fields_equal')):
        raise ValueError('Instrumented baseline changes original v28 observations: ' + case)
    return reference, observed, dict(case=case, frozen_path=entry['candidate_report'], frozen_sha256=reference_digest,
        instrumented_path=observed_entry['report'], instrumented_sha256=observed_digest,
        original_replay_identity=identity, comparison_sha256=sha(comparison_path), instrumented_manifest_sha256=sha(instrumented_manifest_path))


def load_continuous_reference(case):
    manifest, digest = read_verified_report(ROOT, 'reports/landing_release_v030_audit.json', V1_AUDIT_SHA256)
    entry = next((e for e in manifest['entries'] if e['label'] == 'v030_continuous_pose_' + case), None)
    if entry is None:
        return None, dict(**unavailable('No continuous-pose acquisition for this case in preserved v1 evidence',
                                       'a measured continuous-pose run for this condition'), v1_audit_sha256=digest)
    p = entry['provenance']
    observed, observed_digest = read_verified_report(ROOT, p['candidate_path'], p['candidate_sha256'])
    return observed, dict(status='available', continuous_pose_path=p['candidate_path'], continuous_pose_sha256=observed_digest,
                          v1_audit_sha256=digest)


def audit_candidate(reference, baseline_observed, candidate):
    before, after = describe_run(baseline_observed), describe_run(candidate)
    release = first_release_decision(candidate)
    release['weight_evidence'] = release_weight_evidence(candidate)
    release['projection_evidence'] = projection_evidence(candidate)
    release['phase_lag_evidence'] = phase_lag_evidence(candidate)
    decision_s = release.get('first_retention_below_one_s')
    preserved = release['phase_lag_evidence'].get('zero_lag_preservation_observed', False) and decision_s is None
    release['verification_mode'] = 'inactive_zero_lag_preservation' if preserved else 'active_release'
    prefix = (compare_full_legacy(reference, candidate) if preserved else compare_prefix(reference, candidate, decision_s)
              if decision_s is not None else dict(equal=False, reason='No effective release decision was observed'))
    strike = compare_primary_strike(reference['traces'][0], candidate['traces'][0], before['original_metrics']['direction_w'])
    separation = after['original_metrics']['contact'].get('separation_sample_s')
    release['after_observed_separation'] = bool(decision_s is not None and separation is not None and decision_s >= separation - EPS)
    input_changes = [k for k in FIXED_INPUTS if k not in candidate or candidate[k] != reference.get(k)]
    input_changes += [k for k in ('root_fixed', 'post_initialization_state_writes') if candidate.get(k) is not False]
    inputs = dict(match=not input_changes, changed_fields=input_changes)
    zero_s = release['weight_evidence'].get('first_zero_retention_decision_s')
    boundary = ({'reference': target_release_boundary(reference, zero_s), 'candidate': target_release_boundary(candidate, zero_s)}
                if zero_s is not None else unavailable('No observed zero-retention decision', 'a decision with retention exactly zero'))
    window_exit = absolute_window_exit_evidence(candidate)
    if window_exit['status'] == 'available':
        window_exit['reference_target_boundary'] = target_release_boundary(reference, window_exit['decision_time_s'])
    return dict(mode='actual_comparison', release=release, protected_prefix=prefix, primary_strike=strike,
                physical_inputs=inputs, release_target_boundary=boundary, absolute_window_exit=window_exit, reference=before, candidate=after,
                verdict=release_verdict(prefix, strike, after['original_metrics']['outcome'], inputs, release))


def render_report(result):
    runs, seen_references = [], set()
    for entry in result['entries']:
        case = entry['provenance']['case']
        if case not in seen_references:
            runs.append(('v28/' + case, entry['audit']['reference']))
            seen_references.add(case)
        continuous = entry['audit'].get('continuous_pose_comparison', {})
        if continuous.get('status') == 'available':
            runs.append(('v30-continuous-pose/' + case, continuous['reference']))
        if 'candidate' in entry['audit']:
            runs.append((entry['label'], entry['audit']['candidate']))
    lines = ['# Landing-release audit — 2026-11-07', '',
             'Independent CPU analysis for v0.1.30. Original task acceptance is preserved; lower landing peaks '
             'cannot conceal a failed goal, contact surface, running transition, recovery, or physical-integrity gate.', '',
             'Each measured touchdown is the first right-foot ground-normal transition from zero to positive '
             'after ball contact. Its fixed diagnostic window is 50 ms before to 100 ms after that event. '
             'The legacy 1.80–1.90 s interval remains in the JSON. Endpoint-inclusive force peaks and '
             'overlapping-bin impulses are distinct, following the P1 definitions.', '',
             '| Case / run | Mode | Release decision (s) | Protected prefix | Primary strike unchanged | Original full task | Audit pass |',
             '|---|---|---:|---|---|---|---|']
    for entry in result['entries']:
        audit = entry['audit']; actual = audit['mode'] == 'actual_comparison'
        if actual:
            release = audit['release'].get('first_retention_below_one_s')
            mode = (audit['release'].get('debug') or {}).get('mode', 'unavailable')
            if audit['release']['verification_mode'] == 'inactive_zero_lag_preservation':
                mode = 'zero-lag preservation'
            lines.append(f"| {entry['label']} | {mode} | {release} | {audit['protected_prefix'].get('equal')} | {audit['primary_strike']['equal']} | {audit['candidate']['original_metrics']['outcome']['success']} | {audit['verdict']['audit_pass']} |")
        else:
            lines.append(f"| {entry['label']} | baseline dry analysis | unavailable | not tested | not tested | {audit['reference']['original_metrics']['outcome']['success']} | not evaluated |")
    def get(value, *keys):
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return 'unavailable' if value is None else (f'{value:.6g}' if isinstance(value, (int, float)) else str(value))
    lines += ['', '| Case / run | Completed physics s | Terminal markers | Final stable standing s | Whole-run max q̇/limit | Initial outgoing XY speed m/s | Termination reasons |',
              '|---|---:|---:|---:|---:|---:|---|']
    for label, data in runs:
        m = data['original_metrics']
        values = [get(m, 'completed_physics_seconds'), get(m, 'terminal_status_markers'),
                  get(m, 'recovery', 'original_recovery', 'final_standing_seconds'),
                  get(data, 'joint_limit_observations', 'maximum_actual_speed_ratio'),
                  get(m, 'contact', 'outgoing_ball', 'speed_xy_mps'), get(data, 'original_task_evidence', 'termination_reasons')]
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '| Case / run | Touchdown s | Event force peak N | Event foot-speed jump m/s | Event max q̇/limit | Event momentum residual norm N·s | Max residual bin s |',
              '|---|---:|---:|---:|---:|---:|---|']
    for label, data in runs:
        w = data['windows']['event_aligned']
        values = [get(data, 'right_touchdown', 'time_s'), get(w, 'endpoint_inclusive_total_force_peak', 'peak_norm_force_N'),
            get(w, 'foot_point_motion', 'largest_velocity_jump_mps'), get(w, 'joint_speed_limits', 'max_speed_to_limit_ratio'),
            get(w, 'momentum_balance', 'residual_norm_Ns'), get(w, 'momentum_balance', 'max_residual_bin_s')]
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'The following impulses use the complete event window. Vectors are world [x, y, z], in N·s; '
              'normal and friction are independently recorded, then summed. They are bilateral foot-ground impulses, '
              'not ball impulse or inferred energy shares.', '',
              '| Case / run | Normal impulse | Friction impulse | Total impulse |', '|---|---|---|---|']
    for label, data in runs:
        components = data['windows']['event_aligned'].get('ground_reaction', {}).get('combined', {}).get('components', {})
        vectors = ['[' + ', '.join(f'{v:.6g}' for v in components[k]['impulse_w_Ns']) + ']'
                   if k in components else 'unavailable' for k in ('normal', 'friction', 'total')]
        lines.append('| ' + label + ' | ' + ' | '.join(vectors) + ' |')
    lines += ['', '| Case / run | Pelvis net rotation ° | Thorax net rotation ° | Pelvis projected yaw change ° | Thorax projected yaw change ° |',
              '|---|---:|---:|---:|---:|']
    for label, data in runs:
        segments = data['windows']['event_aligned'].get('segments', {})
        values = [get(segments, segment, key) for key in ('net_rotation_angle_deg', 'net_projected_yaw_deg')
                  for segment in ('pelvis', 'thorax')]
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '| Case / run | Pelvis-relative advance m | Relative displacement at 250 ms m | Relative speed at 120 ms m/s | Relative velocity step m/s | P0 diagnostic failures |',
              '|---|---:|---:|---:|---:|---|']
    for label, data in runs:
        follow = data['pelvis_relative_follow_through'].get('measured', {})
        values = [get(follow, key) for key in ('pelvis_relative_advance_m', 'pelvis_relative_displacement_at_250ms_m',
                  'pelvis_relative_speed_at_120ms_mps', 'peak_relative_velocity_step_mps', 'diagnostic_failures')]
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'Every unavailable field keeps its evidence requirement in the JSON. The event-aligned '
              'window uses observed timestamps and can remain incomplete after early termination. Physical '
              'task failures and P0 follow-through diagnostic failures are retained separately; neither is '
              'removed because another number improved.', '', '## Outcomes and evidence', '']
    for entry in result['entries']:
        audit = entry['audit']
        if audit['mode'] == 'actual_comparison':
            counts = ', '.join(f"{name} {e['reference_count']}/{e['candidate_count']}" for name, e in audit['protected_prefix'].get('streams', {}).items())
            lines += [f"- **{entry['label']}**: audit failures `{audit['verdict']['failure_reasons']}`; original task failures `{audit['candidate']['original_metrics']['outcome']['failure_reasons']}`. Protected sample counts (reference/candidate): {counts}. Recorded weight-rule check: `{audit['release']['weight_evidence'].get('valid')}`."]
            projection = audit['release']['projection_evidence']
            if projection['status'] == 'available':
                lines += [f"  Whole-operator formula/endpoint check: `{projection['valid']}` over {projection['decision_count']} decisions; max formula error {projection['maximum_formula_error_rad']:.6g} rad, {projection['exact_endpoint_checks']} exact endpoint checks."]
            lag = audit['release']['phase_lag_evidence']
            if lag['status'] == 'available':
                lines += [f"  Lag/strength check: `{lag['valid']}`; trigger lag {get(lag, 'trigger_phase_lag_s')} s, frozen strength {get(lag, 'frozen_release_strength')}, settled retention {get(lag, 'settled_retention')}; maximum lag-unit disagreement {get(lag, 'maximum_lag_unit_error_s')} s. Verification mode: `{audit['release']['verification_mode']}`."]
            continuous = audit.get('continuous_pose_comparison', {})
            if continuous.get('status') == 'available':
                lines += [f"  Compared with preserved v1 continuous-pose: physical outputs exact `{continuous['physical_outputs_equal']}`; policy/config fingerprints intentionally checked as provenance, not motion equality. Full v28 comparison remains primary."]
        else:
            lines += [f"- **{entry['label']}**: baseline measurements only; no v30 release or outcome has been inferred."]
    actual_entries = [e for e in result['entries'] if e['audit']['mode'] == 'actual_comparison']
    if actual_entries:
        lines += ['', '## Zero-retention target boundary', '',
                  'This diagnostic checks the actual saved command immediately before and after the first '
                  'zero-retention decision. The physical sample ending at the decision holds the previous command; '
                  'the next 5 ms sample holds the new one. Soft bounds come only from the ten recorded '
                  '`approach_soft_limits_rad` joints and the six right-leg `absolute_soft_limits_rad` joints '
                  'where the later acquisition includes them (11 unique joints). Soft-target excess is distinct from actual joint-speed '
                  'violation, and does not replace the frozen task gate.', '',
                  '| Case / run | Side | Decision s | Max target step rad | Joint with max step | Right ankle pitch target before → after rad | Right ankle pitch soft excess after rad | Right ankle roll soft excess after rad |',
                  '|---|---|---:|---:|---|---|---:|---:|']
        for entry in actual_entries:
            for side in ('reference', 'candidate'):
                boundary = entry['audit']['release_target_boundary'].get(side, {})
                joints = {j['joint']: j for j in boundary.get('joints', [])}
                pitch, roll = joints.get('right_ankle_pitch_joint', {}), joints.get('right_ankle_roll_joint', {})
                values = [get(boundary, 'decision_time_s'), get(boundary, 'maximum_absolute_target_step_rad'),
                          get(boundary, 'maximum_step_joint'), get(pitch, 'target_before_rad') + ' → ' + get(pitch, 'target_after_rad'),
                          get(pitch, 'soft_excess_after_rad'), get(roll, 'soft_excess_after_rad')]
                lines.append('| ' + entry['label'] + ' | ' + side + ' | ' + ' | '.join(values) + ' |')
        lines += ['', 'The earlier coefficient-retention implementation changed whether the absolute target '
                  'was clamped when its weight became zero. Those acquisitions did not sample the raw expert '
                  'target. The later whole-operator implementation records base, projected and blended targets '
                  'and is checked separately against q_blend = q_base + r(q_projected − q_base), including exact '
                  'r=0/1 endpoints. Downstream residuals and changing policy state still affect the saved final '
                  'target. Remaining runtime soft bounds stay unavailable (19 joints in legacy acquisitions; '
                  '18 in the later acquisition). Original task success is separate from target continuity.']
    partial = [e for e in actual_entries if e['audit']['absolute_window_exit']['status'] == 'available']
    if partial:
        lines += ['', '## Partial release at the original absolute-window exit', '',
                  'A positive final retention can leave a projected correction when the original absolute '
                  'window becomes inactive. The internal blend then stops being applied by that original mask. '
                  'The offset below is computed from recorded internal targets at the same decision; it is '
                  'separate from the observed total target step across policy decisions.', '',
                  '| Run | Exit decision s | Strength | Retention | Bypassed projection max rad | Final target max step rad | Original v28 target max step rad |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for entry in partial:
            w = entry['audit']['absolute_window_exit']
            values = [get(w, key) for key in ('decision_time_s', 'release_strength', 'retention', 'maximum_bypassed_projection_offset_rad')]
            values += [get(w, 'final_target_boundary', 'maximum_absolute_target_step_rad'), get(w, 'reference_target_boundary', 'maximum_absolute_target_step_rad')]
            lines.append('| ' + entry['label'] + ' | ' + ' | '.join(values) + ' |')
        lines += ['', 'These measurements do not establish whole-command C2 continuity. All per-joint targets, '
                  'sample brackets and actually recorded soft bounds remain in the JSON.']
    lines += ['', '## Methods and provenance', '',
        'The protected prefix ends strictly before the first feedback decision whose recorded retention is '
        'less than one. It is not cut at the earlier trigger timestamp. Every original physical-trace, contact, '
        'kinematic and feedback value is compared exactly; added telemetry/debug fields are excluded. '
        'Policy timestamps are validated on the original 20 ms grid before selecting the prefix. The primary '
        'strike comparison covers only the first contiguous foot-ball force episode, its preimpact fixed-point '
        'velocity, and first zero-force outgoing ball state.', '',
        'Recorded weight pairs are checked against the declared mode: all scales swing, absolute and placement; '
        'pose preserves the swing weight and scales absolute/placement. The saved float32 multiplication allows '
        'an absolute tolerance of 5×10⁻⁷. The audit also checks nonincreasing retention and a completed velocity '
        'pulse whenever retention is below one. It does not infer unseen trigger state.', '',
        'P0 supplies pelvis-relative path and whole-run joint-limit diagnostics. P1 supplies force components, '
        'impulses, foot-velocity jumps, joint overspeed samples, segment motion and the unaccounted momentum '
        'residual. Probe certificates are independently recomputed from their stored samples. The release '
        'audit does not infer energy transfer, fix momentum residuals, or change any acceptance threshold.', '',
        '| Source | SHA-256 |', '|---|---|']
    for path, digest in result['analysis_sources_sha256'].items():
        lines.append(f'| `{path}` | `{digest}` |')
    reported_sources = set()
    for entry in result['entries']:
        p = entry['provenance']
        for key, digest_key in [('frozen_path', 'frozen_sha256'), ('instrumented_path', 'instrumented_sha256'), ('candidate_path', 'candidate_sha256'), ('continuous_pose_path', 'continuous_pose_sha256')]:
            if key in p and p[key] not in reported_sources:
                lines.append(f"| `{p[key]}` | `{p[digest_key]}` |")
                reported_sources.add(p[key])
    for _, data in runs:
        p = data['probe_provenance']
        if p.get('path') and p['path'] not in reported_sources:
            lines.append(f"| `{p['path']}` | `{p['sha256']}` |")
            reported_sources.add(p['path'])
    lines += ['', 'Controller hashes below are the values recorded by each acquisition, rather than '
              'claims that the current working files still have those contents. The raw-report hashes '
              'above anchor this metadata; the JSON also retains the resolved release configuration.', '',
              '| Acquisition | Source | Recorded SHA-256 |', '|---|---|---|']
    for label, data in runs:
        p = data['controller_provenance']
        for name, digest in (p.get('joint_strike_sources_sha256') or {}).items():
            if name in ('placement_strike.py', 'closed_loop_expert.py', 'landing_release.py'):
                lines.append(f'| {label} | `{name}` | `{digest}` |')
        lines.append(f"| {label} | Resolved joint-strike configuration | `{p.get('joint_strike_config_sha256')}` |")
    lines += ['', 'Acquisition source/probe fingerprints, exact prefix counts and first differing keys, '
              'signed force-component impulses, every observed overspeed joint, and all per-bin residuals '
              'remain in the JSON artifact. No simulation, GPU operation, state write, or model training is '
              'performed by this command.', '', '```sh', result['reproduction_command'], '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', nargs='+', type=Path)
    parser.add_argument('--dry-cases', nargs='+', default=['initial_slower_02', 'center', 'combined_right_faster'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    outputs = [p.resolve() for p in (args.output, args.markdown) if p is not None]
    if len(set(outputs)) != len(outputs):
        parser.error('Artifact destinations must differ')
    for path in outputs:
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    entries = []
    if args.candidates:
        for path in args.candidates:
            payload = path.read_bytes(); candidate = json.loads(payload)
            case = candidate['dynamics_telemetry']['registered_case']
            reference, observed, provenance = load_baseline(case)
            provenance.update(candidate_path=str(path.resolve().relative_to(ROOT)), candidate_sha256=hashlib.sha256(payload).hexdigest())
            audit = audit_candidate(reference, observed, candidate)
            if audit['release']['phase_lag_evidence'].get('expected'):
                continuous, source = load_continuous_reference(case)
                if continuous is None:
                    audit['continuous_pose_comparison'] = source
                else:
                    provenance.update(source)
                    identity = compare_frozen_replay(candidate, continuous)
                    physical_equal = all(identity[k] for k in ('records_equal', 'kinematics_equal', 'legacy_trace_fields_equal')) and candidate.get('contacts') == continuous.get('contacts')
                    audit['continuous_pose_comparison'] = dict(status='available', reference=describe_run(continuous), identity=identity,
                        physical_outputs_equal=physical_equal, definition='Exact recorded physical outputs; intentional policy/config/source hash changes are reported separately and are not motion failures')
            entries.append(dict(label=path.stem, provenance=provenance, audit=audit))
    else:
        for case in args.dry_cases:
            _, observed, provenance = load_baseline(case)
            entries.append(dict(label=case, provenance=provenance, audit=dict(mode='baseline_dry_analysis', reference=describe_run(observed))))
    sources = ['tools/audit_landing_release.py', 'tools/audit_p0_dynamic_strike.py', 'tools/explosive_strike_metrics.py',
               'tools/strike_baseline_metrics.py', 'tools/strike_dynamics_telemetry.py', 'tools/analyze_explosive_strike.py',
               'tools/placement_strike.py', 'tools/closed_loop_expert.py']
    command = 'python3 tools/audit_landing_release.py '
    command += ('--candidates ' + ' '.join(str(p) for p in args.candidates) if args.candidates else '--dry-cases ' + ' '.join(args.dry_cases))
    command += ' --output ' + str(args.output)
    if args.markdown:
        command += ' --markdown ' + str(args.markdown)
    command += ' --overwrite'
    result = dict(schema_version=2, analysis_version='0.1.30', entries=entries,
                  analysis_sources_sha256={name: sha(ROOT / name) for name in sources}, reproduction_command=command)
    protected = {str((ROOT / name).resolve()) for name in sources}
    for entry in entries:
        protected.update(str((ROOT / v).resolve()) for k, v in entry['provenance'].items() if k.endswith('_path'))
    if any(str(path) in protected for path in outputs):
        raise ValueError('An analysis output cannot overwrite a source')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_report(result))
    print(json.dumps(dict(output=str(args.output), modes=[entry['audit']['mode'] for entry in entries],
                         actual_verdicts={entry['label']: entry['audit']['verdict'] for entry in entries if 'verdict' in entry['audit']})))


if __name__ == '__main__':
    main()
