#!/usr/bin/env python3
"""CPU acceptance of one 14 s post-shot residual replay; training reward is not acceptance.

Exit 0 means all required checks passed; exit 1 includes measured failures or
unavailable evidence. Existing output files are never overwritten.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_landing_release import FIXED_INPUTS, _compare_stream, _policy_time, compare_prefix, compare_primary_strike
from audit_p0_dynamic_strike import limit_observations
from explosive_strike_metrics import unavailable
from strike_baseline_metrics import task_outcome, validated_episode

ROOT = Path(__file__).resolve().parents[1]
DT = .02
RAD_ATOL = 2e-6  # Roundoff in float32 action addition and conversion back to radians.
FIXED = FIXED_INPUTS + (
    'controller_config', 'config_sha256', 'controller_sources_sha256', 'sequence_sources_sha256',
    'motion_files', 'contribution_foot', 'foot_velocity_point_local', 'contact_evidence_version',
    'contact_sampling', 'sequence_evidence_version', 'policy_goals', 'requested_style',
    'seed', 'trials', 'perturb_reset', 'targeted_config', 'targeted_config_sha256')
CONFIG_DESCRIPTIONS = {'version', 'status', 'scope', 'parent_version', 'parent_commit', 'fitted_model'}
CHECK_NAMES = ('fixed_inputs', 'original_configuration', 'original_19_files', 'residual',
               'protected_prefix', 'primary_strike', 'physical_result', 'no_state_writes', 'model_provenance')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def equal_fields(a, b, names):
    missing = [key for key in names if key not in a or key not in b]
    changed = [key for key in names if key in a and key in b and a[key] != b[key]]
    return dict(status='unavailable' if missing else 'available', passed=not missing and not changed,
                missing_fields=missing, changed_fields=changed, compared_fields=list(names))


def residual_evidence(result, duration_s):
    feedback = result.get('joint_strike', {}).get('feedback', [])
    required = ('actual_delta_rad', 'previous_delta_rad', 'baseline_action', 'final_action')
    if not feedback or any(not isinstance(row.get('post_shot_recovery'), dict)
                           or any(key not in row['post_shot_recovery'] for key in required)
                           for row in feedback):
        return unavailable('Complete applied residual observations are absent',
                           'every 20 ms decision: 29 actual/previous deltas and baseline/final actions')
    cfg = result['joint_strike'].get('config', {}).get('post_shot_recovery', {})
    previous, first, amplitude, step, violations = np.zeros(29), None, 0., 0., []
    expected_count = int(math.ceil(duration_s/DT - 1e-7))
    if len(feedback) != expected_count:
        violations.append('decision_coverage')
    for i, row in enumerate(feedback):
        t = _policy_time(row['time_s'])
        if abs(t-i*DT) > 1e-9:
            violations.append('decision_grid')
        d = row['post_shot_recovery']
        arrays = [np.asarray(d[key], dtype=float) for key in required]
        if any(a.shape != (29,) or not np.isfinite(a).all() for a in arrays):
            return unavailable('Applied residual evidence is not finite and 29 dimensional', *required)
        delta, saved_previous, base, final = arrays
        if not np.array_equal(saved_previous, previous):
            violations.append('previous_applied_delta')
        if np.any((delta == 0.) != (final == base)):
            violations.append('zero_delta_action_identity')
        amplitude = max(amplitude, float(np.max(abs(delta))))
        step = max(step, float(np.max(abs(delta-previous))))
        if first is None and np.any(delta != 0.):
            first = t
        previous = delta
    if cfg.get('maximum_rad') != .12 or cfg.get('maximum_step_rad') != .02:
        violations.append('declared_fixed_bounds')
    if amplitude > .12+RAD_ATOL:
        violations.append('amplitude_bound')
    if step > .02+RAD_ATOL:
        violations.append('step_bound')
    return dict(status='available', passed=not violations, violations=sorted(set(violations)),
                decision_count=len(feedback), expected_decision_count=expected_count,
                joint_count=29, first_nonzero_applied_decision_s=first,
                all_applied_residuals_exactly_zero=first is None,
                maximum_absolute_applied_rad=amplitude, maximum_absolute_step_rad=step,
                amplitude_bound_rad=.12, policy_step_bound_rad=.02, numerical_tolerance_rad=RAD_ATOL,
                definition='Strict nonzero applied delta, not activation or raw actor output. '
                           'Bounds apply to the 29 residual corrections, not total PD targets.')


def physical_result(result):
    rows = validated_episode(result)
    if len(result['joint_names']) != 29 or len(set(result['joint_names'])) != 29:
        raise ValueError('Exactly 29 unique ordered joints required')
    for key in ('joint_velocity_limits', 'joint_effort_limits'):
        limits = np.asarray(result[key], float)
        if limits.shape != (1, 29) or not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError('Finite positive original joint limits required')
    for key in ('joint_vel', 'joint_velocity_target', 'joint_torque_estimate'):
        values = np.asarray([row[key] for row in rows], float)
        if values.shape != (len(rows), 29) or not np.isfinite(values).all():
            raise ValueError('Complete finite ordered joint telemetry required: '+key)
    original = task_outcome(result)
    limits = limit_observations(rows, result)
    recovery = result['sequence_records'][0]['recovery']
    standing = recovery['final_standing_seconds']
    if not isinstance(standing, (int, float)) or not math.isfinite(standing) or standing < 0:
        raise ValueError('Finite observed final-standing duration required')
    reasons = result['termination_reasons']
    checks = dict(
        complete_14s=len(rows) == 2800 and abs(rows[-1]['time_s']-14.) < 1e-7
                     and all(abs(row['physics_dt_s']-.005) < 1e-9 for row in rows),
        original_task=original['success'],
        actual_joint_speed=limits['actual_speed_exceedance_sample_joint_count'] == 0,
        no_physical_termination=reasons == [[]] and not any(row.get('terminated', False) for row in result['traces'][0]),
        original_recovery=recovery['passed'] is True and standing > 0 and recovery['terminated_or_fallen'] is False)
    return dict(status='available', passed=all(checks.values()), checks=checks, outcome=original,
                completed_physics_seconds=rows[-1]['time_s'], physical_steps=len(rows),
                terminal_status_markers=len(result['traces'][0])-len(rows), termination_reasons=reasons,
                actual_joint_speed={key: limits[key] for key in ('maximum_actual_speed_ratio',
                    'actual_speed_exceedance_sample_joint_count', 'worst_actual_speed')},
                recovery=recovery, final_standing_seconds=standing,
                final_stable_standing_start_s=rows[-1]['time_s']-standing if standing > 0 else None,
                definition='Original task/recovery plus all-run measured qdot against original limits. '
                           'Clipped-PD torque estimates and raw target soft-position excess are not actual joint-limit tests.')


def protected_prefix(reference, candidate, residual):
    if residual.get('status') != 'available':
        return unavailable('First applied residual is unknown', 'complete applied residual debug')
    first = residual['first_nonzero_applied_decision_s']
    cutoff = first if first is not None else max(r['traces'][0][-1]['time_s'] for r in (reference, candidate))+DT
    result = compare_prefix(reference, candidate, cutoff)
    if first is not None:
        result['physical_through_decision'] = _compare_stream(
            [r for r in reference['traces'][0] if r['time_s'] <= first+1e-9],
            [r for r in candidate['traces'][0] if r['time_s'] <= first+1e-9])
        result['equal'] &= result['physical_through_decision']['equal']
    else:
        result['zero_model_original_outcomes'] = equal_fields(reference, candidate,
            ('records', 'sequence_records', 'running_evidence', 'stride_shot_evidence', 'stride_shot_success'))
        result['equal'] &= result['zero_model_original_outcomes']['passed']
    result.update(status='available', passed=result['equal'], decision_time_s=first,
                  definition='Every original field before the first strictly nonzero applied residual; '
                  'the physical bin ending at that decision also holds the previous command and must match. '
                  'With no applied residual, require entire original streams and outcomes to match. Added debug is ignored.')
    return result


def fixed_files(reference, candidate, root):
    expected = json.loads((root/'configs/strike_baseline_v1.json').read_text())['frozen_files_sha256']
    recorded = [r['joint_strike']['config']['frozen_files_sha256'] for r in (reference, candidate)]
    files = [dict(path=name, expected_sha256=digest,
                  actual_sha256=sha(root/name) if (root/name).is_file() else None)
             for name, digest in expected.items()]
    return dict(status='available', passed=len(expected) == 19 and all(m == expected for m in recorded)
                and all(f['expected_sha256'] == f['actual_sha256'] for f in files),
                count=len(files), recorded_mappings_match=all(m == expected for m in recorded), files=files,
                definition='Original 19 recorded fingerprints and current local bytes; independent of learned checkpoint.')


def source_cohort(result):
    joint = result.get('joint_strike', {})
    metadata = result.get('dynamics_telemetry', {})
    return dict(policy=result.get('policy'), policy_sha256=result.get('policy_sha256'),
                config_sha256=joint.get('config_sha256'), config_version=joint.get('config', {}).get('version'),
                controller_sources_sha256=result.get('controller_sources_sha256'),
                sequence_sources_sha256=result.get('sequence_sources_sha256'),
                joint_strike_sources_sha256=joint.get('source_sha256'),
                telemetry={k: metadata.get(k) for k in ('registered_case', 'observer_source_sha256',
                    'runner_source_sha256', 'probe_path', 'probe_sha256')},
                checkpoint=joint.get('config', {}).get('post_shot_recovery', {}).get('checkpoint'),
                checkpoint_sha256=joint.get('config', {}).get('post_shot_recovery', {}).get('checkpoint_sha256'),
                training_configuration=joint.get('config', {}).get('post_shot_training'))


def model_provenance(candidate, residual, root):
    config = candidate['joint_strike']['config']['post_shot_recovery']
    path, expected = config.get('checkpoint'), config.get('checkpoint_sha256')
    if path is None:
        return dict(status='available', passed=expected is None and residual.get('all_applied_residuals_exactly_zero') is True,
                    kind='zero_actor', checkpoint=None, checkpoint_sha256=expected)
    path = root/Path(path)
    if not path.is_file():
        return unavailable('Declared learned checkpoint cannot be rehashed', str(path), 'recorded checkpoint SHA256')
    actual = sha(path)
    return dict(status='available', passed=isinstance(expected, str) and actual == expected,
                kind='learned_actor', checkpoint=str(path), expected_sha256=expected, actual_sha256=actual)


def audit_candidate(reference, candidate, root=ROOT):
    """Fail closed per check while retaining other measured results."""
    checks = {}
    def checked(name, function):
        try:
            checks[name] = function()
        except (KeyError, ValueError, TypeError, IndexError, StopIteration, OSError) as error:
            checks[name] = unavailable(str(error), 'complete original and candidate report evidence for '+name)
        return checks[name]

    def inputs():
        top = equal_fields(reference, candidate, FIXED)
        joint = equal_fields(reference['joint_strike'], candidate['joint_strike'],
            ('template_sha256', 'initial_root_velocity', 'ball_offset_xy', 'velocity_y_delta', 'expert_backend'))
        return dict(status='available' if top['status'] == joint['status'] == 'available' else 'unavailable',
                    passed=top['passed'] and joint['passed'], physical=top, condition=joint)
    checked('fixed_inputs', inputs)
    def configuration():
        remove = CONFIG_DESCRIPTIONS | {'post_shot_recovery', 'post_shot_training'}
        a, b = ({k: v for k, v in r['joint_strike']['config'].items() if k not in remove}
                for r in (reference, candidate))
        return equal_fields(a, b, sorted(set(a) | set(b)))
    checked('original_configuration', configuration)
    checked('original_19_files', lambda: fixed_files(reference, candidate, Path(root)))
    observed = checked('physical_result', lambda: physical_result(candidate))
    checked('residual', lambda: residual_evidence(candidate, observed['completed_physics_seconds']))
    checked('protected_prefix', lambda: protected_prefix(reference, candidate, checks['residual']))
    def strike():
        a, b = validated_episode(reference), validated_episode(candidate)
        i = next(i for i, row in enumerate(a) if row['contact_force_N'] > 0)
        direction = np.r_[np.asarray(reference['goals'][0])[:2]-np.asarray(a[max(0, i-1)]['ball_xyz'])[:2], 0.]
        direction /= np.linalg.norm(direction)
        value = compare_primary_strike(a, b, direction)
        value.update(status='available', passed=value['equal'])
        return value
    checked('primary_strike', strike)
    def writes():
        feedback = candidate['joint_strike']['feedback']
        handoffs = candidate['joint_strike']['expert_handoffs']
        flags = dict(free_root=candidate['root_fixed'] is False,
            no_post_initialization_writes=candidate['post_initialization_state_writes'] is False,
            observer_no_writes=candidate['dynamics_telemetry']['physical_writes'] is False,
            every_debug_no_ball_write=bool(feedback) and all(row['post_shot_recovery']['ball_state_write'] is False for row in feedback),
            every_handoff_no_write=all(h['physical_state_write'] is False for h in handoffs))
        return dict(status='available', passed=all(flags.values()), flags=flags,
                    definition='Recorded runtime invariants; a trace alone cannot prove absence of unlogged writes.')
    checked('no_state_writes', writes)
    checked('model_provenance', lambda: model_provenance(candidate, checks['residual'], Path(root)))
    failed = [name for name in CHECK_NAMES if checks[name].get('passed') is not True]
    reference_physical = None
    try:
        reference_physical = physical_result(reference)
    except (KeyError, ValueError, TypeError, IndexError) as error:
        reference_physical = unavailable(str(error), 'complete reference physical report')
    return dict(schema_version=1, status='available', acceptance_pass=not failed, failed_checks=failed,
                checks=checks, reference_physical=reference_physical,
                source_cohort=dict(reference=source_cohort(reference), candidate=source_cohort(candidate)),
                scope='One independent single-environment 14 s physical replay. Training proxy/reward is never acceptance. '
                      'This compact audit does not certify generalized recovery, hardware performance or naturalness.',
                event_aligned_landing_analysis=unavailable('Outside this bounded acceptance audit',
                    'a separately authorized event-aligned follow-up after acceptance'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('reference', 'candidate', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = dict(reference=args.reference.resolve(), candidate=args.candidate.resolve())
    evidence, reports = {}, {}
    for name, path in paths.items():
        if path.is_file():
            payload = path.read_bytes()
            evidence[name] = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())
            try:
                reports[name] = json.loads(payload)
                if not isinstance(reports[name], dict):
                    raise ValueError('Report must be an object')
            except (ValueError, TypeError) as error:
                evidence[name].update(unavailable(str(error), 'complete physical report JSON'))
                reports.pop(name, None)
        else:
            evidence[name] = unavailable('Report was not executed or is absent', str(path))
    if len(reports) == 2:
        result = audit_candidate(reports['reference'], reports['candidate'])
    else:
        result = dict(schema_version=1, status='unexecuted', acceptance_pass=False,
                      failed_checks=list(CHECK_NAMES), checks={name: unavailable('Physical replay evidence unavailable',
                      'complete reference and candidate reports') for name in CHECK_NAMES})
    result['input_reports'] = evidence
    result['analysis_sources_sha256'] = {str(Path('tools')/name): sha(ROOT/'tools'/name) for name in (
        'audit_post_shot_recovery.py', 'audit_landing_release.py', 'audit_p0_dynamic_strike.py',
        'strike_baseline_metrics.py', 'explosive_strike_metrics.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), acceptance_pass=result['acceptance_pass'], failed_checks=result['failed_checks'])))
    return 0 if result['acceptance_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
