#!/usr/bin/env python3
"""CPU aggregation of saved v30 episodes; no simulation, fitting, or case selection.

Original task outcomes, observed velocity limits, and additional motion
diagnostics remain separate. An absent planned episode is unexecuted, never a
pass. Historical v2 and guarded v3 acquisitions retain their distinct hashes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_REPRESENTATIVES = {'center', 'initial_slower_02', 'combined_right_faster', 'heldout_confirm_right_near_slower'}
SPEED_RATIO_TOLERANCE = 1.0001  # Declared study quality tolerance, not original task acceptance.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry_from_studies(studies):
    """Reject missing, duplicate, changed, or regrouped preregistered cases."""
    if set(studies) != {1, 2, 3}:
        raise ValueError('All three prespecified study registries are required')
    registries = []
    for revision in (1, 2, 3):
        registry = {}
        for source, group, count in [('existing_cases', 'development', 18), ('fresh_evaluation', 'fresh_evaluation', 3)]:
            cases = studies[revision].get(source, [])
            if len(cases) != count:
                raise ValueError('Missing or extra planned cases in study ' + str(revision))
            for case in cases:
                case_id = case.get('id')
                if not isinstance(case_id, str) or not case_id:
                    raise ValueError('Every planned case requires an id')
                if case_id in registry:
                    raise ValueError('Duplicate planned case: ' + case_id)
                xy, velocity = case.get('ball_offset_xy_m'), case.get('root_velocity_y_delta_mps')
                if (not isinstance(xy, list) or len(xy) != 2 or
                        not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [*xy, velocity])):
                    raise ValueError('Finite registered condition required: ' + case_id)
                registry[case_id] = dict(case_id=case_id, group=group, ball_offset_xy_m=xy,
                    root_velocity_y_delta_mps=velocity)
        registries.append(registry)
    if any(r != registries[0] for r in registries[1:]):
        raise ValueError('Study revisions changed a case or its development/fresh group')
    return registries[0]


def index_reports(paths, registry):
    """Index explicit saved acquisitions without inferring case identity from filenames."""
    result = {}
    for path in paths:
        path = Path(path).resolve()
        payload = path.read_bytes()
        run = json.loads(payload)
        case_id = run.get('dynamics_telemetry', {}).get('registered_case')
        if not isinstance(case_id, str) or not case_id:
            raise ValueError('Missing registered case in ' + str(path))
        if case_id not in registry:
            raise ValueError('Unknown registered case: ' + case_id)
        if case_id in result:
            raise ValueError('Duplicate candidate case: ' + case_id)
        case, saved = registry[case_id], run.get('joint_strike', {})
        if (saved.get('ball_offset_xy') != case['ball_offset_xy_m'] or
                saved.get('velocity_y_delta') != case['root_velocity_y_delta_mps']):
            raise ValueError('Report perturbation differs from registered case: ' + case_id)
        result[case_id] = dict(path=path, sha256=hashlib.sha256(payload).hexdigest())
    return result


def match_study(run, studies):
    """Match acquisition hashes to their own v2/v3 protocol, not today's source."""
    recorded = {'tools/' + k: v for k, v in run.get('joint_strike', {}).get('source_sha256', {}).items()}
    telemetry = run.get('dynamics_telemetry', {})
    recorded['tools/run_instrumented_strike.py'] = telemetry.get('runner_source_sha256')
    recorded['tools/strike_dynamics_telemetry.py'] = telemetry.get('observer_source_sha256')
    matches = [revision for revision in (2, 3)
        if run.get('joint_strike', {}).get('config_sha256') == studies[revision]['candidate_config_sha256']
        and all(recorded.get(k) == v for k, v in studies[revision]['runtime_sources_sha256'].items())]
    if len(matches) != 1:
        raise ValueError('Acquisition does not uniquely match pinned study v2/v3 sources')
    return matches[0]


def finite_guard_evidence(run):
    """Check saved inputs to the v3 trigger guard, disclosing unobserved t=0 force.

    The raw normal vector avoids falsely treating legacy max(0, NaN) output as
    finite force evidence. The force is an independent view of the same ground
    filter, not an instrumented log of the controller's getter call.
    """
    feedback = run.get('joint_strike', {}).get('feedback', [])
    physical = [r for r in run.get('traces', [[]])[0] if r.get('physics_dt_s', 0) > 0]
    times = {round(r['time_s'], 6): r for r in physical}
    invalid, missing, eligible_missing = [], [], []
    keys = ('phase_lag_s', 'right_lowest_height_m', 'min_vertical_velocity_mps', 'predicted_clearance_m')
    separation = run.get('joint_strike', {}).get('config', {}).get('landing_release', {}).get('separation_s', .015)
    for row in physical:
        vector = row.get('support_foot_normal_force_w_N', {}).get('right')
        if not isinstance(vector, list) or len(vector) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
            invalid.append(dict(time_s=row['time_s'], field='raw_right_ground_normal_vector'))
    for row in feedback:
        t, debug = row.get('time_s'), row.get('landing_release', {})
        for key, value in [('time_s', t)] + [(k, debug.get(k)) for k in keys]:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                invalid.append(dict(time_s=t if isinstance(t, (int, float)) and math.isfinite(t) else None, field=key))
        if not isinstance(t, (int, float)) or not math.isfinite(t):
            continue
        # Saved float32 policy times are within 2 us of the exact 20 ms grid.
        observed_time = round(round(t / .02) * .02, 6)
        if observed_time not in times:
            missing.append(observed_time)
            eligible = (row.get('contact_latched') and debug.get('pulse_finished') and
                debug.get('separated_s', 0.) + 1e-8 >= separation and debug.get('phase_lag_s', 0.) > 1e-6 and
                debug.get('trigger_time_s', -1.) < 0.)
            if eligible:
                eligible_missing.append(observed_time)
    finite = bool(feedback and physical) and not invalid
    return dict(all_recorded_guard_inputs_finite=finite, invalid_inputs=invalid,
        decision_count=len(feedback), physical_sample_count=len(physical),
        unobserved_ground_decision_times_s=missing,
        unobserved_trigger_eligible_ground_times_s=eligible_missing,
        guard_preservation_supported=finite and not eligible_missing,
        interpretation='Finite saved geometry/time/lag and raw normal-force samples support the finite-only guard comparison. Initial t=0 ground force was not recorded; a false earlier gate makes it trigger-ineligible. This is sampled observational evidence, not identical source code or per-getter instrumentation.')


def aggregate(registry, entries, require_complete=False, baseline_originals=None):
    """Count only executed outcomes and preserve failures independently of diagnostics."""
    indexed = {}
    for entry in entries:
        case_id = entry.get('case_id')
        if case_id not in registry:
            raise ValueError('Unknown candidate case: ' + str(case_id))
        if case_id in indexed:
            raise ValueError('Duplicate candidate case: ' + case_id)
        if entry.get('status') != 'executed' or not isinstance(entry.get('original', {}).get('outcome', {}).get('success'), bool):
            raise ValueError('Executed case requires an original boolean task outcome: ' + case_id)
        indexed[case_id] = entry
    missing = [case_id for case_id in registry if case_id not in indexed]
    if require_complete and missing:
        raise ValueError('Unexecuted planned cases: ' + ', '.join(missing))
    cases = []
    baseline_originals = baseline_originals or {}
    for case_id, planned in registry.items():
        entry = indexed.get(case_id, dict(case_id=case_id, status='unexecuted', original=None,
            baseline_original=baseline_originals.get(case_id)))
        cases.append({**planned, **entry})
    groups = {}
    for group in ('development', 'fresh_evaluation'):
        planned = [x for x in cases if x['group'] == group]
        measured = [x for x in planned if x['status'] == 'executed']
        groups[group] = dict(planned=len(planned), executed=len(measured), unexecuted=len(planned)-len(measured),
            original_task_passed=sum(x['original']['outcome']['success'] for x in measured),
            original_task_failed=sum(not x['original']['outcome']['success'] for x in measured),
            actual_velocity_limit_failed=sum(x['joint_limit_observations']['maximum_actual_speed_ratio'] > SPEED_RATIO_TOLERANCE for x in measured))
    return dict(complete=not missing, groups=groups, unexecuted=missing,
        baseline_failures=[x['case_id'] for x in cases if x.get('baseline_original') is not None and not x['baseline_original']['outcome']['success']],
        velocity_quality_failures=[x['case_id'] for x in cases if x['status'] == 'executed'
            and x['joint_limit_observations']['maximum_actual_speed_ratio'] > SPEED_RATIO_TOLERANCE],
        original_task_regressions=[x['case_id'] for x in cases if x['status'] == 'executed' and x.get('baseline_original') is not None
            and x['baseline_original']['outcome']['success'] and not x['original']['outcome']['success']], cases=cases)


def diagnostic_delta(before, after):
    """Signed observed differences, without inventing a naturalness score."""
    def value(data, path):
        for key in path:
            if not isinstance(data, dict): return None
            data = data.get(key)
        return data
    paths = {
        'maximum_actual_speed_ratio_delta': ('joint_limit_observations', 'maximum_actual_speed_ratio'),
        'minimum_sampled_shank_gap_delta_m': ('sampled_clearance', 'shank_to_shank', 'minimum_sampled_gap_m'),
        'maximum_post_separation_vector_step_delta_mps': ('dynamic_follow_through', 'measured_point_velocity', 'maximum_post_separation_vector_step_mps'),
    }
    for label, key in [('pelvis_relative_advance_delta_m', 'pelvis_relative_advance_m'),
                       ('pelvis_relative_displacement_at_250ms_delta_m', 'pelvis_relative_displacement_at_250ms_m'),
                       ('pelvis_relative_speed_at_120ms_delta_mps', 'pelvis_relative_speed_at_120ms_mps')]:
        paths[label] = ('dynamic_follow_through', 'paths', 'goal_horizontal', 'after_separation_tracked_point', key)
    result = {}
    for label, path in paths.items():
        a, b = value(before, path), value(after, path)
        result[label] = b-a if a is not None and b is not None else None
    return result


def p0_diagnostics(run, reference, urdf):
    from audit_p0_dynamic_strike import audit_case, CapsuleGeometry, contact_geometry, limit_observations
    from strike_baseline_metrics import summarize_strike, validated_rows
    original = summarize_strike(run)
    contact = original['contact']
    poses = run.get('kinematics', [[]])[0]
    last_origin = max(contact['first_force_s'], contact.get('outgoing_time_s') or contact['first_force_s']) if contact else None
    if contact is None or (poses and poses[-1]['time_s'] >= last_origin + .75):
        return audit_case(run, reference, urdf)
    # A genuine early physical failure must remain an outcome, not disappear
    # because the retrospective 750 ms motion diagnostic lacks coverage.
    import numpy as np
    rows = validated_rows(run['traces'][0])
    gaps = CapsuleGeometry(urdf, run['joint_names']).gaps([r['joint_pos'] for r in rows])
    clearance = {k: dict(minimum_sampled_gap_m=float(v.min()), time_s=rows[int(np.argmin(v))]['time_s'],
                        negative_sample_count=int(np.sum(v < 0))) for k, v in gaps.items()}
    geometry = contact_geometry(rows)
    geometry.pop('_normals', None); geometry.pop('_axes', None)
    return dict(original=original, joint_limit_observations=limit_observations(rows, run), sampled_clearance=clearance,
        contact_geometry=geometry, dynamic_follow_through=None,
        diagnostic_unavailable_reason='Physical episode ended before required follow-through coverage',
        original_shot_replay_matches=None)


def build_matrix(paths, require_complete=False):
    from audit_landing_release import audit_candidate, describe_run, load_baseline
    from audit_p0_dynamic_strike import performance_delta
    study_paths = {i: ROOT / f'configs/landing_release_study_v{i}.json' for i in (1, 2, 3)}
    studies = {i: json.loads(p.read_text()) for i, p in study_paths.items()}
    registry = registry_from_studies(studies)
    difference = {k for k in studies[2]['runtime_sources_sha256'] | studies[3]['runtime_sources_sha256']
                  if studies[2]['runtime_sources_sha256'].get(k) != studies[3]['runtime_sources_sha256'].get(k)}
    if difference != {'tools/landing_release.py'} or studies[2]['candidate_config_sha256'] != studies[3]['candidate_config_sha256']:
        raise ValueError('v2/v3 difference is outside the declared single-source finite guard amendment')
    config_path = ROOT / studies[3]['candidate_config']
    config = json.loads(config_path.read_text())
    if sha(config_path) != studies[3]['candidate_config_sha256']:
        raise ValueError('Current resolved candidate configuration differs from protocol pin')
    frozen = {name: dict(expected=digest, actual=sha(ROOT / name)) for name, digest in config['frozen_files_sha256'].items()}
    if len(frozen) != 19 or any(v['actual'] != v['expected'] for v in frozen.values()):
        raise ValueError('Original 19 frozen files changed or are incomplete')
    comparison_path = ROOT / 'reports/v028_final_comparison.json'
    comparison = json.loads(comparison_path.read_text())
    old = {c['case']['id']: c for c in comparison['cases']}
    if set(old) != {k for k, v in registry.items() if v['group'] == 'development'}:
        raise ValueError('The original 18 development cases differ from the published v28 registry')
    if paths is None:
        paths = [ROOT / f'reports/v030_lag_pose_{case}.json' for case in registry
                 if (ROOT / f'reports/v030_lag_pose_{case}.json').is_file()]
    indexed = index_reports(paths, registry)
    if require_complete and set(indexed) != set(registry):
        raise ValueError('Unexecuted planned cases: ' + ', '.join(k for k in registry if k not in indexed))
    entries = []
    urdf = ROOT / 'configs/paid_original/paid_main_resolved.urdf'
    for case_id, source in indexed.items():
        if sha(source['path']) != source['sha256']:
            raise ValueError('Candidate changed during offline audit: ' + case_id)
        run = json.loads(source['path'].read_text())
        revision = match_study(run, studies)
        if revision == 2 and case_id not in V2_REPRESENTATIVES:
            raise ValueError('Unexpected study-v2 acquisition outside four declared representatives: ' + case_id)
        if run['joint_strike'].get('config') != config:
            raise ValueError('Saved resolved configuration differs from its pinned candidate: ' + case_id)
        finite = finite_guard_evidence(run)
        baseline = None; baseline_p0 = None
        if case_id in old:
            reference, observed, baseline = load_baseline(case_id)
            p1 = audit_candidate(reference, observed, run)
            p0 = p0_diagnostics(run, reference, urdf)
            baseline_p0 = p0_diagnostics(reference, reference, urdf)
            if baseline_p0['original'] != old[case_id]['candidate']:
                raise ValueError('Recomputed v28 summary differs from its published comparison: ' + case_id)
            if p0['original']['outcome'] != p1['candidate']['original_metrics']['outcome']:
                raise ValueError('P0/P1 original outcome disagreement: ' + case_id)
        else:
            # Fresh cases have no matched v28 counterfactual. The nominal
            # contract is used only for the P0 implementation's other metrics.
            nominal, _, _ = load_baseline('center')
            p0 = p0_diagnostics(run, nominal, urdf)
            p0['invariants'] = dict(status='unavailable', reason='No matched v28 initial-condition baseline')
            p1 = dict(mode='candidate_only', candidate=describe_run(run),
                      verdict=dict(audit_pass=None, reason='No matched v28 baseline; protected prefix/strike comparison unavailable'))
        original = p0.pop('original')
        limits = p0.pop('joint_limit_observations')
        previous = old[case_id]['candidate'] if case_id in old else None
        differences = diagnostic_delta(baseline_p0, {**p0, 'joint_limit_observations': limits}) if baseline_p0 else None
        baseline_motion = ({key: baseline_p0[key] for key in ('joint_limit_observations', 'sampled_clearance', 'dynamic_follow_through')}
                           if baseline_p0 else None)
        entries.append(dict(case_id=case_id, status='executed', original=original, baseline_original=previous,
            baseline_comparison='available' if baseline else 'unavailable',
            provenance=dict(candidate_path=str(source['path'].relative_to(ROOT)), candidate_sha256=source['sha256'],
                study_revision=revision, study_path=str(study_paths[revision].relative_to(ROOT)), study_sha256=sha(study_paths[revision]),
                acquisition_sources_sha256=run['joint_strike']['source_sha256'],
                acquisition_config_sha256=run['joint_strike']['config_sha256'], baseline=baseline),
            finite_guard_evidence=finite, joint_limit_observations=limits, p0_diagnostics=p0, p1_audit=p1,
            baseline_p0_motion_and_clearance=baseline_motion, diagnostic_deltas_vs_v28=differences,
            relative_to_v28=performance_delta(previous, original)))
        print(case_id + ': original=' + str(original['outcome']['success']) + ', speed_ratio=' + str(limits['maximum_actual_speed_ratio']), flush=True)
    result = aggregate(registry, entries, require_complete, {k: v['candidate'] for k, v in old.items()})
    revisions = sorted({x['provenance']['study_revision'] for x in entries})
    result.update(schema_version=1, study='v30 landing matrix',
        study_sources={str(i): dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for i, p in study_paths.items()},
        v28_comparison=dict(path=str(comparison_path.relative_to(ROOT)), sha256=sha(comparison_path)),
        v28_reports={case: dict(path=entry['candidate_report'], sha256=entry['candidate_sha256']) for case, entry in old.items()},
        frozen_files=frozen, all_19_frozen_files_match=True,
        source_cohorts=dict(study_revisions=revisions, identical_acquisition_source_code=len(revisions) <= 1,
            counts={str(i): sum(e['provenance']['study_revision'] == i for e in entries) for i in revisions},
            v2_finite_guard_preservation_supported=all(e['finite_guard_evidence']['guard_preservation_supported'] for e in entries if e['provenance']['study_revision'] == 2)),
        limits=dict(actual_velocity_ratio_tolerance=SPEED_RATIO_TOLERANCE,
            definition='Additional declared velocity-quality gate; original task outcomes are reported independently'),
        analysis_sources_sha256={name: sha(ROOT / name) for name in ['tools/audit_landing_matrix.py', 'tools/audit_p0_dynamic_strike.py', 'tools/audit_landing_release.py']},
        limitations=[
            'Unexecuted rows are missing measurements, never successes or failures. Development and prespecified fresh evaluation are counted separately.',
            'v2 and v3 acquisitions use distinct source hashes. Saved finite-input evidence supports the finite-only guard amendment; it does not make the binaries or source files identical.',
            'P0/P1 follow-through, force, impulse and pose diagnostics are additional; no diagnostic threshold is promoted to original task acceptance or human naturalness.',
            'URDF capsule gaps are sampled at recorded joint positions and do not guarantee continuous clearance.',
            'A force probe verifies sensor semantics; it does not certify complete external robot forces or explain every COM momentum residual.',
            'Finite condition counts are deterministic observations, not estimated population success rates.'])
    return result


def render_report(result):
    def number(value, digits=4):
        return 'unavailable' if value is None else f'{value:.{digits}f}'
    rows = ['# v30 landing matrix: saved evidence', '',
        'Matrix complete: **' + str(result['complete']).lower() + '**. Unexecuted conditions remain unexecuted.', '',
        '| Group | Planned | Executed | Original task pass / fail | Velocity-quality failures | Unexecuted |',
        '|---|---:|---:|---:|---:|---:|']
    for group, x in result['groups'].items():
        rows.append(f"| {group} | {x['planned']} | {x['executed']} | {x['original_task_passed']} / {x['original_task_failed']} | {x['actual_velocity_limit_failed']} | {x['unexecuted']} |")
    for x in result['cases']:
        if x['case_id'] not in result['velocity_quality_failures']: continue
        limits = x['joint_limit_observations']; worst = limits['worst_actual_speed']
        rows += ['', f"**Declared velocity stop criterion observed:** `{x['case_id']}`, `{worst['joint']}` at {worst['time_s']:.3f} s: {worst['absolute_speed_rad_s']:.6f} / {worst['limit_rad_s']:.1f} rad/s, ratio {limits['maximum_actual_speed_ratio']:.9f}; {limits['actual_speed_exceedance_sample_joint_count']} sample-joint exceedance(s). Its original task result remains reported separately."]
    rows += ['', 'Acquisition source cohorts: `' + json.dumps(result['source_cohorts']['counts'], sort_keys=True) + '`; these are different source hashes. The v2 saved finite-input checks support the subsequent finite-only trigger guard amendment: **' + str(result['source_cohorts']['v2_finite_guard_preservation_supported']).lower() + '**. Initial t=0 ground force is unrecorded and trigger-ineligible.', '',
        'The original 19 frozen files match. Original v28 failures remain historical failures: ' + ', '.join(result['baseline_failures']) + '.', '',
        '| Case | State / source | Original outcome | Instep match | XY ball speed (m/s) | Standing (s) | Max actual qd / limit | Sampled shank gap (mm) | Δ speed / Δ standing vs v28 |',
        '|---|---|---|---|---:|---:|---:|---:|---|']
    for x in result['cases']:
        if x['status'] == 'unexecuted':
            prior = x.get('baseline_original')
            old = '; v28: ' + ', '.join(prior['outcome']['failure_reasons']) if prior and not prior['outcome']['success'] else ''
            rows.append(f"| {x['case_id']} | unexecuted | unexecuted{old} | — | — | — | — | — | — |")
            continue
        o = x['original']; contact = o.get('contact') or {}; shot = o['original_shot']
        failures = o['outcome']['failure_reasons']; verdict = 'pass' if o['outcome']['success'] else ', '.join(failures) or 'failed'
        gap = x['p0_diagnostics'].get('sampled_clearance', {}).get('shank_to_shank', {}).get('minimum_sampled_gap_m')
        delta = x['relative_to_v28'] or {}
        rows.append(f"| {x['case_id']} | executed / v{x['provenance']['study_revision']} | {verdict} | {shot['surface_match']} | {number(contact.get('outgoing_speed_xy_mps'))} | {number(o['final_standing_seconds'], 3)} | {number(x['joint_limit_observations']['maximum_actual_speed_ratio'], 6)} | {number(None if gap is None else gap*1000, 3)} | {number(delta.get('outgoing_speed_xy_delta_mps'))} / {number(delta.get('final_standing_delta_s'), 3)} |")
    rows += ['', '## Measured follow-through diagnostics', '',
        'These are additional P0 measurements, not original task gates or human-naturalness standards. Relative motion subtracts pelvis translation; world travel also includes body carry.', '',
        '| Case | Relative forward advance after separation (m) | Relative displacement at 250 ms (m) | Relative speed at 120 ms (m/s) | Relative reversal (s) | Measured point velocity jump (m/s) |',
        '|---|---:|---:|---:|---:|---:|']
    for x in result['cases']:
        if x['status'] != 'executed': continue
        dynamic = x['p0_diagnostics'].get('dynamic_follow_through') or {}
        path = dynamic.get('paths', {}).get('goal_horizontal', {}).get('after_separation_tracked_point', {})
        jump = dynamic.get('measured_point_velocity', {}).get('maximum_post_separation_vector_step_mps')
        values = [path.get(k) for k in ('pelvis_relative_advance_m', 'pelvis_relative_displacement_at_250ms_m', 'pelvis_relative_speed_at_120ms_mps', 'pelvis_relative_reversal_after_s')]
        rows.append('| ' + x['case_id'] + ' | ' + ' | '.join(number(v) for v in [*values, jump]) + ' |')
    rows += ['', '## Additional measured differences versus the same v28 condition', '',
        'Signed candidate-minus-v28 differences; these measurements do not define a new task pass or naturalness score.', '',
        '| Case | Δ max qd ratio | Δ sampled shank gap (mm) | Δ relative forward advance (m) | Δ relative speed at 120 ms (m/s) | Δ measured velocity jump (m/s) |',
        '|---|---:|---:|---:|---:|---:|']
    for x in result['cases']:
        if x['status'] != 'executed': continue
        delta = x['diagnostic_deltas_vs_v28'] or {}
        values = [delta.get('maximum_actual_speed_ratio_delta'),
            None if delta.get('minimum_sampled_shank_gap_delta_m') is None else delta['minimum_sampled_shank_gap_delta_m']*1000,
            delta.get('pelvis_relative_advance_delta_m'), delta.get('pelvis_relative_speed_at_120ms_delta_mps'),
            delta.get('maximum_post_separation_vector_step_delta_mps')]
        rows.append('| ' + x['case_id'] + ' | ' + ' | '.join(number(v) for v in values) + ' |')
    rows += ['', '## Scope and provenance', ''] + ['- ' + x for x in result['limitations']]
    rows += ['', 'Actual-velocity failures use the declared additional ratio tolerance of 1.0001; task-success rows can still fail that quality check. No matrix-wide no-regression or acceptance-completion claim is made.', '',
        'Machine-readable evidence retains each raw report path/hash, matched protocol revision, original failure reasons, P0 geometry/path diagnostics, and the full P1 audit.', '']
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', nargs='*', type=Path, help='Explicit reports; default is saved v030_lag_pose files for registered cases')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    outputs = [p.resolve() for p in (args.output, args.markdown) if p is not None]
    if len(outputs) != len(set(outputs)):
        raise ValueError('Output destinations must differ')
    for path in outputs:
        if path.exists() and not args.overwrite: raise FileExistsError(path)
    result = build_matrix(args.candidates, args.require_complete)
    protected = {(ROOT / k).resolve() for k in result['analysis_sources_sha256']}
    protected |= {(ROOT / v['path']).resolve() for v in result['study_sources'].values()}
    protected |= {(ROOT / x['provenance']['candidate_path']).resolve() for x in result['cases'] if x['status'] == 'executed'}
    protected |= {(ROOT / v['path']).resolve() for v in result['v28_reports'].values()}
    for entry in result['cases']:
        if entry['status'] != 'executed': continue
        baseline = entry['provenance'].get('baseline') or {}
        protected |= {(ROOT / v).resolve() for k, v in baseline.items() if k.endswith('_path')}
        for side in ('reference', 'candidate'):
            probe = entry['p1_audit'].get(side, {}).get('probe_provenance', {})
            if probe.get('path'): protected.add((ROOT / probe['path']).resolve())
    protected |= {(ROOT / k).resolve() for k in result['frozen_files']}
    protected |= {ROOT / 'configs/state_conditioned_strike_v6.json', ROOT / 'reports/v028_final_comparison.json'}
    for path in outputs:
        if path in protected: raise ValueError('Analysis cannot overwrite source or raw evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_report(result))
    print(json.dumps(dict(output=str(args.output), complete=result['complete'], groups=result['groups'])))


if __name__ == '__main__':
    main()
