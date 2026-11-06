#!/usr/bin/env python3
"""CPU snapshot of the fixed v32 recovery matrix; --require-complete enforces full acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_landing_release import compare_primary_strike
from audit_post_shot_recovery import audit_candidate, physical_result, residual_evidence, model_provenance, source_cohort, sha
from explosive_strike_metrics import contact_metrics, unavailable
from strike_baseline_metrics import validated_episode

ROOT = Path(__file__).resolve().parents[1]
STRICT = {'center', 'initial_slower_02', 'combined_right_faster', 'combined_left_slower',
          'heldout_left_near_slower', 'heldout_confirm_right_near_slower'}


def planned_cases(study):
    if len(study['existing_cases']) != 18 or len(study['fresh_evaluation']) != 3:
        raise ValueError('Fixed 18 development plus 3 fresh conditions required')
    cases = [dict(c, group=g) for key, g in [('existing_cases', 'development'), ('fresh_evaluation', 'fresh')] for c in study[key]]
    if len({c['id'] for c in cases}) != 21:
        raise ValueError('Duplicate planned condition')
    return cases


def load_completed(path):
    try:
        payload = path.read_bytes(); run = json.loads(payload)
        j, d = run['joint_strike'], run['dynamics_telemetry']
        for key in ('registered_case', 'runner_source_sha256', 'observer_source_sha256', 'support_ground_force', 'physical_writes'):
            if key not in d: raise ValueError('Incomplete outer observer metadata: '+key)
        for key in ('config', 'config_sha256', 'source_sha256', 'feedback', 'checkpoint', 'checkpoint_sha256', 'ball_offset_xy', 'velocity_y_delta'):
            if key not in j: raise ValueError('Incomplete outer controller metadata: '+key)
        if not run['traces'][0] or not run['records'] or not run['sequence_records'] or not j['feedback']:
            raise ValueError('Incomplete physical/controller report')
        return run, dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return None, unavailable(str(error), 'completed report including final outer acquisition metadata')


def file_pins(expected):
    actual = {name: sha(ROOT/name) if (ROOT/name).is_file() else None for name in expected}
    return dict(passed=bool(expected) and actual == expected, expected_sha256=expected, actual_sha256=actual)


def recorded_cohort(run, case, study, config, frozen):
    j, d = run['joint_strike'], run['dynamics_telemetry']
    recorded = {'tools/'+k: v for k, v in j['source_sha256'].items()}
    recorded.update({'tools/run_instrumented_strike.py': d['runner_source_sha256'], 'tools/strike_dynamics_telemetry.py': d['observer_source_sha256']})
    not_recorded = {study['candidate_config'], 'docs/research/2026-11-07-post-shot-recovery-learning.md', 'tools/train_post_shot_recovery.py'}
    if 'ground_contact_detail' in d: recorded['tools/ground_contact_observer.py'] = d['ground_contact_detail']['source_sha256']
    else: not_recorded.add('tools/ground_contact_observer.py')
    changed = [k for k, v in study['runtime_sources_sha256'].items() if k not in not_recorded and recorded.get(k) != v]
    checks = dict(case=d['registered_case'] == case['id'] and j['ball_offset_xy'] == case['ball_offset_xy_m']
                  and j['velocity_y_delta'] == case['root_velocity_y_delta_mps'], source_hashes=not changed,
        config=j['config_sha256'] == study['candidate_config_sha256'] and j['config'] == config,
        model=j['checkpoint_sha256'] == study['checkpoint_sha256'] == j['config']['post_shot_recovery']['checkpoint_sha256'], mode=j['mode'] == 'post_shot_residual',
        original_19_mapping=j['config']['frozen_files_sha256'] == frozen)
    return dict(passed=all(checks.values()), checks=checks, changed_sources=changed,
                not_recorded_by_replay=sorted(not_recorded), acquisition=source_cohort(run))


def analyze_case(run, case, study, config, frozen):
    rows = validated_episode(run); j = run['joint_strike']; first = next((i for i, r in enumerate(rows) if r['contact_force_N'] > 0), 0)
    direction = np.r_[np.asarray(run['goals'][0])-np.asarray(rows[max(0, first-1)]['ball_xyz'])[:2], 0.]
    direction /= np.linalg.norm(direction)
    physical = physical_result(run); residual = residual_evidence(run, physical['completed_physics_seconds'])
    model = model_provenance(run, residual, ROOT); cohort = recorded_cohort(run, case, study, config, frozen)
    strike = contact_metrics(rows, direction); shot = run['records'][0]; sequence = run['sequence_records'][0]
    no_writes = run['post_initialization_state_writes'] is False and run['root_fixed'] is False and run['dynamics_telemetry']['physical_writes'] is False
    no_writes &= all(f['post_shot_recovery']['ball_state_write'] is False for f in j['feedback'])
    no_writes &= all(h['physical_state_write'] is False for h in j['expert_handoffs'])
    legal = shot['surface_match'] and shot['measured_surface'] == 'instep' and shot['requested_style'] == 'instep' and not sequence['illegal_body_ball_contact']
    legal &= sequence['all_body_contact_coverage_complete'] and sequence['contribution_foot_matches_contact']
    isolated = unavailable('No matched v6 control; v28-to-v32 also includes the v30 landing change', 'same-condition v30-v6 physical replay')
    comparison = unavailable('Fresh condition has no executed historical control', 'matched historical report')
    if case['group'] == 'fresh': isolated = comparison
    historical = comparison; compare_pass = case['group'] == 'fresh'
    prefix = isolated
    if case['group'] == 'development':
        path = ROOT/'reports'/((f"v030_lag_pose_{case['id']}.json") if case['id'] in STRICT else f"v028_final_{case['id']}.json")
        if path.is_file():
            reference = json.loads(path.read_text()); before = physical_result(reference)
            condition_matches = all(reference['joint_strike'][k] == run['joint_strike'][k] for k in ('ball_offset_xy', 'velocity_y_delta'))
            historical = dict(path=str(path), sha256=sha(path), original_task=before['outcome'], goal=reference['records'][0]['goal'],
                condition_matches=condition_matches, final_standing_seconds=before['final_standing_seconds'], scope='matched_v6' if case['id'] in STRICT else 'v28_strike_only_includes_v30_change')
            comparison = compare_primary_strike(validated_episode(reference), rows, direction)
            comparison.update(status='available', passed=comparison['equal'] and condition_matches); compare_pass = comparison['passed']
            if case['id'] in STRICT:
                strict = audit_candidate(reference, run); prefix = strict['checks']['protected_prefix']
                isolated = dict(status='available', passed=strict['acceptance_pass'], failed_checks=strict['failed_checks'])
                compare_pass &= strict['acceptance_pass']
        else: historical = comparison = unavailable('Historical report absent', str(path))
    dynamic = all(physical['checks'][k] for k in ('complete_14s', 'actual_joint_speed', 'no_physical_termination', 'original_recovery'))
    return dict(original_task=physical['outcome']['success'], goal=shot['goal'], legal_instep=bool(legal),
        complete_14s=physical['checks']['complete_14s'], stable=physical['checks']['original_recovery'],
        dynamic_quality_pass=dynamic, physical=physical, residual=residual, model=model, source_cohort=cohort,
        first_strike=strike, first_strike_speed_xy_mps=strike.get('outgoing_ball', {}).get('speed_xy_mps', unavailable('No separated strike', 'complete contact episode')),
        historical_reference=historical, primary_strike_comparison=comparison, isolated_post_residual_prefix=prefix,
        strict_post_residual_audit=isolated, no_state_writes=bool(no_writes),
        candidate_acceptance_pass=bool(physical['passed'] and legal and residual.get('passed') and model.get('passed') and cohort['passed'] and no_writes and compare_pass))


def summarize(planned, entries):
    indexed = {e['case_id']: e for e in entries}
    if len(indexed) != len(entries) or set(indexed)-{c['id'] for c in planned}: raise ValueError('Duplicate or unknown reported case')
    cases = [dict(case_id=c['id'], group=c['group'], **{k: v for k, v in indexed.get(c['id'],
        dict(status='unexecuted', evidence=unavailable('No completed report', 'complete physical replay'))).items() if k != 'case_id'}) for c in planned]
    groups = {}
    for group in ('development', 'fresh'):
        all_cases = [c for c in cases if c['group'] == group]; done = [c for c in all_cases if c['status'] == 'executed']
        groups[group] = dict(planned=len(all_cases), executed=len(done), unexecuted=len(all_cases)-len(done),
            original_task_passed=sum(c.get('original_task') is True for c in done), original_task_failed=sum(c.get('original_task') is False for c in done), goal_misses=sum(c.get('goal') is False for c in done),
            legal_instep=sum(c.get('legal_instep') is True for c in done), complete_14s=sum(c.get('complete_14s') is True for c in done),
            stable=sum(c.get('stable') is True for c in done), dynamic_quality_failures=sum(c.get('dynamic_quality_pass') is False for c in done),
            analysis_unavailable=sum('original_task' not in c for c in done))
    complete = bool(cases) and all(c['status'] == 'executed' for c in cases)
    executed = sum(c['status'] == 'executed' for c in cases)
    return dict(planned=len(cases), executed=executed, unexecuted=len(cases)-executed, complete=complete, matrix_acceptance_pass=complete and all(c.get('candidate_acceptance_pass') is True for c in cases),
                groups=groups, cases=cases, isolated_full_matrix_effect=unavailable('Matched v6 controls exist for only six development cases', 'remaining matched controls'))


def markdown(result):
    lines = ['# v32 recovery matrix snapshot', '', f"Complete: {result['complete']}; matrix acceptance: {result['matrix_acceptance_pass']}.", '',
        'Six cases permit strict v6 comparison; twelve permit v28 strike comparison only; fresh cases have no historical control.', '',
        '| Case | Group | Status | Task | Goal | Instep | Dynamic quality | Standing s | qd/limit | Ball m/s | Control |',
        '|---|---|---|---|---|---|---|---:|---:|---:|---|']
    for c in result['cases']:
        value = lambda k: 'yes' if c.get(k) is True else 'no' if c.get(k) is False else 'unavailable'
        p = c.get('physical', {}); ratio = p.get('actual_joint_speed', {}).get('maximum_actual_speed_ratio')
        number = lambda x: f'{x:.6f}' if isinstance(x, (int, float)) else 'unavailable'
        control = 'strict v6' if c['case_id'] in STRICT else 'v28 strike only' if c['group'] == 'development' else 'unavailable'
        lines.append('| '+' | '.join([c['case_id'], c['group'], c['status'], value('original_task'), value('goal'), value('legal_instep'), value('dynamic_quality_pass'),
            number(p.get('final_standing_seconds')), number(ratio), number(c.get('first_strike_speed_xy_mps')), control])+' |')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true', help='Exit 1 unless all 21 cases completed and accepted')
    args = parser.parse_args(); md = args.output.with_suffix('.md')
    if args.output.exists() or md.exists(): raise FileExistsError('Choose a new JSON/MD snapshot path')
    study_path = ROOT/'configs/post_shot_recovery_study_v1.json'; study = json.loads(study_path.read_text()); planned = planned_cases(study)
    config = json.loads((ROOT/study['candidate_config']).read_text()); frozen = json.loads((ROOT/'configs/strike_baseline_v1.json').read_text())['frozen_files_sha256']
    runtime, original = file_pins(study['runtime_sources_sha256']), file_pins(frozen)
    original['passed'] &= len(frozen) == 19
    entries = []
    for case in planned:
        path = ROOT/'reports'/f"v032_learned_{case['id']}.json"; run, evidence = load_completed(path)
        if run is not None and (run['dynamics_telemetry']['registered_case'] != case['id'] or run['joint_strike']['ball_offset_xy'] != case['ball_offset_xy_m'] or run['joint_strike']['velocity_y_delta'] != case['root_velocity_y_delta_mps']):
            run, evidence = None, unavailable('Saved report belongs to a different registered condition', str(path))
        entry = dict(case_id=case['id'], status='unexecuted' if run is None else 'executed', evidence=evidence)
        if run is not None:
            try: entry.update(analyze_case(run, case, study, config, frozen))
            except (KeyError, ValueError, TypeError, IndexError, OSError) as error: entry.update(candidate_acceptance_pass=False, analysis=unavailable(str(error), 'complete valid physical evidence'))
        entries.append(entry)
    result = summarize(planned, entries); result['matrix_acceptance_pass'] &= runtime['passed'] and original['passed']
    result.update(study=dict(path=str(study_path), sha256=sha(study_path)), runtime_local_pins=runtime, original_19_files=original,
        analysis_sources_sha256={name: sha(ROOT/'tools'/name) for name in ('audit_recovery_matrix.py', 'audit_post_shot_recovery.py', 'audit_landing_release.py', 'explosive_strike_metrics.py', 'strike_baseline_metrics.py')},
        semantics='Only complete outer acquisition metadata counts as executed. Goal miss and dynamic quality are independent. Local pins are not evidence of unrecorded acquisition sources. Training proxy is not acceptance.')
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n'); md.write_text(markdown(result))
    print(json.dumps({k: result[k] for k in ('complete', 'matrix_acceptance_pass', 'groups')}))
    return 1 if args.require_complete and not result['matrix_acceptance_pass'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
