"""Matrix accounting must not hide unexecuted cases, failures, or source changes."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


def module():
    path = ROOT / 'tools/audit_landing_matrix.py'
    assert path.is_file(), 'The matrix aggregator is not implemented'
    spec = importlib.util.spec_from_file_location('audit_landing_matrix', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def studies():
    old = [dict(id='old_' + str(i), ball_offset_xy_m=[0., 0.], root_velocity_y_delta_mps=0.) for i in range(18)]
    fresh = [dict(id='fresh_' + str(i), ball_offset_xy_m=[.01, 0.], root_velocity_y_delta_mps=.1) for i in range(3)]
    base = dict(existing_cases=old, fresh_evaluation=fresh, candidate_config_sha256='config',
                runtime_sources_sha256={'tools/landing_release.py': 'guard2',
                    'tools/run_instrumented_strike.py': 'runner', 'tools/strike_dynamics_telemetry.py': 'observer'})
    result = {i: copy.deepcopy(base) for i in (1, 2, 3)}
    result[3]['runtime_sources_sha256']['tools/landing_release.py'] = 'guard3'
    return result


def report(case='old_0'):
    return dict(joint_strike=dict(config_sha256='config', source_sha256={'landing_release.py': 'guard2'},
            ball_offset_xy=[0., 0.], velocity_y_delta=0.),
        dynamics_telemetry=dict(registered_case=case, runner_source_sha256='runner', observer_source_sha256='observer'))


def entry(case, success, reasons, baseline_success=True, ratio=.9):
    original = dict(outcome=dict(success=success, failure_reasons=reasons),
        original_shot=dict(goal=success, recovered=True, surface_match=True), final_standing_seconds=8.)
    return dict(case_id=case, status='executed', original=original,
        baseline_original=dict(outcome=dict(success=baseline_success, failure_reasons=[] if baseline_success else ['missed_goal'])),
        joint_limit_observations=dict(maximum_actual_speed_ratio=ratio), provenance=dict(study_revision=2))


def test_registry_rejects_missing_duplicates_and_changes_to_the_fresh_group():
    m = module()
    for mutation in ('missing', 'duplicate', 'regroup'):
        values = studies()
        if mutation == 'missing': values[3]['existing_cases'].pop()
        if mutation == 'duplicate': values[3]['existing_cases'][1] = copy.deepcopy(values[3]['existing_cases'][0])
        if mutation == 'regroup':
            values[3]['existing_cases'][0], values[3]['fresh_evaluation'][0] = values[3]['fresh_evaluation'][0], values[3]['existing_cases'][0]
        with pytest.raises(ValueError): m.registry_from_studies(values)


def test_reports_require_registered_unique_known_case_ids(tmp_path):
    m = module(); registry = m.registry_from_studies(studies())
    p = tmp_path / 'one.json'; q = tmp_path / 'two.json'
    p.write_text(json.dumps(report())); q.write_text(json.dumps(report()))
    with pytest.raises(ValueError, match='Duplicate'): m.index_reports([p, q], registry)
    bad = report(); del bad['dynamics_telemetry']['registered_case']; p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='registered'): m.index_reports([p], registry)
    p.write_text(json.dumps(report('unknown')))
    with pytest.raises(ValueError, match='Unknown'): m.index_reports([p], registry)
    with pytest.raises(FileNotFoundError): m.index_reports([tmp_path / 'missing.json'], registry)


def test_partial_matrix_keeps_old_goal_miss_new_failure_and_unexecuted_separate():
    m = module(); registry = m.registry_from_studies(studies())
    result = m.aggregate(registry, [entry('old_0', False, ['missed_goal'], False),
        entry('old_1', False, ['recovery_failed']), entry('old_2', True, [], ratio=1.05)])
    assert result['complete'] is False
    assert result['groups']['development']['executed'] == 3
    assert result['groups']['development']['original_task_passed'] == 1
    assert result['groups']['development']['original_task_failed'] == 2
    assert result['groups']['development']['actual_velocity_limit_failed'] == 1
    assert result['groups']['fresh_evaluation'] == dict(planned=3, executed=0, unexecuted=3,
        original_task_passed=0, original_task_failed=0, actual_velocity_limit_failed=0)
    assert result['baseline_failures'] == ['old_0']
    assert result['original_task_regressions'] == ['old_1']
    assert result['velocity_quality_failures'] == ['old_2']
    assert result['cases'][0]['original']['outcome']['failure_reasons'] == ['missed_goal']
    assert result['cases'][3]['status'] == 'unexecuted'
    assert result['cases'][3]['original'] is None
    with pytest.raises(ValueError, match='Unexecuted'): m.aggregate(registry, [], require_complete=True)
    with pytest.raises(ValueError, match='Duplicate'): m.aggregate(registry, [entry('old_0', True, []), entry('old_0', True, [])])


def test_acquisition_hashes_match_their_own_revision_and_reject_unpinned_sources():
    m = module(); values = studies(); raw = report()
    assert m.match_study(raw, values) == 2
    raw['joint_strike']['source_sha256']['landing_release.py'] = 'guard3'
    assert m.match_study(raw, values) == 3
    raw['dynamics_telemetry']['observer_source_sha256'] = 'unknown'
    with pytest.raises(ValueError, match='pinned'): m.match_study(raw, values)


def finite_report():
    feedback = []
    for t in (0., .02, .04):
        feedback.append(dict(time_s=t, contact_latched=t > 0., landing_release=dict(
            phase_lag_s=.04, right_lowest_height_m=.08, min_vertical_velocity_mps=-1.,
            predicted_clearance_m=-.08, separated_s=.015 if t else 0., pulse_finished=bool(t), trigger_time_s=-1.)))
    return dict(joint_strike=dict(feedback=feedback, config={'landing_release': {'separation_s': .015}}),
        traces=[[dict(time_s=t, physics_dt_s=.005, support_foot_normal_force_w_N={'right': [0., 0., 0.]},
                      support_foot_force_N={'right': 0.}) for t in (.005, .01, .015, .02, .025, .03, .035, .04)]])


def test_finite_guard_evidence_discloses_initial_unmeasured_force_without_inventing_it():
    evidence = module().finite_guard_evidence(finite_report())
    assert evidence['guard_preservation_supported'] is True
    assert evidence['unobserved_ground_decision_times_s'] == [0.]
    assert evidence['unobserved_trigger_eligible_ground_times_s'] == []
    assert evidence['all_recorded_guard_inputs_finite'] is True


@pytest.mark.parametrize('bad_field', ['geometry', 'raw_normal'])
def test_nonfinite_guard_inputs_are_not_sanitized_into_finite_evidence(bad_field):
    raw = finite_report()
    if bad_field == 'geometry': raw['joint_strike']['feedback'][1]['landing_release']['min_vertical_velocity_mps'] = float('-inf')
    else: raw['traces'][0][3]['support_foot_normal_force_w_N']['right'][2] = float('nan')
    result = module().finite_guard_evidence(raw)
    assert result['guard_preservation_supported'] is False
    assert result['all_recorded_guard_inputs_finite'] is False


def test_diagnostic_deltas_keep_overspeed_regression_separate_from_smaller_motion_peak():
    before = dict(joint_limit_observations={'maximum_actual_speed_ratio': 1.04},
        sampled_clearance={'shank_to_shank': {'minimum_sampled_gap_m': .03}},
        dynamic_follow_through=dict(paths={'goal_horizontal': {'after_separation_tracked_point': {
            'pelvis_relative_advance_m': .12, 'pelvis_relative_displacement_at_250ms_m': -.02,
            'pelvis_relative_speed_at_120ms_mps': .1}}},
            measured_point_velocity={'maximum_post_separation_vector_step_mps': 7.}))
    after = copy.deepcopy(before)
    after['joint_limit_observations']['maximum_actual_speed_ratio'] = 1.05
    after['sampled_clearance']['shank_to_shank']['minimum_sampled_gap_m'] = .02
    after['dynamic_follow_through']['measured_point_velocity']['maximum_post_separation_vector_step_mps'] = 6.
    result = module().diagnostic_delta(before, after)
    assert result['maximum_actual_speed_ratio_delta'] == pytest.approx(.01)
    assert result['minimum_sampled_shank_gap_delta_m'] == pytest.approx(-.01)
    assert result['maximum_post_separation_vector_step_delta_mps'] == pytest.approx(-1.)
    assert result['pelvis_relative_advance_delta_m'] == 0.
