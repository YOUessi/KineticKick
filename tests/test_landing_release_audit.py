"""Independent CPU checks for release timing, frozen prefixes, and landing alignment."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


def audit_module():
    path = ROOT / 'tools/audit_landing_release.py'
    if not path.is_file():
        raise AssertionError('The independent landing-release audit is not implemented')
    spec = importlib.util.spec_from_file_location('audit_landing_release', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical(t, ball_force=0., support=0.):
    return dict(time_s=t, physics_dt_s=.005, contact_force_N=ball_force,
                foot='right', ball_xyz=[0., 0., .1], ball_velocity=[4. if t >= .030 else 0., 0., 0.],
                right_foot_point_velocity=[5., 0., 0.], joint_vel=[2.],
                support_foot_force_N={'left': 100., 'right': support}, contacts=[])


def report():
    return dict(traces=[[physical(i * .005) for i in range(1, 13)]],
                contacts=[[{'time_s': .020, 'force_N': 10.}, {'time_s': .050, 'force_N': 1.}]],
                kinematics=[[{'time_s': i * .02, 'joint_pos': [0.]} for i in range(3)]],
                joint_strike={'feedback': [{'time_s': i * .02, 'phase_s': i * .02} for i in range(3)]})


class ReleaseTimingTests(unittest.TestCase):
    def test_first_effective_release_uses_retention_not_earlier_trigger_timestamp(self):
        r = report()
        for f, retention in zip(r['joint_strike']['feedback'], [1., 1., .8]):
            f['landing_release'] = {'retention': retention, 'trigger_time_s': .02, 'mode': 'all'}
        r['joint_strike']['feedback'][2]['time_s'] = .0400000006
        m = audit_module().first_release_decision(r)
        self.assertEqual(m['first_retention_below_one_s'], .040)
        self.assertEqual(m['raw_decision_time_s'], .0400000006)
        self.assertEqual(m['debug']['trigger_time_s'], .020)

    def test_missing_release_debug_is_not_reported_as_an_unchanged_intervention(self):
        m = audit_module().first_release_decision(report())
        self.assertEqual(m['status'], 'unavailable')

    def test_invalid_retention_or_missing_policy_sample_is_rejected(self):
        r = report()
        for f in r['joint_strike']['feedback']:
            f['landing_release'] = {'retention': 1.}
        r['joint_strike']['feedback'][1]['landing_release']['retention'] = float('nan')
        with self.assertRaises(ValueError):
            audit_module().first_release_decision(r)
        r['joint_strike']['feedback'][1]['landing_release']['retention'] = .8
        r['joint_strike']['feedback'].pop(0)
        with self.assertRaises(ValueError):
            audit_module().first_release_decision(r)


class FrozenPrefixTests(unittest.TestCase):
    def test_changed_motion_after_decision_does_not_fail_the_protected_prefix(self):
        baseline = report(); candidate = copy.deepcopy(baseline)
        candidate['traces'][0][8]['joint_vel'] = [20.]
        candidate['kinematics'][0][2]['joint_pos'] = [1.]
        candidate['contacts'][0][1]['force_N'] = 20.
        m = audit_module().compare_prefix(baseline, candidate, .040)
        self.assertTrue(m['equal'])
        self.assertEqual(m['streams']['physical']['reference_count'], 7)

    def test_changed_or_missing_original_field_before_decision_fails(self):
        baseline = report(); candidate = copy.deepcopy(baseline)
        candidate['traces'][0][5]['joint_vel'] = [-2.]
        m = audit_module().compare_prefix(baseline, candidate, .040)
        self.assertFalse(m['equal'])
        self.assertEqual(m['streams']['physical']['first_difference_s'], .030)
        candidate = copy.deepcopy(baseline)
        del candidate['traces'][0][3]['joint_vel']
        self.assertFalse(audit_module().compare_prefix(baseline, candidate, .040)['equal'])

    def test_added_debug_is_allowed_but_changed_legacy_feedback_and_contact_are_not(self):
        baseline = report(); candidate = copy.deepcopy(baseline)
        candidate['joint_strike']['feedback'][0]['landing_release'] = {'retention': 1.}
        self.assertTrue(audit_module().compare_prefix(baseline, candidate, .040)['equal'])
        candidate['joint_strike']['feedback'][1]['phase_s'] += .01
        candidate['contacts'][0][0]['force_N'] += 1.
        m = audit_module().compare_prefix(baseline, candidate, .040)
        self.assertFalse(m['streams']['feedback']['equal'])
        self.assertFalse(m['streams']['contacts']['equal'])


class StrikeAndLandingTests(unittest.TestCase):
    def test_first_outgoing_episode_is_unchanged_despite_later_contact_differences(self):
        baseline = [physical(i*.005, 100. if i in [4, 5] else 0.) for i in range(1, 13)]
        candidate = copy.deepcopy(baseline)
        candidate[9]['contact_force_N'] = 500.
        m = audit_module().compare_primary_strike(baseline, candidate, [1, 0, 0])
        self.assertTrue(m['equal'])
        candidate[5]['ball_velocity'] = [3., 0., 0.]
        self.assertFalse(audit_module().compare_primary_strike(baseline, candidate, [1, 0, 0])['equal'])

    def test_first_poststrike_right_touchdown_ignores_earlier_support(self):
        rows = [physical(.005, support=100), physical(.010), physical(.015, ball_force=100),
                physical(.020), physical(.025, support=20), physical(.030, support=1000)]
        m = audit_module().first_right_touchdown(rows, .015)
        self.assertEqual(m['time_s'], .025)
        self.assertEqual(m['onset_bracket_s'], [.020, .025])

    def test_no_air_to_ground_transition_is_unavailable_not_the_largest_force_peak(self):
        rows = [physical(.005, support=100), physical(.010, ball_force=100, support=200),
                physical(.015, support=1000)]
        self.assertEqual(audit_module().first_right_touchdown(rows, .010)['status'], 'unavailable')

    def test_task_failure_cannot_be_hidden_by_an_improved_dynamic_peak(self):
        m = audit_module().release_verdict(
            {'equal': True}, {'equal': True}, {'success': False, 'failure_reasons': ['missed_goal']},
            {'match': True}, {'activation_observed': True, 'after_observed_separation': True})
        self.assertFalse(m['audit_pass'])
        self.assertIn('missed_goal', m['failure_reasons'])


class ReleaseEvidenceTests(unittest.TestCase):
    def test_whole_operator_formula_exact_endpoints_and_projected_bounds(self):
        r = report()
        names = ['right_' + n + '_joint' for n in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
        for row, retention in zip(r['joint_strike']['feedback'], (1., .5, 0.)):
            row['landing_release'] = dict(projection_semantics='whole_operator_retention', retention=retention,
                absolute_joint_names=names, absolute_base_target_rad=[2.]*6,
                absolute_projected_target_rad=[1.]*6, absolute_blended_target_rad=[2.-retention]*6,
                absolute_soft_limits_rad=[[-1., 1.]]*6)
        m = audit_module().projection_evidence(r)
        self.assertTrue(m['valid'])
        self.assertEqual(m['exact_endpoint_checks'], 2)
        r['joint_strike']['feedback'][1]['landing_release']['absolute_blended_target_rad'][0] = 1.9
        self.assertFalse(audit_module().projection_evidence(r)['valid'])
        r['joint_strike']['feedback'][1]['landing_release']['absolute_blended_target_rad'][0] = 1.5
        r['joint_strike']['feedback'][2]['landing_release']['absolute_blended_target_rad'][0] += 1e-8
        self.assertFalse(audit_module().projection_evidence(r)['valid'])

    def test_pose_retains_swing_while_all_releases_all_three_weights(self):
        for mode in ('all', 'pose'):
            r = report()
            for row, retention in zip(r['joint_strike']['feedback'], (1., .5, 0.)):
                debug = dict(mode=mode, retention=retention, pulse_finished=True)
                for name in ('swing', 'absolute', 'placement'):
                    debug[name + '_weight_original'] = .8
                    debug[name + '_weight_effective'] = .8 * (1. if name == 'swing' and mode == 'pose' else retention)
                row['landing_release'] = debug
            m = audit_module().release_weight_evidence(r)
            self.assertTrue(m['valid'])
            self.assertEqual(m['first_zero_retention_decision_s'], .04)
            r['joint_strike']['feedback'][1]['landing_release']['swing_weight_effective'] = .3
            self.assertFalse(audit_module().release_weight_evidence(r)['valid'])

    def test_missing_weight_evidence_cannot_certify_a_release(self):
        r = report()
        for row in r['joint_strike']['feedback']:
            row['landing_release'] = dict(mode='pose', retention=.5, pulse_finished=True)
        m = audit_module().release_weight_evidence(r)
        self.assertEqual(m['status'], 'unavailable')


class PhaseLagEvidenceTests(unittest.TestCase):
    def phase_report(self, zero=False):
        r = report()
        r['joint_strike']['config'] = {'landing_release': {'phase_lag_scale_s': .08, 'duration_s': .04}}
        rows = []
        for i in range(5):
            t = i * .02
            lag = 0. if zero else (.04 if i <= 1 else 0.)
            rows.append(dict(time_s=t, phase_s=t-lag, phase_action=-lag,
                landing_release=dict(phase_lag_s=lag, frame_phase_lag_s=lag,
                    release_strength=0. if zero or i == 0 else .5,
                    trigger_time_s=-1. if zero or i == 0 else .02,
                    retention=1. if zero or i <= 1 else (.75 if i == 2 else .5))))
        r['joint_strike']['feedback'] = rows
        return r

    def test_lag_seconds_and_frozen_strength_survive_later_lag_return_to_zero(self):
        r = self.phase_report()
        m = audit_module().phase_lag_evidence(r)
        self.assertTrue(m['valid'])
        self.assertEqual(m['trigger_phase_lag_s'], .04)
        self.assertEqual(m['frozen_release_strength'], .5)
        self.assertEqual(m['settled_retention'], .5)
        r['joint_strike']['feedback'][2]['landing_release']['release_strength'] = .6
        self.assertFalse(audit_module().phase_lag_evidence(r)['valid'])

    def test_frame_count_cannot_be_mistaken_for_lag_seconds(self):
        r = self.phase_report()
        r['joint_strike']['feedback'][1]['landing_release']['frame_phase_lag_s'] = 2.
        self.assertFalse(audit_module().phase_lag_evidence(r)['valid'])

    def test_zero_lag_requires_no_trigger_and_no_decay(self):
        r = self.phase_report(zero=True)
        m = audit_module().phase_lag_evidence(r)
        self.assertTrue(m['valid'])
        self.assertTrue(m['zero_lag_preservation_observed'])
        r['joint_strike']['feedback'][2]['landing_release']['retention'] = .9
        self.assertFalse(audit_module().phase_lag_evidence(r)['valid'])
        r = self.phase_report(zero=True)
        r['joint_strike']['feedback'][2]['landing_release'].update(trigger_time_s=.04, release_strength=.2)
        self.assertFalse(audit_module().phase_lag_evidence(r)['valid'])

    def test_zero_lag_preservation_needs_exact_full_run_even_with_no_release(self):
        a = report(); b = copy.deepcopy(a)
        a['records'] = b['records'] = [{'success': True}]
        m = audit_module().compare_full_legacy(a, b)
        self.assertTrue(m['equal'])
        b['traces'][0][-1]['joint_vel'] = [99.]
        self.assertFalse(audit_module().compare_full_legacy(a, b)['equal'])

    def test_partial_strength_inspects_original_window_exit_with_positive_retention(self):
        r = report(); module = audit_module(); r['joint_names'] = module.ABSOLUTE_JOINT_NAMES
        for row in r['traces'][0]:
            row['joint_target'] = ([1.5]*6 if row['time_s'] <= .04 else [2.]*6)
        for i, row in enumerate(r['joint_strike']['feedback']):
            row['landing_release'] = dict(retention=[1., .75, .5][i], release_strength=.5, trigger_time_s=0.,
                absolute_weight_original=[1., .25, 0.][i], absolute_joint_names=module.ABSOLUTE_JOINT_NAMES,
                absolute_base_target_rad=[2.]*6, absolute_projected_target_rad=[1.]*6,
                absolute_blended_target_rad=[1., 1.25, 1.5][i:i+1]*6,
                absolute_soft_limits_rad=[[-1., 1.]]*6)
        m = module.absolute_window_exit_evidence(r)
        self.assertEqual(m['decision_time_s'], .04)
        self.assertEqual(m['retention'], .5)
        self.assertEqual(m['maximum_bypassed_projection_offset_rad'], .5)
        self.assertEqual(m['final_target_boundary']['state_sample_times_s'], [.04, .045])

    def test_target_boundary_uses_first_postdecision_bin_and_observed_partial_bounds(self):
        r = report()
        r['joint_names'] = ['right_ankle_pitch_joint', 'right_knee_joint']
        for row in r['traces'][0]:
            row['joint_target'] = [.4, .8] if row['time_s'] <= .040 else [.7, .6]
        for row in r['joint_strike']['feedback']:
            row['approach_soft_limits_rad'] = [[-1., .5] for _ in range(10)]
        m = audit_module().target_release_boundary(r, .040)
        self.assertEqual(m['state_sample_times_s'], [.040, .045])
        ankle = m['joints'][0]
        self.assertAlmostEqual(ankle['target_delta_rad'], .3)
        self.assertAlmostEqual(ankle['soft_excess_after_rad'], .2)
        self.assertTrue(ankle['crossed_soft_bound'])
        self.assertEqual(m['joints'][1]['soft_bounds']['status'], 'unavailable')
        self.assertEqual(m['unobserved_soft_bound_joints'], ['right_knee_joint'])

    def test_missing_boundary_sample_does_not_substitute_a_later_command(self):
        r = report(); r['joint_names'] = ['right_ankle_pitch_joint']
        for row in r['traces'][0]:
            row['joint_target'] = [.4]
        r['traces'][0] = [row for row in r['traces'][0] if row['time_s'] <= .040]
        m = audit_module().target_release_boundary(r, .040)
        self.assertEqual(m['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
