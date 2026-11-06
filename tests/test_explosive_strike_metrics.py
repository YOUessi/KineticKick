"""Analytic checks for event timing, signed dynamics, and evidence boundaries."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


def row(t, force=0., velocity=2., torque=3., support=100., com=2.):
    return dict(time_s=t, physics_dt_s=.005, contact_force_N=force,
                foot='right', ball_velocity=[0., 0., 0.], ball_xyz=[0., 0., .1],
                joint_vel=[velocity], joint_torque_estimate=[torque],
                right_foot_point_velocity=[velocity, 0., 0.],
                support_foot_force_N={'left': support, 'right': 0.},
                whole_body_com_velocity=[com, 0., 0.], terminated=False)


def dynamics_metadata():
    return {'robot_total_mass_kg': 2., 'support_ground_force': {
        'includes_normal': True, 'includes_friction': True, 'force_on': 'robot',
        'frame': 'world', 'ground_filtered': True, 'coverage_complete': True,
        'sign_probe_verified': True}}


def instrumented_row(t):
    r = row(t)
    r.update(support_foot_normal_force_w_N={'left': [0., 0., 10.], 'right': [0., 0., 20.]},
             support_foot_friction_force_w_N={'left': [1., 0., 0.], 'right': [-3., 0., 0.]},
             support_foot_ground_force_w_N={'left': [1., 0., 10.], 'right': [-3., 0., 20.]},
             support_contact_capacity_exhausted=False)
    return r


def load_metrics():
    path = ROOT / 'tools' / 'explosive_strike_metrics.py'
    if not path.is_file():
        raise AssertionError('Event-aligned explosive-strike metrics are not implemented')
    spec = importlib.util.spec_from_file_location('explosive_strike_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SignedTimingTests(unittest.TestCase):
    def test_kinematic_component_integrals_keep_sign_and_check_point_velocity_closure(self):
        rows = [row(.005, velocity=6), row(.010, velocity=5), row(.015, force=20)]
        rows[0]['foot_velocity_contributions'] = {'hip': [1, 0, 0], 'knee': [-2, 0, 0],
                                                 'ankle': [3, 0, 0], 'base': [4, 0, 0]}
        rows[1]['foot_velocity_contributions'] = {'hip': [-1, 0, 0], 'knee': [2, 0, 0],
                                                 'ankle': [0, 0, 0], 'base': [4, 0, 0]}
        metrics = load_metrics()
        self.assertTrue(hasattr(metrics, 'kinematic_contributions'), 'Kinematic evidence must remain distinct from work')
        m = metrics.kinematic_contributions(rows, .015, [1, 0, 0], .015)
        self.assertAlmostEqual(m['parts']['hip']['signed_integral_m'], 0.)
        self.assertAlmostEqual(m['parts']['hip']['positive_integral_m'], .005)
        self.assertAlmostEqual(m['parts']['knee']['negative_integral_m'], -.010)
        self.assertAlmostEqual(m['point_velocity_closure_max_error_mps'], 0.)

    def test_signed_extrema_keep_negative_peak_and_tied_plateaus(self):
        m = load_metrics().signed_extrema(
            [.005, .010, .015, .020, .025, .030, .035], [0, 4, 4, 1, -6, -6, 2], .04)
        self.assertEqual(m['positive']['value'], 4.)
        self.assertEqual(m['positive']['times_s'], [.010, .015])
        self.assertEqual(m['negative']['value'], -6.)
        self.assertAlmostEqual(m['absolute']['time_to_contact_s'], -.015)
        self.assertEqual(m['absolute']['value'], -6.)
        self.assertEqual(len(m['positive_local_extrema']), 2)
        self.assertEqual(m['positive_local_extrema'][0]['plateau_end_s'], .015)
        self.assertTrue(m['positive_local_extrema'][1]['window_boundary'])

    def test_absent_sign_has_no_fabricated_zero_peak_time(self):
        m = load_metrics().signed_extrema([.005, .010, .015], [-1, -2, -1], .02)
        self.assertIsNone(m['positive'])
        self.assertEqual(m['positive_local_extrema'], [])
        self.assertEqual(m['negative']['time_s'], .010)

    def test_absolute_magnitude_ties_preserve_each_sample_sign(self):
        m = load_metrics().signed_extrema([.005, .010], [-4, 4], .015)
        self.assertEqual(m['absolute'].get('values'), [-4., 4.])

    def test_peak_timing_rejects_nonfinite_or_unsorted_samples(self):
        metrics = load_metrics()
        for times, values in [([.01, .005], [1, 2]), ([.005, .01], [1, math.nan])]:
            with self.assertRaises(ValueError):
                metrics.signed_extrema(times, values, .02)

    def test_power_work_and_peaks_exclude_collision_and_clip_window_bins(self):
        rows = [row(.005, velocity=2, torque=3), row(.010, velocity=-4, torque=3),
                row(.015, velocity=1, torque=-2), row(.020, force=100, velocity=100, torque=100)]
        m = load_metrics().joint_dynamics(rows, ['knee'], .020, window_s=.0175)
        j = m['joints']['knee']
        self.assertAlmostEqual(m['measured_duration_s'], .0125)
        self.assertAlmostEqual(j['power']['positive_work_J'], .015)
        self.assertAlmostEqual(j['power']['negative_work_J'], -.070)
        self.assertEqual(j['velocity']['peaks']['negative']['value'], -4.)
        self.assertEqual(j['power']['peaks']['positive']['time_s'], .005)
        self.assertEqual(j['power']['peaks']['negative']['time_s'], .010)

    def test_missing_torque_does_not_erase_velocity_or_imply_zero_work(self):
        rows = [row(.005), row(.010), row(.015, force=20)]
        del rows[1]['joint_torque_estimate']
        j = load_metrics().joint_dynamics(rows, ['knee'], .015, .015)['joints']['knee']
        self.assertEqual(j['velocity']['status'], 'available')
        self.assertEqual(j['power']['status'], 'unavailable')
        self.assertIn('joint_torque_estimate', j['power']['evidence_needed'][0])


class SupportBrakingTests(unittest.TestCase):
    def test_braking_uses_last_loaded_support_and_signed_com_velocity(self):
        rows = [row(.005, support=100, com=10), row(.010, support=0, com=9),
                row(.015, support=100, com=3), row(.020, com=2),
                row(.025, com=2.5), row(.030, force=50, com=-100)]
        m = load_metrics().support_braking(rows, .030, [1, 0, 0], .030)
        self.assertEqual(m['first_loaded_sample_s'], .015)
        np.testing.assert_allclose(m['support_onset_bracket_s'], [.010, .015])
        self.assertAlmostEqual(m['precontact_loaded_duration_s'], .015)
        p = m['com_kinematics']
        self.assertAlmostEqual(p['delta_forward_velocity_mps'], -.5)
        self.assertAlmostEqual(p['accumulated_speed_decrease_mps'], 1.)
        self.assertAlmostEqual(p['peak_deceleration_mps2'], 200.)
        self.assertEqual(m['horizontal_ground_reaction']['status'], 'unavailable')

    def test_vertical_force_never_substitutes_for_horizontal_ground_reaction(self):
        rows = [row(.005, support=9999), row(.010, support=9999), row(.015, force=20)]
        m = load_metrics().support_braking(rows, .015, [1, 0, 0], .015)
        self.assertEqual(m['horizontal_ground_reaction']['status'], 'unavailable')
        self.assertNotIn('braking_impulse_Ns', m['horizontal_ground_reaction'])

    def test_uncertified_vector_force_is_unavailable_and_certified_force_keeps_sign(self):
        rows = [row(.005), row(.010), row(.015, force=20)]
        rows[0]['support_foot_ground_force_w_N'] = {'left': [-10, 0, 100]}
        rows[1]['support_foot_ground_force_w_N'] = {'left': [5, 0, 100]}
        metrics = load_metrics()
        a = metrics.support_braking(rows, .015, [1, 0, 0], .015)
        self.assertEqual(a['horizontal_ground_reaction']['status'], 'unavailable')
        metadata = {'support_ground_force': {'includes_normal': True, 'includes_friction': True,
                    'force_on': 'robot', 'frame': 'world', 'ground_filtered': True,
                    'coverage_complete': True, 'sign_probe_verified': True}}
        b = metrics.support_braking(rows, .015, [1, 0, 0], .015, metadata=metadata)
        force = b['horizontal_ground_reaction']
        self.assertEqual(force['status'], 'available')
        self.assertAlmostEqual(force['signed_forward_impulse_Ns'], -.025)
        self.assertAlmostEqual(force['braking_impulse_Ns'], -.050)
        self.assertAlmostEqual(force['propulsive_impulse_Ns'], .025)

    def test_missing_com_does_not_invent_braking(self):
        rows = [row(.005), row(.010), row(.015, force=20)]
        del rows[0]['whole_body_com_velocity']
        m = load_metrics().support_braking(rows, .015, [1, 0, 0], .015)
        self.assertEqual(m['com_kinematics']['status'], 'unavailable')


class ContactAndRecoveryTests(unittest.TestCase):
    def test_primary_impulse_does_not_merge_separate_contacts(self):
        rows = [row(.005, velocity=5), row(.010, force=100), row(.015, force=200),
                row(.020), row(.025, force=500), row(.030)]
        m = load_metrics().contact_metrics(rows, [1, 0, 0])
        self.assertAlmostEqual(m['normal_impulse_Ns'], 1.5)
        self.assertAlmostEqual(m['all_episodes_normal_impulse_Ns'], 4.)
        self.assertEqual(m['episode_count'], 2)
        self.assertEqual(m['separation_sample_s'], .020)
        self.assertEqual(m['preimpact_foot']['speed_mps'], 5.)
        self.assertEqual(m['preimpact_foot']['time_s'], .005)

    def test_missing_preimpact_foot_velocity_preserves_measured_impulse(self):
        rows = [row(.005), row(.010, force=100), row(.015)]
        del rows[0]['right_foot_point_velocity']
        m = load_metrics().contact_metrics(rows, [1, 0, 0])
        self.assertAlmostEqual(m['normal_impulse_Ns'], .5)
        self.assertEqual(m['preimpact_foot']['status'], 'unavailable')

    def test_simultaneous_feet_contact_does_not_attribute_all_impulse_to_right_foot(self):
        rows = [row(.005), row(.010, force=100), row(.015)]
        rows[1]['contacts'] = [{'foot': 'left'}, {'foot': 'right'}]
        m = load_metrics().contact_metrics(rows, [1, 0, 0])
        self.assertEqual(m['observed_contact_feet'], ['left', 'right'])
        self.assertEqual(m['preimpact_foot']['status'], 'unavailable')

    def test_no_contact_and_incomplete_separation_are_explicit(self):
        metrics = load_metrics()
        self.assertEqual(metrics.contact_metrics([row(.005), row(.010)], [1, 0, 0])['status'], 'unavailable')
        rows = [row(.005), row(.010, force=20)]
        contact = metrics.contact_metrics(rows, [1, 0, 0])
        self.assertIsNone(contact['separation_sample_s'])
        self.assertEqual(metrics.follow_through(rows, contact, [1, 0, 0])['status'], 'unavailable')

    def test_follow_through_reports_reversal_and_truncated_coverage_without_a_score(self):
        rows = [row(.005, velocity=4), row(.010, force=20, velocity=3),
                row(.015, velocity=2), row(.020, velocity=1), row(.025, velocity=-1)]
        metrics = load_metrics()
        contact = metrics.contact_metrics(rows, [1, 0, 0])
        m = metrics.follow_through(rows, contact, [1, 0, 0], window_s=.020)
        self.assertAlmostEqual(m['coverage_fraction'], .5)
        self.assertAlmostEqual(m['continuous_forward_duration_s'], .005)
        self.assertEqual(m['first_nonforward_sample_s'], .025)
        self.assertAlmostEqual(m['signed_forward_displacement_estimate_m'], 0.)
        self.assertFalse(m['coverage_complete'])


class SegmentEvidenceTests(unittest.TestCase):
    def test_empty_legacy_poses_and_null_optional_segments_remain_unavailable(self):
        rows = [row(.005), row(.010), row(.015, force=50)]
        rows[0]['segment_quaternions_wxyz'] = None
        m = load_metrics().segment_sequencing({'kinematics': []}, rows, .015, .015)
        self.assertEqual(m['segments']['pelvis']['status'], 'unavailable')
        self.assertEqual(m['segments']['thorax']['status'], 'unavailable')

    def test_pelvis_heading_unwrap_and_missing_thorax_are_separate(self):
        def quat(deg):
            a = math.radians(deg) / 2
            return [math.cos(a), 0, 0, math.sin(a)]
        result = {'kinematics': [[{'time_s': t, 'root_pose': [0, 0, 0] + quat(a)}
                                 for t, a in [(.005, 179), (.010, -179), (.015, -178)]]]}
        rows = [row(.005), row(.010), row(.015), row(.020, force=50)]
        m = load_metrics().segment_sequencing(result, rows, .020, .020)
        self.assertAlmostEqual(m['segments']['pelvis']['net_projected_yaw_deg'], 3.)
        self.assertEqual(m['segments']['thorax']['status'], 'unavailable')
        self.assertEqual(m['pelvis_thorax_relative_heading']['status'], 'unavailable')

    def test_optional_world_segment_orientations_provide_relative_heading(self):
        rows = [row(.005), row(.010), row(.015, force=10)]
        for index, r in enumerate(rows[:2]):
            a = (index * .1) / 2
            r['segment_quaternions_wxyz'] = {
                'pelvis': [1, 0, 0, 0], 'thorax': [math.cos(a), 0, 0, math.sin(a)]}
        m = load_metrics().segment_sequencing({}, rows, .015, .015)
        self.assertEqual(m['segments']['thorax']['status'], 'available')
        self.assertAlmostEqual(m['pelvis_thorax_relative_heading']['net_change_deg'], math.degrees(.1))

    def test_original_recovery_criterion_survives_missing_segment_evidence(self):
        rows = [row(.005), row(.010), row(.015, force=50), row(.020), row(.025), row(.030)]
        result = dict(traces=[rows], kinematics=[], goals=[[2., 0.]], joint_names=['knee'],
                      records=[dict(contact_time_s=.015, success=True, failed=False, surface_match=True,
                                    goal=True, recovered=True)], root_fixed=False,
                      post_initialization_state_writes=False, stride_shot_success=True,
                      running_evidence={'running_success': True},
                      stride_shot_evidence={'continuous_running_contact': True},
                      sequence_records=[{'illegal_body_ball_contact': False,
                                         'recovery': {'passed': True, 'final_standing_seconds': .010}}])
        before = copy.deepcopy(result)
        m = load_metrics().summarize_explosive_strike(result, expected_duration_s=.030, window_s=.015)
        self.assertTrue(m['outcome']['success'])
        self.assertTrue(m['recovery']['original_recovery']['passed'])
        self.assertAlmostEqual(m['recovery']['final_stable_standing_start_s'], .020)
        self.assertAlmostEqual(m['recovery']['time_from_contact_to_final_stable_standing_s'], .005)
        self.assertEqual(result, before)

    def test_post_plant_joint_peaks_exclude_the_earlier_approach_peak(self):
        rows = [row(.005, velocity=100, support=0), row(.010, velocity=-6),
                row(.015, velocity=3), row(.020, force=50)]
        result = dict(traces=[rows], kinematics=[], goals=[[2., 0.]], joint_names=['knee'],
                      records=[dict(contact_time_s=.020, success=True, failed=False, surface_match=True,
                                    goal=True, recovered=True)], root_fixed=False,
                      post_initialization_state_writes=False, stride_shot_success=True,
                      running_evidence={'running_success': True},
                      stride_shot_evidence={'continuous_running_contact': True},
                      sequence_records=[{'illegal_body_ball_contact': False}])
        m = load_metrics().summarize_explosive_strike(result, expected_duration_s=.020, window_s=.020)
        self.assertIn('post_plant_joint_dynamics', m)
        self.assertEqual(m['precontact_joint_dynamics']['joints']['knee']['velocity']['peaks']['absolute']['value'], 100.)
        post = m['post_plant_joint_dynamics']
        self.assertEqual(post['joints']['knee']['velocity']['peaks']['absolute']['value'], -6.)
        self.assertAlmostEqual(post['measured_duration_s'], .010)


class AnalysisProvenanceTests(unittest.TestCase):
    def test_replay_match_ignores_added_observation_fields_but_detects_changed_motion(self):
        path = ROOT / 'tools' / 'analyze_explosive_strike.py'
        spec = importlib.util.spec_from_file_location('analyze_explosive_strike', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'compare_frozen_replay'), 'Instrumented replay matching is not implemented')
        original = {'records': [{'success': True}], 'kinematics': [[{'time_s': 0.}]],
                    'traces': [[row(.005), row(.010)]]}
        observed = copy.deepcopy(original)
        observed['traces'][0][0]['segment_quaternions_wxyz'] = {'pelvis': [1, 0, 0, 0]}
        m = module.compare_frozen_replay(observed, original)
        self.assertTrue(m['records_equal'])
        self.assertTrue(m['kinematics_equal'])
        self.assertTrue(m['legacy_trace_fields_equal'])
        observed['traces'][0][1]['joint_vel'][0] += 1.
        m = module.compare_frozen_replay(observed, original)
        self.assertFalse(m['legacy_trace_fields_equal'])
        self.assertEqual(m['first_trace_difference_s'], .010)

    def test_report_digest_mismatch_is_rejected_before_analysis(self):
        path = ROOT / 'tools' / 'analyze_explosive_strike.py'
        self.assertTrue(path.is_file(), 'Offline aggregation is not implemented')
        spec = importlib.util.spec_from_file_location('analyze_explosive_strike', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'episode.json'
            p.write_text(json.dumps({'traces': []}))
            with self.assertRaises(ValueError):
                module.read_verified_report(Path(temp), 'episode.json', '0' * 64)


class LandingWindowTests(unittest.TestCase):
    def landing(self, rows, start=.005, end=.015, **kwargs):
        metrics = load_metrics()
        self.assertTrue(hasattr(metrics, 'landing_window_dynamics'), 'Landing-window dynamics are not implemented')
        return metrics.landing_window_dynamics(rows, dynamics_metadata(), start, end, [1, 0, 0], **kwargs)

    def test_force_impulses_clip_both_window_boundaries_and_preserve_friction_sign(self):
        rows = [instrumented_row(t) for t in [.005, .010, .015]]
        m = self.landing(rows, .0075, .0125)
        force = m['ground_reaction']
        np.testing.assert_allclose(force['feet']['right']['components']['friction']['impulse_w_Ns'], [-.015, 0, 0])
        np.testing.assert_allclose(force['combined']['components']['total']['impulse_w_Ns'], [-.010, 0, .150])
        self.assertAlmostEqual(m['measured_duration_s'], .005)
        self.assertEqual(m['momentum_balance']['status'], 'unavailable')

    def test_start_boundary_peak_is_observed_but_its_preceding_bin_is_not_integrated(self):
        rows = [instrumented_row(t) for t in [.005, .010, .015]]
        rows[0]['support_foot_normal_force_w_N']['left'] = [0, 0, 1000]
        rows[0]['support_foot_ground_force_w_N']['left'] = [1, 0, 1000]
        m = self.landing(rows)
        self.assertIn('endpoint_inclusive_total_force_peak', m)
        self.assertEqual(m['endpoint_inclusive_total_force_peak']['time_s'], .005)
        self.assertEqual(m['endpoint_inclusive_total_force_peak']['force_w_N'][2], 1020.)
        self.assertAlmostEqual(m['ground_reaction']['combined']['components']['total']['impulse_w_Ns'][2], .3)

    def test_component_sum_mismatch_does_not_admit_certified_total(self):
        rows = [instrumented_row(t) for t in [.005, .010, .015]]
        rows[1]['support_foot_ground_force_w_N']['right'][0] = 100.
        m = self.landing(rows)
        self.assertEqual(m['ground_reaction']['feet']['right']['status'], 'unavailable')
        self.assertEqual(m['ground_reaction']['combined']['status'], 'unavailable')
        self.assertEqual(m['ground_reaction']['feet']['left']['status'], 'available')

    def test_capacity_saturation_overrules_global_coverage_claim(self):
        rows = [instrumented_row(t) for t in [.005, .010, .015]]
        rows[1]['support_contact_capacity_exhausted'] = True
        m = self.landing(rows)
        self.assertEqual(m['ground_reaction']['status'], 'unavailable')

    def test_foot_ground_plus_gravity_balance_matches_analytic_com_change(self):
        rows = [instrumented_row(t) for t in [.005, .010, .015]]
        for r, vx in zip(rows, [1., .995, .990]):
            r['whole_body_com_velocity'] = [vx, 0., 0.]
        m = self.landing(rows, gravity_w=[0, 0, -15])
        balance = m['momentum_balance']
        self.assertEqual(balance['status'], 'available')
        np.testing.assert_allclose(balance['unaccounted_impulse_w_Ns'], [0, 0, 0], atol=1e-12)
        self.assertAlmostEqual(balance['max_bin_residual_norm_Ns'], 0.)

    def test_unaccounted_impulse_is_localized_to_its_actual_bin(self):
        rows = [instrumented_row(t) for t in [.005, .010, .015]]
        for r, vx in zip(rows, [1., .995, 1.090]):
            r['whole_body_com_velocity'] = [vx, 0., 0.]
        m = self.landing(rows, gravity_w=[0, 0, -15])
        balance = m['momentum_balance']
        np.testing.assert_allclose(balance['unaccounted_impulse_w_Ns'], [.2, 0, 0], atol=1e-12)
        self.assertEqual(balance['max_residual_bin_s'], [.010, .015])
        self.assertAlmostEqual(balance['max_bin_residual_norm_Ns'], .2)

    def test_world_segment_rotation_and_relative_heading_use_observed_endpoints(self):
        rows = [instrumented_row(t) for t in [.005, .010, .015]]
        for index, r in enumerate(rows):
            r['segment_quaternions_wxyz'] = {}
            for name, angle in [('pelvis', index * .1), ('thorax', index * .2)]:
                r['segment_quaternions_wxyz'][name] = [math.cos(angle/2), 0, 0, math.sin(angle/2)]
        m = self.landing(rows)
        self.assertAlmostEqual(m['segments']['pelvis']['net_rotation_angle_deg'], math.degrees(.2))
        self.assertAlmostEqual(m['segments']['thorax']['net_projected_yaw_deg'], math.degrees(.4))
        self.assertAlmostEqual(m['pelvis_thorax_relative_heading']['net_change_deg'], math.degrees(.2))

    def test_speed_limit_diagnostics_keep_every_overspeed_joint_and_signed_speed(self):
        rows = [instrumented_row(t) for t in [.005, .010, .015]]
        for r, velocity in zip(rows, [[-3, 1], [-9, 5], [6, 1]]):
            r['joint_vel'] = velocity
        m = self.landing(rows, joint_names=['hip', 'knee'], joint_velocity_limits=[[6., 4.]])
        limits = m['joint_speed_limits']
        self.assertEqual(limits['all_overspeed_joints'], ['hip', 'knee'])
        self.assertAlmostEqual(limits['max_speed_to_limit_ratio'], 1.5)
        event = next(s for s in limits['samples'] if s['time_s'] == .010)
        self.assertEqual([j['signed_velocity_rad_s'] for j in event['overspeed_joints']], [-9., 5.])
        self.assertEqual(limits['samples'][-1]['overspeed_joints'], [])


class ScalingHypothesisTests(unittest.TestCase):
    def test_hypothesis_separates_internal_com_motion_from_floating_base_twist(self):
        metrics = load_metrics()
        self.assertTrue(hasattr(metrics, 'joint_velocity_scaling_hypothesis'), 'Scaling hypothesis is not reproducible yet')
        r = dict(time_s=.005, joint_vel=[-8, 2], root_xyz=[0, 0, 0], root_velocity_w=[1, 0, 0],
                 whole_body_com_xyz=[1, 0, -1], whole_body_com_velocity=[3, 6, 0],
                 segment_quaternions_wxyz={'pelvis': [1, 0, 0, 0]},
                 segment_angular_velocity_w_rad_s={'pelvis': [0, 0, 2]})
        m = metrics.joint_velocity_scaling_hypothesis(r, ['hip', 'knee'], [[4, 4]], 2., [0, 0, -1], [-2, -4, 0])
        self.assertEqual(m['gamma'], .5)
        np.testing.assert_allclose(m['internal_com_velocity_w_mps'], [2, 4, 0])
        np.testing.assert_allclose(m['predicted_impulse_w_Ns'], [-2, -4, 0])
        self.assertAlmostEqual(m['prediction_residual_error_norm_Ns'], 0.)
        self.assertFalse(m['local_solver_mechanism_verified'])

    def test_hypothesis_does_not_fill_missing_root_twist_with_zero(self):
        metrics = load_metrics()
        self.assertTrue(hasattr(metrics, 'joint_velocity_scaling_hypothesis'), 'Scaling hypothesis is not reproducible yet')
        m = metrics.joint_velocity_scaling_hypothesis(row(.005), ['knee'], [[1]], 2., [0, 0, -1], [0, 0, 0])
        self.assertEqual(m['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
