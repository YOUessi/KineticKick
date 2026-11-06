"""Adversarial checks for the offline P0 evidence audit; no simulator required."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_p0_dynamic_strike import (
    CapsuleGeometry, contact_geometry, limit_observations, measure_path, performance_delta, segment_distance,
    verify_invariants,
)


class P0DynamicAuditTests(unittest.TestCase):
    def test_body_translation_cannot_be_counted_as_relative_follow_through(self):
        times = np.arange(.94, 1.801, .02)
        foot = np.c_[2 * (times - 1), np.zeros((len(times), 2))]
        motion = measure_path(times, foot, foot.copy(), 1., [1., 0., 0.])
        self.assertAlmostEqual(motion['forward_advance_m'], .5)
        self.assertAlmostEqual(motion['pelvis_relative_advance_m'], 0.)
        self.assertIn('short_relative_swing', motion['diagnostic_failures'])

    def test_missing_and_nonfinite_samples_cannot_certify_continuity(self):
        times = np.arange(.94, 1.801, .02)
        foot = np.c_[times, np.zeros((len(times), 2))]
        for tt, ff in [(np.delete(times, 8), np.delete(foot, 8, axis=0)),
                       (times[:-5], foot[:-5]), (times[::-1], foot[::-1])]:
            with self.assertRaises(ValueError):
                measure_path(tt, ff, ff * 0, 1., [1., 0., 0.])
        foot[5, 0] = np.nan
        with self.assertRaises(ValueError):
            measure_path(times, foot, foot * 0, 1., [1., 0., 0.])

    def test_direction_and_world_translation_preserve_the_physical_measurement(self):
        times = np.arange(.94, 1.801, .02)
        foot = np.c_[times, np.zeros((len(times), 2))]
        roots = foot * .2
        a = measure_path(times, foot, roots, 1., [1., 0., 0.])
        b = measure_path(times, foot + [100, -50, 6], roots + [100, -50, 6],
                         1., [1., 0., 0.])
        self.assertAlmostEqual(a['pelvis_relative_advance_m'], .2)
        self.assertAlmostEqual(a['pelvis_relative_advance_m'], b['pelvis_relative_advance_m'])
        reverse = measure_path(times, -foot, roots * 0, 1., [1., 0., 0.])
        self.assertEqual(reverse['reversal_after_s'], 0.)
        self.assertIn('early_reversal', reverse['diagnostic_failures'])

    def test_segment_distance_handles_crossing_parallel_and_degenerate_segments(self):
        self.assertAlmostEqual(float(segment_distance([0, 0, 0], [1, 0, 0],
                                                     [.5, -1, 0], [.5, 1, 0])), 0.)
        self.assertAlmostEqual(float(segment_distance([0, 0, 0], [1, 0, 0],
                                                     [0, 2, 0], [1, 2, 0])), 2.)
        self.assertAlmostEqual(float(segment_distance([0, 0, 0], [0, 0, 0],
                                                     [1, 0, 0], [1, 0, 0])), 1.)

    def test_urdf_clearance_uses_capsule_surface_radii_and_joint_origins(self):
        links = ['<link name="pelvis"><collision><geometry><sphere radius=".05"/></geometry></collision></link>']
        names = []
        for side, y in [('left', .2), ('right', -.2)]:
            for part, axis, origin in [('hip_pitch', '0 1 0', f'0 {y} 0'),
                                       ('hip_roll', '1 0 0', '0 0 0'),
                                       ('hip_yaw', '0 0 1', '0 0 0'),
                                       ('knee', '0 1 0', '0 0 -.4')]:
                name = f'{side}_{part}_joint'
                names.append(name)
                links.append(f'<joint name="{name}"><origin xyz="{origin}"/><axis xyz="{axis}"/></joint>')
            for link, z in [('hip_roll', -.15), ('knee', -.2)]:
                links.append(f'<link name="{side}_{link}_link"><collision><origin xyz="0 0 {z}"/>'
                             '<geometry><cylinder radius=".02" length=".2"/></geometry></collision></link>')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.urdf'
            path.write_text('<robot>' + ''.join(links) + '</robot>')
            gaps = CapsuleGeometry(path, names).gaps(np.zeros((1, len(names))))
        self.assertAlmostEqual(float(gaps['right_pelvis_thigh'][0]), np.hypot(.2, .05) - .07)
        self.assertAlmostEqual(float(gaps['shank_to_shank'][0]), .36)

    def test_clamped_command_does_not_hide_actual_velocity_overshoot(self):
        run = dict(joint_names=['knee', 'ankle'], joint_effort_limits=[[10., 5.]],
                   joint_velocity_limits=[[20., 10.]])
        rows = [dict(time_s=.005, joint_vel=[21., -9.], joint_torque_estimate=[10., 0.],
                     joint_velocity_target=[20., -10.])]
        result = limit_observations(rows, run)
        self.assertAlmostEqual(result['maximum_actual_speed_ratio'], 1.05)
        self.assertEqual(result['maximum_velocity_target_ratio'], 1.)
        self.assertEqual(result['actual_speed_exceedance_sample_joint_count'], 1)
        self.assertEqual(result['worst_actual_speed']['joint'], 'knee')

    def test_raw_contact_frame_reconstruction_detects_corrupted_geometry(self):
        event = dict(foot='right', force_N=10., point_foot=[.06, 0., -.02],
                     normal_foot=[0., 0., 1.], raw_sphere_point_world=[.06, 0., -.021],
                     raw_outward_normal_world=[0., 0., 1.], raw_separation_m=-.001,
                     contact_foot_position_w=[0., 0., 0.],
                     contact_foot_quat_w=[1., 0., 0., 0.],
                     foot_pose_time_s=.995, time_s=1., point_frame='pre_step_capsule_surface',
                     capsule_surface_residual_m=0.)
        rows = [dict(time_s=1., physics_dt_s=.005, contacts=[event])]
        good = contact_geometry(rows)
        self.assertTrue(good['frame_reconstruction_passed'])
        self.assertAlmostEqual(good['instep_impulse_fraction'], 1.)
        event['point_foot'][0] += .01
        self.assertFalse(contact_geometry(rows)['frame_reconstruction_passed'])
        del event['contact_foot_quat_w']
        self.assertFalse(contact_geometry(rows)['frame_reconstruction_passed'])

    def test_missing_invariant_metadata_cannot_pass_by_matching_missing_metadata(self):
        result = verify_invariants({}, {})
        self.assertFalse(result['recorded_invariants_match'])
        self.assertIn('root_fixed', result['failures'])

    def test_speed_gain_never_converts_goal_miss_to_success(self):
        old = dict(outcome=dict(success=True), final_standing_seconds=8.,
                   contact=dict(outgoing_speed_xy_mps=4., outgoing_heading_error_deg=2.))
        new = copy.deepcopy(old)
        new['outcome']['success'] = False
        new['contact']['outgoing_speed_xy_mps'] = 5.
        delta = performance_delta(old, new)
        self.assertTrue(delta['original_task_regression'])
        self.assertEqual(delta['outgoing_speed_xy_delta_mps'], 1.)
        self.assertIsNone(performance_delta(None, new))


if __name__ == '__main__':
    unittest.main()
