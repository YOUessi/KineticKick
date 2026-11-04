from pathlib import Path
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from runup_sequence import resolve_motion_files, phase_label, CoordinationTargets, approach_offset
import numpy as np


class ReferenceSelectionTests(unittest.TestCase):
    def test_default_remains_one_reference_for_legacy_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'default.npz'
            p.touch()
            self.assertEqual(resolve_motion_files([{}, {}], tmp, p), [str(p)])

    def test_per_environment_order_and_duplicates_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('a.npz', 'b.npz'):
                (Path(tmp) / name).touch()
            result = resolve_motion_files([{'motion_file': x} for x in ('b.npz', 'a.npz', 'b.npz')], tmp, 'a.npz')
            self.assertEqual(result, [str(Path(tmp) / x) for x in ('b.npz', 'a.npz', 'b.npz')])

    def test_partial_reference_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'every trial'):
            resolve_motion_files([{'motion_file': 'a.npz'}, {}], '/tmp', 'a.npz')

    def test_missing_reference_fails_before_simulator(self):
        with self.assertRaisesRegex(ValueError, 'reference'):
            resolve_motion_files([{'motion_file': '/tmp/absent-runup-motion.npz'}], '/tmp', 'a.npz')

    def test_contact_is_required_before_recovery_phase(self):
        self.assertEqual(phase_label(5, None, False, False), 'approach')
        self.assertEqual(phase_label(2.42, 2.4, False, False), 'contact_follow_through')
        self.assertEqual(phase_label(4, 2.4, False, False), 'recover')
        self.assertEqual(phase_label(5, 2.4, True, False), 'stable')
        self.assertEqual(phase_label(5, 2.4, True, True), 'failed')


class CoordinationTests(unittest.TestCase):
    def make(self, entry=None):
        entry = entry or dict(start_frame=100, times_s=[.2, .4, .7],
            prepare={'knee': 1.2}, strike={'knee': .2}, minimum_support_N=80,
            maximum_ball_distance_m=.8)
        return CoordinationTargets([entry], ['hip', 'knee'], np.ones((1, 2)),
                                   np.zeros((1, 2)), np.array([[[-2, 2], [0, 1.5]]]))

    def step(self, controller, time, support=100, distance=.5, frame=110):
        return controller.apply(np.array([[.3, .5]]), time, [frame], [support], [distance])

    def test_support_and_distance_are_required_for_activation(self):
        c = self.make()
        for support, distance in ((0, .5), (100, 1.)):
            np.testing.assert_array_equal(self.step(c, 2, support, distance), np.array([[.3, .5]], np.float32))
        self.assertEqual(c.started, [None])
        self.step(c, 2)
        self.assertEqual(c.started, [2])

    def test_preparation_extension_and_release_are_continuous(self):
        c = self.make()
        np.testing.assert_allclose(self.step(c, 2), [[.3, .5]])
        np.testing.assert_allclose(self.step(c, 2.2), [[.3, 1.2]])
        np.testing.assert_allclose(self.step(c, 2.4), [[.3, .2]])
        np.testing.assert_allclose(self.step(c, 2.7), [[.3, .5]])
        self.assertEqual(c.phases, ['released'])

    def test_gate_loss_does_not_restart_leg_mid_swing(self):
        c = self.make()
        self.step(c, 2)
        np.testing.assert_allclose(self.step(c, 2.2, support=0), [[.3, 1.2]])
        self.assertEqual(c.started, [2])

    def test_unknown_joint_and_unsafe_timing_fail_before_control(self):
        for entry in (dict(start_frame=100, times_s=[.2, .1, .7], prepare={'knee': 1}, strike={'knee': 0}),
                      dict(start_frame=100, times_s=[.2, .4, .7], prepare={'missing': 1}, strike={'missing': 0})):
            with self.assertRaises(ValueError):
                self.make(entry)

    def test_velocity_feedforward_tracks_cubic_swing_then_releases(self):
        c = self.make(dict(start_frame=100, times_s=[.2,.4,.7], prepare={'knee':1.2},
                           strike={'knee':.2}, velocity_feedforward=True))
        self.step(c,2)
        self.step(c,2.1)
        np.testing.assert_allclose(c.velocity_targets, [[0,5.25]])
        self.step(c,2.3)
        np.testing.assert_allclose(c.velocity_targets, [[0,-7.5]])
        self.step(c,2.7)
        np.testing.assert_allclose(c.velocity_targets, [[0,0]])

    def test_approach_waypoint_fades_without_moving_physical_ball(self):
        entry = dict(approach_offset_xy=[.16,-.04], approach_offset_fade_s=[2.5,3.5])
        self.assertEqual(approach_offset(entry,2), [.16,-.04,0])
        np.testing.assert_allclose(approach_offset(entry,3), [.08,-.02,0])
        self.assertEqual(approach_offset(entry,4), [0,0,0])
        with self.assertRaises(ValueError):
            approach_offset(dict(approach_offset_xy=[float('nan'),0]), 0)


if __name__ == '__main__':
    unittest.main()
