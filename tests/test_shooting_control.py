import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from shooting_control import StrikeResidual


class StrikeResidualTest(unittest.TestCase):
    def make(self, entries):
        return StrikeResidual(entries, ['hip', 'ankle'], np.array([[.5, .25]]),
                              np.array([[.1, -.2]]), np.array([[[-1., 1.], [-.6, .6]]]))

    def test_physical_radians_and_return_to_policy(self):
        control = self.make([dict(window=[80, 110, 150], offsets={'ankle': .3})])
        action = np.array([[.2, .4]], dtype=np.float32)
        np.testing.assert_array_equal(control.apply(action, [79]), action)
        np.testing.assert_allclose(control.apply(action, [110]), [[.2, 1.6]])
        np.testing.assert_array_equal(control.apply(action, [150]), action)
        np.testing.assert_array_equal(action, np.array([[.2, .4]], np.float32))

    def test_only_modified_targets_clipped_to_joint_limits(self):
        control = self.make([dict(window=[80, 110, 150], offsets={'ankle': 2.})])
        np.testing.assert_allclose(control.apply(np.array([[3., .4]]), [110]), [[3., 3.2]])

    def test_smooth_independent_phase_windows(self):
        control = self.make([dict(window=[80, 110, 150], offsets={'hip': .5}), {}])
        action = np.zeros((2, 2))
        np.testing.assert_allclose(control.apply(action, [95, 110]), [[.5, 0], [0, 0]])
        self.assertLess(abs(control.apply(action, [80.001, 0])[0, 0]), 1e-7)
        self.assertLess(abs(control.apply(action, [149.999, 0])[0, 0]), 1e-7)

    def test_reject_unknown_joint_and_invalid_windows(self):
        with self.assertRaises(ValueError):
            self.make([dict(window=[80, 110, 150], offsets={'typo': .3})])
        with self.assertRaises(ValueError):
            self.make([dict(window=[110, 80, 150], offsets={'hip': .3})])

    def test_reference_position_and_velocity_are_consistent(self):
        control = self.make([dict(window=[80, 110, 150], offsets={'hip': .5})])
        obs = np.zeros((1, 160), np.float32)
        updated = control.apply_reference(obs, [95], .02)
        self.assertAlmostEqual(updated[0, 0], .25)
        self.assertAlmostEqual(updated[0, 2], .5 * np.pi / (2 * 30 * .02), places=6)
        np.testing.assert_array_equal(updated[:, 4:], obs[:, 4:])
        np.testing.assert_array_equal(control.apply_reference(obs, [150], .02), obs)

    def test_absolute_targets_hold_then_return_without_accumulation(self):
        control = self.make([dict(window=[80, 100, 120, 150], targets={'hip': .6})])
        action = np.array([[-1., .4]], dtype=np.float32)
        np.testing.assert_allclose(control.apply(action, [110]), [[1., .4]])
        np.testing.assert_allclose(control.apply(action, [90]), [[0., .4]], atol=1e-7)
        np.testing.assert_array_equal(control.apply(action, [150]), action)

    def test_continuation_is_independent_and_returns_to_policy(self):
        entry=dict(window=[80,100,120,130], targets={'hip':.6},
                   continuation=dict(window=[124,134,144,160], targets={'ankle':.3}))
        phased=self.make([entry, {}])
        base=self.make([{k:v for k,v in entry.items() if k!='continuation'}, {}])
        action=np.array([[-1.,.4],[.2,.3]],np.float32)
        for frame in [79,90,110,124,160,180]:
            np.testing.assert_array_equal(phased.apply(action,[frame,frame]),base.apply(action,[frame,frame]))
        np.testing.assert_allclose(phased.apply(action,[134,134]),[[-1.,2.],[.2,.3]])
        # Smooth boundary, normal joint limits and independently timed release.
        self.assertLess(abs(phased.apply(action,[124.001,0])[0,1]-base.apply(action,[124.001,0])[0,1]),1e-6)
        clipped=self.make([dict(continuation=dict(window=[120,130,140],targets={'ankle':9.}))])
        np.testing.assert_allclose(clipped.apply(action[:1],[130]),[[-1.,3.2]])

    def test_continuation_rejects_nested_entries_and_reference_application(self):
        with self.assertRaises(ValueError):self.make([dict(continuation=dict(continuation={}))])
        control=self.make([dict(continuation=dict(offsets={'hip':.2}))])
        with self.assertRaises(ValueError):control.apply_reference(np.zeros((1,160)),[110],.02)


if __name__ == '__main__':
    unittest.main()
