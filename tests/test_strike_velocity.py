from pathlib import Path
import sys
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from strike_velocity import StrikeVelocityPulse


class StrikeVelocityPulseTests(unittest.TestCase):
    def setUp(self):
        self.pulse = StrikeVelocityPulse
        self.joints = ['hip', 'knee', 'ankle']

    def entry(self, **changes):
        return dict(window=[100, 110, 120, 130],
                    velocities={'hip': -4., 'knee': 8.}, **changes)

    def test_support_gate_requires_threshold_and_is_rechecked_each_call(self):
        p = self.pulse([self.entry()], self.joints)
        np.testing.assert_array_equal(p.apply([115], [79.9]), [[0, 0, 0]])
        np.testing.assert_array_equal(p.apply([115], [80]), [[-4, 8, 0]])
        np.testing.assert_array_equal(p.apply([115], [0]), [[0, 0, 0]])
        np.testing.assert_array_equal(p.apply([115], [81]), [[-4, 8, 0]])

    def test_cosine_window_has_zero_endpoints_and_signed_peak_plateau(self):
        p = self.pulse([self.entry()], self.joints)
        for frame, expected in [(90, [0, 0, 0]), (100, [0, 0, 0]),
                                (102.5, [-.5857864376, 1.1715728753, 0]),
                                (105, [-2, 4, 0]), (110, [-4, 8, 0]),
                                (115, [-4, 8, 0]), (120, [-4, 8, 0]),
                                (125, [-2, 4, 0]),
                                (127.5, [-.5857864376, 1.1715728753, 0]),
                                (130, [0, 0, 0]),
                                (140, [0, 0, 0])]:
            with self.subTest(frame=frame):
                np.testing.assert_allclose(p.apply([frame], [100]), [expected], atol=1e-7)

    def test_batch_entries_keep_trial_and_joint_order(self):
        p = self.pulse([self.entry(), None,
                        {'window': [200, 210, 220, 230], 'velocities': {'ankle': -12}}], self.joints)
        result = p.apply([115, 115, 215], [80, 100, 90])
        np.testing.assert_array_equal(result, [[-4, 8, 0], [0, 0, 0], [0, 0, -12]])
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.dtype, np.float32)

    def test_custom_support_threshold_is_used(self):
        p = self.pulse([self.entry(minimum_support_N=100)], self.joints)
        np.testing.assert_array_equal(p.apply([115], [99]), [[0, 0, 0]])
        np.testing.assert_array_equal(p.apply([115], [100]), [[-4, 8, 0]])

    def test_finite_large_frame_values_do_not_overflow_envelope(self):
        p = self.pulse([{'window': [0, 1e308, 1.2e308, 1.6e308],
                         'velocities': {'hip': -4., 'knee': 8.}}], self.joints)
        with np.errstate(over='raise', invalid='raise'):
            np.testing.assert_allclose(p.apply([7.5e307], [80]),
                                       [[-3.4142135624, 6.8284271247, 0]], rtol=1e-7)

    def test_inputs_and_previous_output_are_not_modified(self):
        entry = self.entry()
        p = self.pulse([entry], self.joints)
        frames, support = np.array([115.]), np.array([80.])
        first = p.apply(frames, support)
        entry['velocities']['knee'] = -11.
        entry['window'][1] = 125
        p.apply([95], [80])
        np.testing.assert_array_equal(first, [[-4, 8, 0]])
        np.testing.assert_array_equal(p.apply(frames, support), [[-4, 8, 0]])
        np.testing.assert_array_equal(frames, [115])
        np.testing.assert_array_equal(support, [80])

    def test_invalid_windows_are_rejected_before_use(self):
        for window in ([1, 2, 3], [1, 2, 2, 4], [1, 3, 2, 4],
                       [-1, 2, 3, 4], [1, 2, 3, float('nan')],
                       [1, 2, 3, float('inf')]):
            with self.subTest(window=window), self.assertRaises(ValueError):
                self.pulse([{'window': window, 'velocities': {'knee': 1}}], self.joints)

    def test_invalid_velocities_and_unknown_joints_are_rejected(self):
        for velocities in ({}, {'missing': 1}, {'knee': 12.01}, {'knee': -12.01},
                           {'knee': float('nan')}, {'knee': float('inf')}):
            with self.subTest(velocities=velocities), self.assertRaises(ValueError):
                self.pulse([{'window': [1, 2, 3, 4], 'velocities': velocities}], self.joints)

    def test_invalid_support_thresholds_are_rejected(self):
        for threshold in (0, -1, float('nan'), float('inf')):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.pulse([self.entry(minimum_support_N=threshold)], self.joints)

    def test_invalid_entry_and_joint_schemas_are_rejected(self):
        for entries, names in [([{}], self.joints), ([1], self.joints),
                               ([None], ['knee', 'knee']), ([None], []),
                               ([None], ['']), ([], self.joints)]:
            with self.subTest(entries=entries, names=names), self.assertRaises(ValueError):
                self.pulse(entries, names)

    def test_invalid_runtime_values_and_shapes_are_rejected(self):
        p = self.pulse([self.entry()], self.joints)
        for frames, support in [([float('nan')], [80]), ([115], [float('inf')]),
                                ([115], [float('nan')]), ([-1], [80]),
                                ([115, 116], [80]), ([115], [80, 90]),
                                ([[115]], [80])]:
            with self.subTest(frames=frames, support=support), self.assertRaises(ValueError):
                p.apply(frames, support)


if __name__ == '__main__':
    unittest.main()
