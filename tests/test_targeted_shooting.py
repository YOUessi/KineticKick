"""Aim and strength must be measured independently of the controller inputs."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from targeted_shooting import TargetShotTracker


def sample(t, position, velocity=(0, -2, 0), force=0):
    return dict(time_s=t, ball_xyz=position, ball_velocity=velocity,
                contact_force_N=force, terminated=False)


class TargetMetricTests(unittest.TestCase):
    def test_interpolates_fixed_plane_crossing_and_requested_point_error(self):
        tracker = TargetShotTracker(target_x=.3, speed_mps=2.)
        tracker.update(sample(1., (.1, -.93, .11), force=5))
        tracker.update(sample(1.1, (.1, -1.13, .11)))
        tracker.update(sample(2., (.2, -3.5, .11)))
        tracker.update(sample(2.2, (.4, -3.9, .11)))
        result = tracker.summary()
        self.assertAlmostEqual(result['crossing']['time_s'], 2.1)
        self.assertAlmostEqual(result['crossing']['position'][0], .3)
        self.assertAlmostEqual(result['crossing']['position'][1], -3.7)
        self.assertAlmostEqual(result['aim_error_m'], 0)
        self.assertTrue(result['target_hit'])
        self.assertTrue(result['speed_match'])

    def test_first_crossing_outside_goal_is_recorded_and_not_replaced(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=2.)
        tracker.update(sample(0., (1., -.93, .11), force=5))
        tracker.update(sample(.1, (1., -1.13, .11)))
        tracker.update(sample(1., (1., -3.6, .11)))
        tracker.update(sample(1.2, (1., -3.8, .11)))
        tracker.update(sample(1.4, (0., -3.6, .11)))
        tracker.update(sample(1.6, (0., -3.8, .11)))
        result = tracker.summary()
        self.assertAlmostEqual(result['aim_error_m'], 1.)
        self.assertFalse(result['target_hit'])

    def test_reverse_crossing_does_not_supply_target_evidence(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=2.)
        tracker.update(sample(1., (0., -3.8, .11), force=5))
        tracker.update(sample(1.2, (0., -3.6, .11), velocity=(0, 2, 0)))
        self.assertIsNone(tracker.summary()['crossing'])
        self.assertFalse(tracker.summary()['target_success'])

    def test_speed_is_interpolated_at_contact_plus_point_one_not_peak(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=3.)
        tracker.update(sample(1., (0, -.93, .11), velocity=(0, -8, 0), force=100))
        tracker.update(sample(1.08, (0, -1.4, .11), velocity=(0, -4, 0)))
        tracker.update(sample(1.12, (0, -1.5, .11), velocity=(0, -2, 0)))
        self.assertAlmostEqual(tracker.summary()['outgoing_speed_mps'], 3.)
        self.assertTrue(tracker.summary()['speed_match'])

    def test_short_trace_cannot_supply_speed(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=2.)
        tracker.update(sample(1., (0, -.93, .11), force=100))
        tracker.update(sample(1.09, (0, -1.1, .11)))
        self.assertIsNone(tracker.summary()['outgoing_speed_mps'])
        self.assertFalse(tracker.summary()['target_success'])

    def test_missing_contact_nonfinite_input_and_termination_cannot_pass(self):
        for bad in ('contact', 'nan', 'terminated'):
            tracker = TargetShotTracker(target_x=0, speed_mps=2.)
            tracker.update(sample(1., (0, -.93, .11), force=0 if bad == 'contact' else 5))
            row = sample(1.1, (0, -1.13, .11))
            if bad == 'nan':
                row['ball_velocity'] = (float('nan'), -2, 0)
            if bad == 'terminated':
                row['terminated'] = True
            tracker.update(row)
            tracker.update(sample(2., (0, -3.6, .11)))
            tracker.update(sample(2.2, (0, -3.8, .11)))
            self.assertFalse(tracker.summary()['target_success'], bad)

    def test_duplicate_or_reversed_timestamps_are_rejected(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=2.)
        tracker.update(sample(1., (0, -.93, .11)))
        with self.assertRaises(ValueError):
            tracker.update(sample(1., (0, -.93, .11)))


if __name__ == '__main__':
    unittest.main()
