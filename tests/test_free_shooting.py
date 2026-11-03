"""Behavioral checks for honest free-root shot acceptance."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from free_shooting_metrics import goal_crossing, contact_surface, ShotTracker, paid_contact_patch, contact_point_events


class GoalGeometryTests(unittest.TestCase):
    def test_entire_ball_must_cross_inside_the_aperture(self):
        kw = dict(goal=(0, -2), forward=(0, -1), width=1.6, height=1.1, radius=.11)
        self.assertFalse(goal_crossing((0, -1.9, .2), (0, -2.05, .2), **kw))
        self.assertTrue(goal_crossing((0, -1.9, .2), (0, -2.2, .2), **kw))
        self.assertFalse(goal_crossing((.72, -1.9, .2), (.72, -2.2, .2), **kw))
        self.assertFalse(goal_crossing((0, -1.9, 1.05), (0, -2.2, 1.05), **kw))

    def test_reverse_motion_and_a_ball_already_behind_goal_do_not_score(self):
        kw = dict(goal=(0, -2), forward=(0, -1), width=1.6, height=1.1, radius=.11)
        self.assertFalse(goal_crossing((0, -2.2, .2), (0, -1.9, .2), **kw))
        self.assertFalse(goal_crossing((0, -2.2, .2), (0, -2.3, .2), **kw))

    def test_crossing_uses_interpolated_position(self):
        self.assertTrue(goal_crossing((0, -1.9, .2), (1.0, -2.5, .2),
                                     goal=(0, -2), forward=(0, -1),
                                     width=1.6, height=1.1, radius=.11))

    def test_surface_is_derived_from_normal_not_requested_label(self):
        for normal, expected in [((0, 0, 1), "instep"), ((0, 1, 0), "inside"),
                                 ((0, -1, 0), "outside"), ((1, 0, 0), "toe"),
                                 ((0, 0, -1), "sole"), ((0, 0, 0), "ambiguous"),
                                 ((1, 1, 1), "ambiguous")]:
            self.assertEqual(contact_surface(normal, foot="right"), expected)
        self.assertEqual(contact_surface((0, -1, 0), foot="left"), "inside")

    def test_surface_requires_a_contact_point_on_the_audited_foot(self):
        self.assertEqual(paid_contact_patch((.075, .036, -.025), (0, 1, 0), foot="right"), "inside")
        self.assertEqual(paid_contact_patch((.075, -.036, -.025), (0, -1, 0), foot="right"), "outside")
        self.assertEqual(paid_contact_patch((.039, 0, -.015), (0, 0, 1), foot="right"), "instep")
        self.assertEqual(paid_contact_patch((.039, 0, .25), (0, 0, 1), foot="right"), "ambiguous")
        self.assertEqual(paid_contact_patch((.075, -.036, -.025), (0, 1, 0), foot="right"), "ambiguous")


def sample(time, root=(0, -.4, .8), ball=(0, -1, .11), contact=0,
           normal=(0, 0, 1), speed=0, terminated=False):
    return dict(time_s=time, root_xyz=root, root_speed_xy=speed,
                root_tilt_deg=0, ball_xyz=ball, contact_force_N=contact,
                contact_normal_foot=normal, terminated=terminated,
                feet_support_N=300, nonfoot_contact_N=0,
                contact_point_foot=(.039, 0, -.015))


class ShotAcceptanceTests(unittest.TestCase):
    def tracker(self):
        return ShotTracker(initial_root=(0, 0, .8), initial_ball=(0, -1, .11),
                           goal=(0, -3), forward=(0, -1), style="instep")

    def test_full_chain_requires_contact_goal_and_a_continuous_standing_window(self):
        t = self.tracker()
        t.update(sample(0, root=(0, 0, .8)))
        t.update(sample(.5, contact=10))
        t.update(sample(.6, ball=(0, -3.2, .11), speed=.5))
        for now in np.arange(.7, 1.61, .1):
            t.update(sample(float(now), ball=(0, -3.3, .11)))
        self.assertFalse(t.summary()["success"])
        t.update(sample(1.8, ball=(0, -3.4, .11)))
        self.assertTrue(t.summary()["success"])

    def test_wrong_surface_or_missing_approach_cannot_pass(self):
        for normal, root in [((1, 0, 0), (0, -.4, .8)), ((0, 0, 1), (0, 0, .8))]:
            t = self.tracker()
            t.update(sample(.5, contact=10, normal=normal, root=root))
            for now in np.arange(.6, 2, .1):
                t.update(sample(float(now), ball=(0, -3.3, .11), root=root))
            self.assertFalse(t.summary()["success"])

    def test_a_fall_is_permanent_even_after_autoreset(self):
        t = self.tracker()
        t.update(sample(.5, contact=10))
        t.update(sample(.6, terminated=True))
        for now in np.arange(.7, 2, .1):
            t.update(sample(float(now), ball=(0, -3.3, .11)))
        self.assertFalse(t.summary()["success"])
        self.assertTrue(t.summary()["failed"])

    def test_separate_short_standing_windows_do_not_add_up(self):
        t = self.tracker()
        t.update(sample(.5, contact=10))
        for now in np.arange(.6, 1.21, .1):
            t.update(sample(float(now), ball=(0, -3.3, .11)))
        t.update(sample(1.3, ball=(0, -3.3, .11), speed=.6))
        for now in np.arange(1.4, 2.01, .1):
            t.update(sample(float(now), ball=(0, -3.3, .11)))
        self.assertFalse(t.summary()["success"])

    def test_hands_or_knees_supporting_an_upright_root_is_not_standing(self):
        t = self.tracker()
        t.update(sample(.5, contact=10))
        for now in np.arange(.6, 2, .1):
            row = sample(float(now), ball=(0, -3.3, .11))
            row.update(feet_support_N=10, nonfoot_contact_N=250)
            t.update(row)
        self.assertFalse(t.summary()["success"])

    def test_contact_in_an_intermediate_physics_substep_is_retained(self):
        t = self.tracker()
        for now, force in [(.505, 0), (.510, 10), (.515, 0), (.520, 0)]:
            t.update(sample(now, contact=force))
        self.assertAlmostEqual(t.summary()["contact_time_s"], .510)
        self.assertEqual(t.summary()["measured_surface"], "instep")

    def test_a_small_correct_touch_cannot_hide_a_different_main_strike(self):
        t = self.tracker()
        t.update(sample(.5, contact=3))
        row = sample(.505, contact=200, normal=(1, 0, 0))
        row['contact_point_foot'] = (.132, 0, -.025)
        t.update(row)
        for now in np.arange(.6, 2., .1):
            t.update(sample(float(now), ball=(0, -3.3, .11)))
        self.assertFalse(t.summary()['success'])
        self.assertEqual(t.summary()['dominant_surface'], 'toe')

    def test_mixed_manifold_preserves_each_points_surface_and_impulse(self):
        t = self.tracker()
        events = contact_point_events([40., 30., 30.],
            [[.039, 0, -.015], [.132, 0, -.025], [.132, .005, -.025]],
            [[0, 0, 1], [1, 0, 0], [1, 0, 0]], foot='right', time_s=.5)
        row = sample(.5, contact=100.)
        row['contacts'] = events
        t.update(row)
        for now in np.arange(.6, 2., .1):
            t.update(sample(float(now), ball=(0, -3.3, .11)))
        self.assertFalse(t.summary()['success'])
        self.assertEqual(t.summary()['dominant_surface'], 'toe')
        self.assertAlmostEqual(t.summary()['requested_surface_impulse_fraction'], .4)

    def test_individually_small_points_still_contribute_to_total_impulse(self):
        t = self.tracker()
        events = contact_point_events([3., 1.9, 1.9],
            [[.039, 0, -.015], [.132, 0, -.025], [.132, .005, -.025]],
            [[0, 0, 1], [1, 0, 0], [1, 0, 0]], foot='right', time_s=.5)
        row = sample(.5, contact=6.8)
        row['contacts'] = events
        t.update(row)
        self.assertEqual(t.summary()['dominant_surface'], 'toe')
        self.assertAlmostEqual(t.summary()['requested_surface_impulse_fraction'], 3/6.8)


if __name__ == "__main__":
    unittest.main()
