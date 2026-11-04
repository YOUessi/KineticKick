"""Regression for sphere-side contact points being mistaken for foot surface points."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from contact_frame_sampler import capsule_side_point, ContactFrameSampler
from free_shooting_metrics import contact_point_events, paid_contact_patch


class ContactFrameGeometry(unittest.TestCase):
    def test_penetrating_sphere_recovers_actual_capsule_surface(self):
        # Sphere above a capsule of radius .01 centered at z=-.025.
        # Its bottom penetrates the capsule top by .016 m.
        normal = np.array([[0., 0., 1.]])
        sphere_point = np.array([[.10, 0., -.031]])
        point = capsule_side_point(sphere_point, normal, np.array([-.016]))
        np.testing.assert_allclose(point, [[.10, 0., -.015]], atol=1e-12)
        self.assertEqual(paid_contact_patch(sphere_point[0], normal[0]), 'ambiguous')
        self.assertEqual(paid_contact_patch(point[0], normal[0]), 'instep')

    def test_separated_sphere_projects_in_opposite_direction(self):
        normal = np.array([[0., -1., 0.]])
        point = capsule_side_point(np.array([[.05, -.039, -.025]]), normal, np.array([.003]))
        np.testing.assert_allclose(point, [[.05, -.036, -.025]], atol=1e-12)
        self.assertEqual(paid_contact_patch(point[0], normal[0]), 'outside')

    def test_unmatched_contact_views_are_rejected(self):
        sampler = ContactFrameSampler.__new__(ContactFrameSampler)
        sampler.cached = {'right': {'forces': np.array([10.]), 'current_local': np.array([[0., 0., 0.]])}}
        with self.assertRaisesRegex(ValueError, 'Contact views disagree'):
            sampler.events(contact_point_events, [11.], [[0., 0., 0.]], [[0., 0., 1.]], foot='right', time_s=1.)


if __name__ == '__main__':
    unittest.main()
