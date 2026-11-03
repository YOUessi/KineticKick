from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from cartesian_strike import LegKinematics, CartesianStrike


class LegIKTests(unittest.TestCase):
    def test_unreachable_cartesian_target_does_not_replace_policy_actions(self):
        entry = dict(window=[100, 110, 120, 130], normal_local=[0, 1, 0],
                     elevation_rad=0., orientation_xyzw=[0, 0, 0, 1], point_local=[.075, .036, -.025],
                     foot_speed_mps=2., contact_frame=115, max_joint_correction_rad=.3)
        trial = dict(ball=[5, 5], goal=[0, -3.7], target=dict(target_x=0), cartesian=entry)
        leg = LegKinematics(ROOT / 'configs/paid_original/paid_main_resolved.urdf')
        controller = CartesianStrike([trial], leg.names, np.ones((1, 6)), np.zeros((1, 6)),
                                     leg.limits[None], ROOT / 'configs/paid_original/paid_main_resolved.urdf')
        action = np.array([[-.2, -.1, .2, .6, -.25, .04]])
        result = controller.apply(action, np.array([115]), np.array([[0, 0, .75, 1, 0, 0, 0]]), action, .02)
        np.testing.assert_array_equal(result, action)

    def test_pose_servo_reaches_a_displaced_rotated_foot_with_free_root_pose(self):
        leg = LegKinematics(ROOT / 'configs/paid_original/paid_main_resolved.urdf')
        root = np.eye(4)
        root[:3, :3] = Rotation.from_euler('xyz', [.03, -.02, -.8]).as_matrix()
        root[:3, 3] = [.1, -.5, .75]
        q = np.array([-.2, -.1, .2, .6, -.25, .04])
        target = leg.forward(q, root)[0]
        target[:3, 3] += [.01, -.015, .005]
        target[:3, :3] = Rotation.from_euler('z', .04).as_matrix() @ target[:3, :3]
        solution, errors = leg.solve(target, root, q)
        actual = leg.forward(solution, root)[0]
        self.assertLess(np.linalg.norm(actual[:3, 3] - target[:3, 3]), .001)
        self.assertLess(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()), .01)
        self.assertTrue(np.all(solution >= leg.limits[:, 0]))
        self.assertTrue(np.all(solution <= leg.limits[:, 1]))

    def test_unreachable_pose_reports_error_instead_of_silent_success(self):
        leg = LegKinematics(ROOT / 'configs/paid_original/paid_main_resolved.urdf')
        root = np.eye(4)
        target = np.eye(4)
        target[:3, 3] = [5, 5, 5]
        solution, errors = leg.solve(target, root, np.zeros(6))
        self.assertGreater(errors[0], 5.)
        self.assertTrue(np.all(np.isfinite(solution)))
        self.assertTrue(np.all(solution >= leg.limits[:, 0]))
        self.assertTrue(np.all(solution <= leg.limits[:, 1]))


if __name__ == '__main__':
    unittest.main()
