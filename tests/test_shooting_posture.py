import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from shooting_posture import audit_posture, rotate_vector


def wxyz(rotation):
    q = rotation.as_quat().tolist()
    return q[3:] + q[:3]


def run_fixture(style='instep', pitch=65, hip_yaw=.3, body_heading=-90, rotation=0., point=None):
    yaw = -90 if style == 'instep' else 0
    turn = Rotation.from_euler('z', rotation, degrees=True)
    foot = turn * Rotation.from_euler('z', yaw, degrees=True) * Rotation.from_euler('y', pitch if style == 'instep' else 0, degrees=True)
    root_q = wxyz(turn * Rotation.from_euler('z', body_heading, degrees=True))
    xyz = lambda p: turn.apply(p).tolist()
    point = point or ([.075, 0, -.015] if style == 'instep' else [.075, -.036, -.025])
    normal = [0, 0, 1] if style == 'instep' else [0, -1, 0]
    event = dict(force_N=10., point_foot=point, normal_foot=normal, foot='right',
                 time_s=2., foot_quat_w=wxyz(foot), joint_pos=[hip_yaw])
    rows=[]
    for now, y, force in [(1., -1., 0), (2., -1., 10), (2.1, -1.2, 0), (3., -3.2, 0), (4., -3.4, 0)]:
        rows.append(dict(time_s=now, root_xyz=xyz([0, -.5 if now>=2 else 0, .8]), root_speed_xy=0.,
            root_tilt_deg=0., ball_xyz=xyz([0,y,.11]), contact_force_N=force,
            contact_normal_foot=normal if force else [0,0,0], contact_point_foot=point if force else None,
            foot='right', feet_support_N=300., nonfoot_contact_N=0., terminated=False,
            contacts=[event] if force else [], physics_dt_s=.005))
    return dict(requested_style=style, trials=[dict(style=style)], points=[xyz([0,-1,0])[:2]],
        goals=[xyz([0,-3,0])[:2]], initial_robot_state=dict(root_position=[xyz([0,0,.8])]),
        joint_names=['right_hip_yaw_joint'], traces=[rows], contacts=[[event]],
        kinematics=[[dict(time_s=2.,root_pose=xyz([0,-.5,.8])+root_q,joint_pos=[hip_yaw])]])


class PostureTests(unittest.TestCase):
    def test_wxyz_quaternion_convention(self):
        np.testing.assert_allclose(rotate_vector([1,0,0,0],[1,0,0]), [1,0,0])
        np.testing.assert_allclose(rotate_vector([math.sqrt(.5),0,0,math.sqrt(.5)],[1,0,0]), [0,1,0], atol=1e-12)
        with self.assertRaises(ValueError): rotate_vector([0,0,0,0],[1,0,0])

    def test_dorsal_surface_label_alone_does_not_certify_the_requested_pose(self):
        correct = audit_posture(run_fixture(), 0)
        flat = audit_posture(run_fixture(pitch=27, point=[.095,.023,-.023]), 0)
        self.assertTrue(correct['full_chain_success'])
        self.assertTrue(correct['posture_ok'])
        self.assertTrue(flat['full_chain_success'])
        self.assertFalse(flat['posture_ok'])
        self.assertFalse(flat['central_contact'])

    def test_outside_surface_cannot_hide_excessive_hip_twist_or_body_turn(self):
        good = audit_posture(run_fixture(style='outside', hip_yaw=1.5), 0)
        twisted = audit_posture(run_fixture(style='outside', hip_yaw=2.36, body_heading=-154), 0)
        self.assertTrue(good['posture_ok'])
        self.assertFalse(twisted['posture_ok'])
        self.assertGreater(twisted['pelvis_heading_error_deg'], 60)

    def test_metrics_are_invariant_to_world_heading(self):
        a, b = (audit_posture(run_fixture(rotation=v),0) for v in (0, 43))
        for key in ['top_forward', 'toe_world_z', 'pelvis_heading_error_deg']:
            self.assertAlmostEqual(a[key], b[key], places=6)
        self.assertEqual(a['posture_ok'], b['posture_ok'])

    def test_missing_or_wrong_kicking_foot_is_a_failed_trial_not_a_batch_error(self):
        wrong = run_fixture()
        wrong['traces'][0][1]['foot'] = 'left'
        wrong['traces'][0][1]['contacts'][0]['foot'] = 'left'
        missing = run_fixture()
        for row in missing['traces'][0]:
            row['contacts'] = []
            row['contact_force_N'] = 0.
        for run, reason in [(wrong, 'wrong_kicking_foot'), (missing, 'no_contact')]:
            result = audit_posture(run, 0)
            self.assertFalse(result['posture_and_full_chain'])
            self.assertEqual(result['posture_failure'], reason)

    def test_explicit_trial_style_needs_no_global_default(self):
        run = run_fixture()
        del run['requested_style']
        self.assertTrue(audit_posture(run, 0)['posture_ok'])


if __name__=='__main__': unittest.main()
