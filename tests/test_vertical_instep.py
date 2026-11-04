import copy
import math
import unittest

from test_shooting_posture import run_fixture, wxyz, Rotation
from vertical_instep import audit_vertical_instep, check_vertical_acceptance, VERTICAL_CRITERIA
from test_posture_acceptance import matrix_fixture


def vertical_fixture(pitch=80, height=.22):
    run = run_fixture(pitch=pitch)
    quaternion = run['traces'][0][1]['contacts'][0]['foot_quat_w']
    sample = run['kinematics'][0][0]
    run['kinematics'][0] = []
    for t in [1.94, 1.96, 1.98, 2., 2.02, 2.04]:
        row = copy.deepcopy(sample)
        row.update(time_s=t, feet={'right': {'position': [0., -.9, height], 'quaternion': quaternion}})
        run['kinematics'][0].append(row)
    return run


def vertical_matrix_fixture():
    config, runs = matrix_fixture()
    config['vertical_instep_criteria'] = VERTICAL_CRITERIA.copy()
    vertical = vertical_fixture()
    for run in runs:
        run['kinematics'] = [copy.deepcopy(vertical['kinematics'][0]) for _ in range(3)]
        for trace in run['traces']:
            trace[399]['contacts'][0]['foot_quat_w'] = vertical['traces'][0][1]['contacts'][0]['foot_quat_w']
    return config, runs


class VerticalInstepTests(unittest.TestCase):
    def test_old_pose_is_not_near_vertical_but_toes_down_eighty_degrees_is(self):
        old = audit_vertical_instep(vertical_fixture(pitch=55), 0)
        new = audit_vertical_instep(vertical_fixture(), 0)
        self.assertFalse(old['vertical_and_full_chain'])
        self.assertTrue(new['vertical_and_full_chain'])
        self.assertAlmostEqual(new['toe_down_angle_deg'], 80.)
        self.assertAlmostEqual(new['near_contact_clearance_m'],
                               .22-.147*math.sin(math.radians(80))-.05*math.cos(math.radians(80)))

    def test_one_upright_frame_cannot_hide_mostly_flat_contact_impulse(self):
        run = vertical_fixture()
        row = run['traces'][0][2]
        event = copy.deepcopy(run['traces'][0][1]['contacts'][0])
        event.update(force_N=100., time_s=row['time_s'], foot_quat_w=wxyz(
            Rotation.from_euler('z', -90, degrees=True)*Rotation.from_euler('y', 30, degrees=True)))
        row.update(contacts=[event], contact_force_N=100., contact_normal_foot=[0,0,1], contact_point_foot=[.075,0,-.015])
        self.assertFalse(audit_vertical_instep(run, 0)['vertical_and_full_chain'])

    def test_ground_intersection_and_missing_pose_samples_fail(self):
        run = vertical_fixture(height=.13)
        self.assertFalse(audit_vertical_instep(run, 0)['vertical_and_full_chain'])
        run = vertical_fixture()
        del run['kinematics'][0][2]
        result = audit_vertical_instep(run, 0)
        self.assertFalse(result['vertical_and_full_chain'])
        self.assertEqual(result['vertical_failure'], 'missing_foot_pose_coverage')

    def test_horizontal_environment_translation_does_not_change_clearance(self):
        a, b = vertical_fixture(), vertical_fixture()
        for row in b['kinematics'][0]:
            row['root_pose'][0] += 20.
            row['root_pose'][1] -= 30.
            row['feet']['right']['position'][0] += 20.
            row['feet']['right']['position'][1] -= 30.
        self.assertAlmostEqual(audit_vertical_instep(a,0)['near_contact_clearance_m'],
                               audit_vertical_instep(b,0)['near_contact_clearance_m'])

    def test_strict_matrix_rejects_two_flat_trials_even_if_original_gate_passes(self):
        config, runs = vertical_matrix_fixture()
        self.assertTrue(check_vertical_acceptance(config, runs, 'frozen')['passed'])
        for index in [0, 1]:
            runs[0]['traces'][index][399]['contacts'][0]['foot_quat_w'] = vertical_fixture(pitch=55)['traces'][0][1]['contacts'][0]['foot_quat_w']
            result = check_vertical_acceptance(config, runs, 'frozen')
            self.assertEqual(result['passed'], index == 0)
        self.assertEqual(result['vertical_instep_successes'], 4)
        self.assertEqual(result['styles']['instep']['posture_and_full_chain'], 6)
        runs[0]['diagnostic_only'] = True
        with self.assertRaises(ValueError): check_vertical_acceptance(config, runs, 'frozen')
        runs[0].pop('diagnostic_only')
        config['vertical_instep_criteria']['max_angle_from_down_deg'] = 45
        with self.assertRaises(ValueError): check_vertical_acceptance(config, runs, 'frozen')

    def test_pose_timestamps_must_be_finite_in_trials_and_frozen_matrices(self):
        for invalid in [float('nan'), float('inf'), float('-inf')]:
            run = vertical_fixture()
            run['kinematics'][0][2]['time_s'] = invalid
            result = audit_vertical_instep(run, 0)
            self.assertFalse(result['vertical_and_full_chain'])
            self.assertEqual(result['vertical_failure'], 'missing_foot_pose_coverage')
        config, runs = vertical_matrix_fixture()
        for run in runs:
            for samples in run['kinematics']:
                samples[2]['time_s'] = float('nan')
        result = check_vertical_acceptance(config, runs, 'frozen')
        self.assertFalse(result['passed'])
        self.assertEqual(result['vertical_instep_successes'], 0)


if __name__ == '__main__': unittest.main()
