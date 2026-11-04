"""Physical sequence evidence, including failures hidden by endpoint snapshots."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from runup_metrics import summarize_sequence, summarize_trial

JOINTS = ['right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint']


def sequence(hip=1., knee=2.):
    rows = []
    for i in range(640):
        t = (i + 1) * .005
        moving = t < 2.
        x = min(t * .4, .8)
        rows.append(dict(time_s=t, physics_dt_s=.005,
            root_xyz=[x, 0., .8], root_speed_xy=.6 if moving else .05,
            root_tilt_deg=5., feet_support_N=300., nonfoot_contact_N=0.,
            support_foot_force_N={'left': 150., 'right': 150.},
            whole_body_com_xyz=[x, 0., .7], whole_body_com_velocity=[.6 if moving else .05, 0., 0.],
            joint_pos=[0., .5, 0.], joint_vel=[1., 2., 0.],
            foot_velocity_contributions={'hip': [hip, 0., 0.], 'knee': [knee, 0., 0.],
                                         'ankle': [0., 0., 0.], 'base': [9., 0., 0.]},
            contacts=[{'foot': 'right', 'force_N': 100.}] if i == 399 else [],
            terminated=False))
    return rows


def summarize(rows, mode='knee', **kwargs):
    return summarize_sequence(rows, JOINTS, kwargs.pop('contact_time_s', 2.),
        forward=kwargs.pop('forward', [1., 0., 0.]), initial_root=[0., 0., .8],
        requested_mode=mode, **kwargs)


class RunupMetricsTests(unittest.TestCase):
    def result(self):
        rows = sequence()
        for row in rows:
            row['robot_ball_contact_force_N'] = {'right_ankle_roll_link':100.} if row['contacts'] else {}
        return dict(traces=[rows], trials=[{'mode':'knee','ball':[1.1,0.], 'goal':[3.7,0.]}],
            records=[{'success':True,'contact_time_s':2.}], points=[[1.1,0.]],goals=[[3.7,0.]],
            initial_robot_state={'root_position':[[0.,0.,.8]]}, joint_names=JOINTS,
            root_fixed=False,post_initialization_state_writes=False,contribution_foot='right')

    def test_full_gate_requires_physics_goal_and_legal_body_contacts(self):
        result = self.result()
        self.assertTrue(summarize_trial(result,0)['sequence_success'])
        result['traces'][0][370]['robot_ball_contact_force_N']={'right_knee_link':30.}
        measured=summarize_trial(result,0)
        self.assertFalse(measured['sequence_success'])
        self.assertTrue(measured['illegal_body_ball_contact'])
        self.assertEqual(measured['first_body_ball_contact']['body'],'right_knee_link')

    def test_running_diagnostic_cannot_certify_shooting(self):
        result=self.result();result['running_diagnostic']=True
        self.assertFalse(summarize_trial(result,0)['sequence_success'])

    def test_missing_all_body_evidence_or_wrong_kicking_foot_cannot_pass(self):
        for missing in ('robot_ball_contact_force_N',):
            result=self.result();result['traces'][0][0].pop(missing)
            self.assertFalse(summarize_trial(result,0)['sequence_success'])
        result=self.result();result['traces'][0][399]['robot_ball_contact_force_N']={'left_ankle_roll_link':100.}
        self.assertFalse(summarize_trial(result,0)['sequence_success'])
        result=self.result();result['post_initialization_state_writes']=True
        self.assertFalse(summarize_trial(result,0)['sequence_success'])

    def test_two_modes_follow_measured_directional_contributions_not_joint_amplitude(self):
        knee = summarize(sequence())
        hip = summarize(sequence(hip=2., knee=1.), mode='hip')
        self.assertTrue(knee['motion_sequence_pass'])
        self.assertTrue(hip['motion_sequence_pass'])
        self.assertAlmostEqual(knee['mode_evidence']['knee_fraction'], 2. / 3.)
        self.assertAlmostEqual(hip['mode_evidence']['hip_fraction'], 2. / 3.)
        self.assertAlmostEqual(knee['mode_evidence']['observed_duration_s'], .145)
        self.assertAlmostEqual(knee['mode_evidence']['knee_positive_displacement_m'], .290)
        self.assertFalse(summarize(sequence(), mode='hip')['mode_evidence']['passed'])

    def test_backward_motion_and_base_motion_do_not_count_as_knee_swing(self):
        result = summarize(sequence(hip=1., knee=-9.))
        self.assertEqual(result['mode_evidence']['knee_fraction'], 0.)
        self.assertFalse(result['mode_evidence']['passed'])
        self.assertTrue(summarize(sequence(hip=1., knee=-9.), mode='hip')['mode_evidence']['passed'])

    def test_zero_or_missing_motion_evidence_cannot_certify_a_mode(self):
        self.assertFalse(summarize(sequence(0., 0.))['mode_evidence']['passed'])
        for missing in ('foot_velocity_contributions', 'joint_vel'):
            rows = sequence()
            for row in rows[370:390]:
                row.pop(missing)
            self.assertFalse(summarize(rows)['mode_evidence']['coverage_complete'])
            self.assertFalse(summarize(rows)['motion_sequence_pass'])

    def test_truncated_window_and_sample_gaps_are_not_full_evidence(self):
        self.assertFalse(summarize(sequence()[380:])['mode_evidence']['coverage_complete'])
        rows = sequence()
        del rows[378:383]
        self.assertFalse(summarize(rows)['mode_evidence']['coverage_complete'])
        self.assertFalse(summarize(rows)['runup']['coverage_complete'])

    def test_impact_velocity_is_excluded_and_earlier_positive_contact_is_used(self):
        rows = sequence()
        rows[399]['foot_velocity_contributions']['hip'] = [10000., 0., 0.]
        self.assertAlmostEqual(summarize(rows)['mode_evidence']['knee_fraction'], 2. / 3.)
        rows[389]['contacts'] = [{'foot': 'right', 'force_N': .1}]
        result = summarize(rows)
        self.assertAlmostEqual(result['contact']['observed_contact_time_s'], 1.95)
        self.assertAlmostEqual(result['mode_evidence']['window_end_s'], 1.95)

    def test_in_place_or_sideways_steps_do_not_meet_forward_runup_distance(self):
        rows = sequence()
        for row in rows:
            row['root_xyz'][:2] = [0., row['root_xyz'][0]]
        result = summarize(rows)
        self.assertEqual(result['runup']['forward_displacement_m'], 0.)
        self.assertFalse(result['runup']['passed'])
        self.assertFalse(result['motion_sequence_pass'])

    def test_recovery_requires_final_continuous_ground_support_and_com_settling(self):
        self.assertGreaterEqual(summarize(sequence())['recovery']['final_standing_seconds'], 1.)
        for field, value in [
            ('support_foot_force_N', {'left': 0., 'right': 0.}),
            ('whole_body_com_velocity', [1., 0., 0.]),
            ('whole_body_com_xyz', None),
        ]:
            rows = sequence()
            rows[-1][field] = value
            self.assertFalse(summarize(rows)['recovery']['passed'])
        rows = sequence()
        for row in rows[-100:]:
            row['root_speed_xy'] = .6
        self.assertFalse(summarize(rows)['recovery']['passed'])

    def test_a_transient_nonfoot_contact_fall_or_termination_cannot_be_erased_by_recovery(self):
        for field, value in [('nonfoot_contact_N', 25.), ('root_xyz', [.5, 0., .4]),
                             ('root_tilt_deg', 65.), ('terminated', True)]:
            rows = sequence()
            rows[410][field] = value
            result = summarize(rows)
            self.assertFalse(result['recovery']['passed'])
            self.assertFalse(result['motion_sequence_pass'])

    def test_requested_contact_without_real_force_cannot_pass(self):
        rows = sequence()
        rows[399]['contacts'] = []
        result = summarize(rows)
        self.assertFalse(result['contact']['observed'])
        self.assertFalse(result['mode_evidence']['passed'])
        self.assertFalse(result['motion_sequence_pass'])

    def test_invalid_values_and_unknown_modes_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize(sequence(), mode='ankle')
        with self.assertRaises(ValueError):
            summarize(sequence(), forward=[0., 0., 0.])
        for field, value in [('time_s', float('nan')), ('root_speed_xy', float('nan'))]:
            rows = sequence()
            rows[2][field] = value
            with self.assertRaises(ValueError):
                summarize(rows)
        rows = sequence()
        rows[2]['time_s'] = rows[1]['time_s']
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_overlapping_bins_cannot_inflate_coverage_and_input_direction_is_unchanged(self):
        rows = sequence()
        rows[385]['time_s'] -= .002
        with self.assertRaises(ValueError):
            summarize(rows)
        direction = np.array([2., 0., 1.])
        summarize(sequence(), forward=direction)
        np.testing.assert_array_equal(direction, [2., 0., 1.])


if __name__ == '__main__':
    unittest.main()
