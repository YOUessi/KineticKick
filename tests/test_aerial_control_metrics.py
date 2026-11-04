"""Reject ground-assisted slowdowns and impacts mislabeled as active traps."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from aerial_control_metrics import summarize_aerial_control


class AerialControlAcceptance(unittest.TestCase):
    def scenario(self):
        rows, sensors = [], []
        for i in range(120):
            contact = i in (10, 11)
            rows.append(dict(time_s=(i+1)*.01, physics_dt_s=.01, terminated=False,
                robot_ball_contact_force_N={'right_ankle_roll_link': 20.} if contact else {},
                ball_xyz=[.2, 0, .3 if i < 20 else .11],
                ball_velocity=[0, 0, -4.] if i < 10 else [0, 0, -.2] if i < 20 else [0, 0, 0],
                right_foot_point_velocity=[0, 0, -.6], root_xyz=[0, 0, .76],
                root_tilt_deg=0., root_speed_xy=0., nonfoot_contact_N=0.,
                support_foot_force_N={'left': 150., 'right': 150.}))
            sensors.append(dict(ball_ground_force_N=4. if i >= 20 else 0.))
        return dict(traces=[rows], aerial_sensor_trace=sensors,
                    records=[dict(requested_surface_impulse_fraction=1.)])

    def test_complete_air_trap(self):
        self.assertTrue(summarize_aerial_control(self.scenario())['success'])

    def test_foot_touch_after_ground_is_not_aerial_control(self):
        r = self.scenario();r['aerial_sensor_trace'][5]['ball_ground_force_N'] = 4.
        self.assertFalse(summarize_aerial_control(r)['success'])

    def test_ground_slowdown_is_not_foot_cushioning(self):
        r = self.scenario();r['aerial_sensor_trace'][10]['ball_ground_force_N'] = 4.
        value = summarize_aerial_control(r)
        self.assertIsNone(value['outgoing_speed_mps'])
        self.assertFalse(value['cushioned'])

    def test_passive_impact_is_not_anticipatory_cushioning(self):
        r = self.scenario();r['traces'][0][9]['right_foot_point_velocity'][2] = .3
        value = summarize_aerial_control(r)
        self.assertTrue(value['air_contact_and_recovery'])
        self.assertTrue(value['speed_reduced_in_air'])
        self.assertFalse(value['cushioned'])
        self.assertFalse(value['success'])

    def test_ball_rolling_away_is_not_a_trap(self):
        r = self.scenario()
        for row in r['traces'][0][20:]:row['ball_xyz'][0] = 1.;row['ball_velocity'][0] = .4
        value = summarize_aerial_control(r)
        self.assertTrue(value['air_contact_and_recovery'])
        self.assertEqual(value['ball_controlled_final_seconds'], 0.)
        self.assertFalse(value['success'])


if __name__ == '__main__':
    unittest.main()
