"""Keep ground, air, passive impact and illegal contacts distinct."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from inside_receive_metrics import summarize_inside_receive


class InsideReceiveAcceptance(unittest.TestCase):
    def scenario(self, mode):
        rows, sensors = [], []
        for i in range(120):
            air = mode == 'aerial' and i < 20
            rows.append(dict(time_s=(i+1)*.01, physics_dt_s=.01, terminated=False,
                robot_ball_contact_force_N={'right_ankle_roll_link':20.} if i in (10,11) else {},
                ball_xyz=[.2,0.,.3 if air else .11], ball_velocity=[1.,0.,-2. if air else 0.] if i<10 else [0.,0.,-.2 if air else 0.],
                root_xyz=[0.,0.,.76],root_tilt_deg=0.,root_speed_xy=0.,nonfoot_contact_N=0.,support_foot_force_N={'left':150.,'right':150.}))
            sensors.append(dict(ball_ground_force_N=0. if air else 4.,inside_point_velocity=[.2,0.,-.4 if air else 0.],ball_angular_velocity=[0.,0.,0.]))
        return dict(traces=[rows],inside_sensor_trace=sensors,records=[dict(requested_surface_impulse_fraction=1.)])

    def test_ground_success_is_not_aerial_success(self):
        result=self.scenario('ground')
        self.assertTrue(summarize_inside_receive(result,'ground')['success'])
        self.assertFalse(summarize_inside_receive(result,'aerial')['success'])

    def test_air_must_slow_before_floor_contact(self):
        result=self.scenario('aerial')
        self.assertTrue(summarize_inside_receive(result,'aerial')['success'])
        result['inside_sensor_trace'][10]['ball_ground_force_N']=4.
        value=summarize_inside_receive(result,'aerial')
        self.assertIsNone(value['outgoing_speed_mps'])
        self.assertFalse(value['success'])

    def test_static_barrier_is_not_active_cushioning(self):
        result=self.scenario('ground')
        result['inside_sensor_trace'][9]['inside_point_velocity']=[0.,0.,0.]
        self.assertFalse(summarize_inside_receive(result,'ground')['success'])

    def test_shin_contact_or_residual_spin_prevents_success(self):
        result=self.scenario('aerial')
        result['traces'][0][15]['robot_ball_contact_force_N']['right_knee_link']=3.
        self.assertFalse(summarize_inside_receive(result,'aerial')['success'])
        result=self.scenario('ground')
        for s in result['inside_sensor_trace'][20:]:s['ball_angular_velocity']=[5.,0.,0.]
        self.assertEqual(summarize_inside_receive(result,'ground')['ball_controlled_final_seconds'],0.)


if __name__=='__main__':unittest.main()
