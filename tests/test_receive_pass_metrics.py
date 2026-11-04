from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from receive_pass_metrics import directional_travel,measured_step

class ReceivePassMetricsTests(unittest.TestCase):
    def test_crossing_uses_displacement_from_contact_and_commanded_direction(self):
        rows=[dict(time_s=2.,ball_xyz=[.3,.4,.11],ball_velocity=[0,-1,0]),
              dict(time_s=3.,ball_xyz=[.4,-.6,.11],ball_velocity=[.1,-1,0])]
        r=directional_travel(rows,[0,-1],2.)
        self.assertTrue(r['crossed']);self.assertAlmostEqual(r['heading_error_deg'],5.710593,places=5)
        self.assertFalse(directional_travel(rows,[0,1],2.)['crossed'])
    def test_moving_a_grounded_foot_is_not_a_step(self):
        rows=[dict(time_s=t,support_foot_force_N={'right':30.}) for t in [1.,1.1,1.2]]
        sensors=[dict(right_foot_position=[x,0,.035]) for x in [0,.05,.1]]
        self.assertFalse(measured_step(rows,sensors,1.,1.3)['success'])
    def test_step_needs_liftoff_translation_and_subsequent_load(self):
        rows=[dict(time_s=t,support_foot_force_N={'right':f}) for t,f in [(1.,30.),(1.1,0.),(1.2,30.),(1.3,30.)]]
        sensors=[dict(right_foot_position=p) for p in [[0,0,.035],[.05,0,.1],[.1,0,.035],[.1,0,.035]]]
        value=measured_step(rows,sensors,1.,1.4)
        self.assertTrue(value['success']);self.assertAlmostEqual(value['displacement_m'],.1)

if __name__=='__main__':unittest.main()
