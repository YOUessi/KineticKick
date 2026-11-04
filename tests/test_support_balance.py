"""Feedback equilibrium, braking direction and finite foot support limits."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from support_balance import support_cop_correction

class SupportBalanceTests(unittest.TestCase):
    def setUp(self):
        self.foot=np.eye(4);self.foot[2,3]=.035
        self.jacobian=np.zeros((6,6));self.jacobian[3:,:3]=np.eye(3)
        self.cfg=dict(position_gain=12.,velocity_gain=6.,foot_x_min=-.045,foot_x_max=.123,foot_y_limit=.026)
    def evaluate(self,com,velocity):
        return support_cop_correction(np.array(com),np.array(velocity),self.foot,self.jacobian,33.,self.cfg)
    def test_stationary_balanced_com_needs_no_correction(self):
        torque,data=self.evaluate([.025,0.,.6],[0.,0.,0.])
        np.testing.assert_allclose(torque,0.,atol=1e-12)
    def test_forward_motion_requests_braking_pressure_and_torque(self):
        torque,data=self.evaluate([.04,0.,.6],[.1,0.,0.])
        self.assertGreater(data['desired_cop_xy'][0],.04)
        self.assertGreater(torque[1],0.)
        implied_acceleration=9.81*(.04-data['desired_cop_xy'][0])/.6
        self.assertLess(implied_acceleration,0.)
    def test_requested_cop_stays_inside_foot(self):
        _,data=self.evaluate([1.,-1.,.6],[2.,-2.,0.])
        np.testing.assert_allclose(data['local_cop_xy'],[.123,-.026])

if __name__=='__main__':unittest.main()
