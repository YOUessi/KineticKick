from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from support_balance import support_cop_correction
from receiving_stance import grounded_support_correction

class GroundedSupportTests(unittest.TestCase):
    def setUp(self):
        self.left=np.eye(4);self.left[2,3]=.035
        self.right=self.left.copy();self.right[1,3]=-.18
        self.j=np.zeros((6,6));self.j[3:,:3]=np.eye(3)
        self.cfg=dict(position_gain=12.,velocity_gain=6.,foot_x_min=-.045,foot_x_max=.123,foot_y_limit=.026)
    def test_full_left_support_preserves_verified_correction(self):
        com=np.array([.04,.01,.65]);vel=np.array([.1,0.,0.])
        expected,_=support_cop_correction(com,vel,self.left,self.j,33.,self.cfg)
        actual,_=grounded_support_correction(com,vel,self.left,self.right,self.j,33.,1.,self.cfg)
        np.testing.assert_allclose(actual,expected)
    def test_two_foot_static_equilibrium_needs_no_extra_moment(self):
        torque,data=grounded_support_correction(np.array([.025,-.018,.65]),np.zeros(3),self.left,self.right,self.j,33.,.9,self.cfg)
        np.testing.assert_allclose(torque,0.,atol=1e-12)
    def test_requested_pressure_stays_on_the_supporting_foot(self):
        _,data=grounded_support_correction(np.array([1.,1.,.65]),np.ones(3),self.left,self.right,self.j,33.,.9,self.cfg)
        np.testing.assert_allclose(data['local_cop_xy'],[.123,.026])

if __name__=='__main__':unittest.main()
