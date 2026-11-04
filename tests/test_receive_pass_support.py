from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from receive_pass_support import passing_support_correction,required_right_load

class PassingSupportTests(unittest.TestCase):
    def test_repositioning_the_right_foot_does_not_drag_the_com_target(self):
        left=np.eye(4);left[2,3]=.035;right=left.copy();right[1,3]=-.15
        jac=np.zeros((6,6));jac[3:,:3]=np.eye(3)
        cfg=dict(position_gain=12.,velocity_gain=6.,foot_x_min=-.045,foot_x_max=.123,foot_y_limit=.026)
        for dx in [0.,.05]:
            right[0,3]=dx
            _,d=passing_support_correction(np.array([.025,0.,.65]),np.zeros(3),left,right,jac,33.,.9,cfg,np.array([.025,0.]))
            np.testing.assert_allclose(d['desired_cop_xy'],[.025,0.],atol=1e-12)

    def test_capture_motion_toward_right_foot_requests_its_support(self):
        left=np.eye(4);right=np.eye(4);right[1,3]=-.15
        cfg=dict(position_gain=12.,velocity_gain=6.)
        rest=required_right_load(np.array([.025,0.,.65]),np.zeros(3),left,right,np.array([.025,0.]),cfg)
        moving=required_right_load(np.array([.025,-.04,.65]),np.array([0.,-.06,0.]),left,right,np.array([.025,0.]),cfg)
        self.assertLess(rest,.01);self.assertGreater(moving,.2);self.assertLessEqual(moving,.45)

if __name__=='__main__':unittest.main()
