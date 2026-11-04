"""The receive trajectory must withdraw before impact and yield without springback."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from inside_receive_motion import InsideReceiveMotion


class InsideReceiveMotionTests(unittest.TestCase):
    def motion(self):
        foot=np.eye(4);foot[2,3]=.138
        plan=dict(receive_mode='ground',feet={'right':foot.tolist()},root_pose=[0,0,.7,1,0,0,0],
            contact_normal_local=[0,1,0],contact_center_local=[.1,.026,-.025],
            cushion=dict(velocity_fraction=.3,trajectory_lead_s=.16,freeze_prediction_lead_s=.08,follow_seconds=.25,absorption_limit_m=.04))
        return InsideReceiveMotion(plan)

    def test_retreat_is_monotonic_and_intercepts_moving_ball(self):
        motion=self.motion();ball=np.array([.1,.746,.113]);velocity=np.array([0,-1.2,0])
        motion.update_prediction(0.,ball,velocity)
        hit=motion.contact_time
        for t in np.linspace(hit-.16,hit,17):
            pose,speed,_,_=motion.targets(t)
            self.assertGreaterEqual(float(speed@velocity),-1e-8)
        pose,speed,_,_=motion.targets(hit)
        np.testing.assert_allclose(pose[:3,3]+motion.offset,ball+hit*velocity,atol=1e-9)
        np.testing.assert_allclose(speed,.3*velocity,atol=1e-9)

    def test_contact_yield_is_bounded_and_never_restores_into_ball(self):
        motion=self.motion();target=motion.initial_foot.copy();actual=target[:3,3]+np.array([0,-.02,0])
        shifted=motion.yield_to_contact(target,actual,10.)
        np.testing.assert_allclose(shifted[:3,3],actual)
        again=motion.yield_to_contact(shifted,actual+np.array([0,.01,0]),10.)
        np.testing.assert_allclose(again,shifted)
        motion.yield_to_contact(again,actual+np.array([0,-1.,0]),10.)
        self.assertAlmostEqual(motion.absorbed_displacement,.04)


if __name__=='__main__':unittest.main()
