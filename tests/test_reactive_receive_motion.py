from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from reactive_receive_motion import ReactiveReceiveMotion

class ReactivePredictionTests(unittest.TestCase):
    def planner(self):
        f=np.eye(4);f[2,3]=.035
        return ReactiveReceiveMotion(dict(feet={'right':f.tolist()},root_pose=[0,0,.74,1,0,0,0],
            contact_center_local=[.075,.026,-.025],cushion={'absorption_limit_m':.12},
            reaction=dict(anchor=[-.35,-.40],lead_s=.55,freeze_s=.08,ground_fraction=.3,aerial_fraction=.13,follow_s=.25)))
    def observe(self,p,z,vz=0.,speed=3.):
        for t in [0.,.02,.04]:p.update_prediction(t,[-.35,-1.2+speed*t,z+vz*t-4.905*t*t],[0,speed,vz-9.81*t] if z>.15 else [0,speed,0])
    def test_height_changes_interception_without_changing_ready_pose(self):
        low,high=self.planner(),self.planner()
        self.observe(low,.11);self.observe(high,.9,-.3)
        self.assertIsNotNone(low.contact_time);self.assertIsNotNone(high.contact_time)
        self.assertGreater(high.hit_position[2],low.hit_position[2]+.05)
        np.testing.assert_allclose(low.initial_foot,high.initial_foot)
        self.assertLess(low.initial_foot[2,3],.06)
    def test_speed_changes_contact_time(self):
        slow,fast=self.planner(),self.planner();self.observe(slow,.11,speed=1.8);self.observe(fast,.11,speed=3.)
        self.assertLess(fast.contact_time,slow.contact_time)
    def test_measured_ground_contact_starts_braking_without_pose_jump(self):
        p=self.planner();p.reaction['ground_brake']=True
        self.observe(p,.11)
        p.actual_foot=p.targets(.5)[0]
        p.update_prediction(.5,[-.35,-.25,.11],[0,.3,0],True)
        p.update_prediction(.64,[-.35,-.21,.11],[0,.2,0],True)
        pose,speed,_,phase=p.targets(.64)
        self.assertEqual(phase,'brake_ground')
        np.testing.assert_allclose(pose,p.actual_foot)
        self.assertTrue(np.isfinite(speed).all())

    def test_aerial_withdrawal_uses_observed_speed_and_is_bounded(self):
        fractions=[]
        for speed in [2.5,3.,3.5]:
            p=self.planner()
            p.reaction.update(aerial_reference_speed_mps=3.,aerial_speed_gain=-.11)
            self.observe(p,.8,-.3,speed=speed)
            self.assertTrue(p.predictions)
            fractions.append(p.predictions[-1]['velocity_fraction'])
        self.assertGreater(fractions[0],fractions[1])
        self.assertGreater(fractions[1],fractions[2])
        self.assertTrue(all(.04<=f<=.20 for f in fractions))
        p=self.planner();p.reaction.update(aerial_reference_speed_mps=3.,aerial_speed_gain=-.11)
        self.observe(p,.11,speed=2.5)
        self.assertEqual(p.predictions[-1]['velocity_fraction'],p.reaction['ground_fraction'])

    def test_medial_face_trim_responds_to_observed_incoming_direction(self):
        normals=[]
        for vx in [.4,-.4]:
            p=self.planner()
            p.reaction.update(aerial_yaw_trim_rad=-.008,aerial_yaw_trim_start_angle_rad=-.13,aerial_yaw_trim_full_angle_rad=.13)
            p.update_prediction(0.,[-.35,-1.2,.8],[vx,3.,-.3])
            self.assertTrue(p.predictions)
            self.assertGreaterEqual(p.predictions[-1]['yaw_trim_rad'],-.008)
            self.assertLessEqual(p.predictions[-1]['yaw_trim_rad'],0.)
            normals.append(p.normal)
        self.assertLess(normals[0][0],normals[1][0])

    def test_stationary_or_departing_ball_does_not_lift_foot(self):
        for v in ([0,0,0],[0,-2.,0]):
            p=self.planner()
            for t in [0.,.02,.04]:p.update_prediction(t,[-.35,-1.2,.11],v)
            self.assertIsNone(p.move_start)
            pose,_,_,_=p.targets(.4)
            np.testing.assert_allclose(pose,p.initial_foot)

if __name__=='__main__':unittest.main()
