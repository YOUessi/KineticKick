from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from receive_pass_motion import ReceivePassMotion

class ReceivePassTests(unittest.TestCase):
    def sample(self,ball=None,velocity=None,forces=(300.,35.)):
        left=np.eye(4);left[:3,3]=[-.04,-.02,.035]
        right=np.eye(4);right[:3,3]=[-.17,.11,.035]
        root=np.eye(4);root[:3,3]=[0,0,.74]
        return dict(ball=np.array(ball if ball is not None else [-.27,-.22,.11]),
            velocity=np.array(velocity if velocity is not None else [0.,0.,0.]),spin=np.zeros(3),root=root,
            feet={'left':left,'right':right},forces=np.array(forces),root_speed=0.,root_tilt=0.,received=True)
    def ready(self,p,s=None):
        s=s or self.sample()
        for i in range(51):p.update(i*.02,**s)
        return p
    def test_moving_or_unsupported_ball_does_not_start_footwork(self):
        for state in [self.sample(velocity=[.3,0,0]),self.sample(forces=(300.,0.)),self.sample(ball=[-.27,-.22,.4])]:
            p=self.ready(ReceivePassMotion({}),state)
            self.assertFalse(p.active)
    def test_step_is_replanned_from_the_actual_stopped_ball(self):
        a=self.ready(ReceivePassMotion({}));b=self.ready(ReceivePassMotion({}),self.sample(ball=[-.22,-.22,.11]))
        self.assertTrue(a.active);self.assertTrue(b.active)
        np.testing.assert_allclose(b.step_goal[:3,3]-a.step_goal[:3,3],[.05,0,0],atol=1e-12)
        self.assertGreater(np.linalg.norm(a.step_goal[:2,3]-self.sample()['feet']['right'][:2,3]),.03)
    def test_requested_direction_sets_the_medial_face_and_followthrough(self):
        p=self.ready(ReceivePassMotion(dict(direction=[.6,-.8])))
        np.testing.assert_allclose(p.strike_rotation[:,1],[.6,-.8,0.],atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(p.direction),1.)
    def test_grounded_step_keeps_stance_heading_before_opening_for_the_pass(self):
        p=self.ready(ReceivePassMotion(dict(direction=[.2,-1.])))
        np.testing.assert_allclose(p.step_goal[:3,:3],self.sample()['feet']['right'][:3,:3],atol=1e-12)
        self.assertGreater(np.linalg.norm(p.strike_rotation-p.step_goal[:3,:3]),.1)

    def test_local_adjustment_does_not_create_an_overwide_stance(self):
        p=self.ready(ReceivePassMotion({}),self.sample(ball=[-.42,-.22,.11]))
        self.assertTrue(p.active)
        shift=np.linalg.norm(p.step_goal[:2,3]-self.sample()['feet']['right'][:2,3])
        self.assertLessEqual(shift,.130001);self.assertGreater(shift,.03)

    def test_target_point_is_resolved_from_measured_stop_position(self):
        p=self.ready(ReceivePassMotion(dict(target_xy=[.33,-1.02])))
        np.testing.assert_allclose(p.direction,[.6,-.8,0.],atol=1e-12)

    def test_elapsed_time_cannot_substitute_for_measured_step_landing(self):
        p=self.ready(ReceivePassMotion({}));s=self.sample(forces=(300.,0.));s['feet']['right'][2,3]=.12
        for i in range(51,301):p.update(i*.02,**s)
        self.assertNotIn(p.phase,('pass_lift','pass_strike','pass_follow','pass_recover','pass_stand'))
        self.assertFalse(any(e['kind']=='step_landed' for e in p.events))
    def test_transferring_weight_does_not_lift_an_unstable_body(self):
        p=self.ready(ReceivePassMotion({}));s=self.sample()
        p.transition('pass_unload',1.,s['feet']['right'])
        s['root_speed']=.12
        p.update(1.6,**s)
        self.assertEqual(p.phase,'pass_unload')
        s['root_speed']=.01;s['forces']=np.array([325.,3.])
        p.update(1.7,**s)
        self.assertEqual(p.phase,'pass_lift')

    def test_completed_step_may_proceed_when_already_stably_unloaded(self):
        p=self.ready(ReceivePassMotion({}));s=self.sample(forces=(325.,0.))
        p.update(1.4,**s)
        s['feet']['right'][2,3]=.12;p.update(1.6,**s)
        s['feet']['right']=p.step_goal.copy();s['feet']['right'][2,3]=.035;s['forces']=np.array([300.,30.])
        p.update(2.2,**s);self.assertEqual(p.phase,'step_load')
        s['forces']=np.array([325.,1.5]);p.update(3.1,**s)
        self.assertEqual(p.phase,'pass_unload')

    def test_after_passing_the_foot_returns_to_the_verified_ready_stance(self):
        p=self.ready(ReceivePassMotion({}));p.planted=p.step_goal.copy()
        p.transition('pass_recover',2.,p.step_goal)
        np.testing.assert_allclose(p.targets(3.)['foot'],p.initial_foot,atol=1e-12)

    def test_nonfinite_pass_target_is_rejected_before_movement(self):
        with self.assertRaises(ValueError):ReceivePassMotion(dict(target_xy=[float('nan'),0.]))

    def test_zero_direction_is_rejected(self):
        with self.assertRaises(ValueError):ReceivePassMotion(dict(direction=[0.,0.]))

if __name__=='__main__':unittest.main()
