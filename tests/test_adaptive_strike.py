import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from train_running_balance import parse_args

class TrainingRandomization(unittest.TestCase):
    def args(self, *extra):
        with patch.object(sys, 'argv', ['train', '--motion', __file__, '--output', '/tmp/kinetickick-nonexistent-args-v024', *extra]):
            return parse_args()

    def test_velocity_jitter_is_explicit_bounded_and_defaults_to_frozen_reset(self):
        self.assertEqual(self.args().initial_velocity_jitter, 0.)
        self.assertEqual(self.args('--initial-velocity-jitter', '.25').initial_velocity_jitter, .25)
        for v in ('nan', '-.1', '.6'):
            with self.assertRaises(SystemExit): self.args('--initial-velocity-jitter', v)

class MeasuredFeedback(unittest.TestCase):
    def planner(self):
        from adaptive_strike import StrikeFeedback
        return StrikeFeedback([0.,1.,2.,3.], [[0.,0.],[0.,-2.],[0.,-4.],[0.,-6.]], [0.,-4.], {})

    def test_ball_and_body_state_change_commands_without_a_case_id(self):
        p=self.planner()
        nominal=p.update(1.3,[0.,-2.6],[0.,-2.],[0.,-4.],300.,False)
        q=self.planner()
        shifted=q.update(1.3,[.03,-2.6],[0.,-2.],[.08,-3.9],300.,False)
        self.assertAlmostEqual(nominal['phase_s'],1.3)
        self.assertGreater(shifted['phase_s'],nominal['phase_s'])
        self.assertGreater(shifted['foot_delta_xy'][0],0.)
        self.assertLessEqual(abs(shifted['phase_s']-1.3),.12)

    def test_contact_latches_and_phase_never_runs_backwards(self):
        p=self.planner()
        last=-1.
        for i in range(100):
            t=i*.02
            r=p.update(t,[0.,-2*t],[0.,-2.],[.05 if i<75 else 9.,-4.],300.,i>=75)
            self.assertGreaterEqual(r['phase_s'],last)
            self.assertLessEqual(abs(r['phase_s']-t),.120001)
            self.assertLessEqual(max(map(abs,r['foot_delta_xy'])),.12)
            last=r['phase_s']
        self.assertTrue(r['contact_latched'])
        self.assertLess(abs(r['placement_xy'][0]),.06)

    def test_invalid_state_fails_instead_of_creating_joint_targets(self):
        p=self.planner()
        with self.assertRaises(ValueError): p.update(1.,[float('nan'),0.],[0.,-2.],[0.,-4.],0.,False)

class FootPlacement(unittest.TestCase):
    def test_translation_keeps_striking_ankle_and_respects_joint_bounds(self):
        import numpy as np
        from adaptive_strike import retarget_swing_translation
        from cartesian_strike import LegKinematics
        leg=LegKinematics(Path(__file__).resolve().parents[1]/'configs/paid_original/paid_main_resolved.urdf')
        q=np.array([-.891,.278,.039,1.499,.454,-.187]);root=np.eye(4)
        out,error=retarget_swing_translation(leg,q,root,[.04,.02])
        np.testing.assert_array_equal(out[-2:],q[-2:])
        self.assertLess(error,.005)
        self.assertLessEqual(max(abs(out-q)),.25)
        zero,error=retarget_swing_translation(leg,q,root,[0.,0.])
        np.testing.assert_allclose(zero,q,atol=1e-12)

if __name__ == '__main__': unittest.main()
