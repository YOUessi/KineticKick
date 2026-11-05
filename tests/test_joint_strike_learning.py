import unittest
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from joint_strike_core import bounded_phase_step,residual_action

class ResidualContract(unittest.TestCase):
    def test_zero_residual_preserves_recorded_action_exactly(self):
        base=torch.tensor([[.1234567,-.7654321]])
        scale=torch.tensor([[.3,.4]]);offset=torch.tensor([[.1,-.2]])
        limits=torch.tensor([[[-2.,2.],[-2.,2.]]])
        actual=residual_action(base,torch.zeros_like(base),scale,offset,limits,.25)
        self.assertTrue(torch.equal(actual,base))
        actual=residual_action(base,torch.tensor([[100.,-100.]]),scale,offset,limits,.25)
        self.assertLessEqual(float(((actual-base)*scale).abs().max()),.250001)

    def test_tiny_residual_does_not_clip_existing_pd_reference(self):
        # The frozen policy requests ankle targets beyond the kinematic joint
        # interval; original actuator/physical limits still constrain motion.
        base=torch.tensor([[2.,-2.]])
        limits=torch.tensor([[[-.5,.5],[-.5,.5]]])
        actual=residual_action(base,torch.tensor([[-1e-5,1e-5]]),1.,0.,limits,.25)
        self.assertLess(float((actual-base).abs().max()),1e-4)
        outward=residual_action(base,torch.tensor([[1.,-1.]]),1.,0.,limits,.25)
        self.assertTrue(torch.equal(outward,base))

    def test_phase_is_causal_bounded_and_does_not_drift_after_strike(self):
        frame=torch.zeros(2);elapsed=torch.zeros(2)
        for _ in range(200):
            new=bounded_phase_step(frame,elapsed,torch.tensor([100.,-100.]))
            self.assertTrue(bool((new>=frame).all()))
            elapsed+=1
            self.assertLessEqual(float((new-elapsed).abs().max()),7.50001)
            frame=new
        self.assertLessEqual(float((frame-elapsed).abs().max()),7.50001)
        self.assertTrue(torch.equal(bounded_phase_step(torch.tensor([10.]),torch.tensor([10.]),torch.tensor([0.])),torch.tensor([11.])))

if __name__=='__main__':unittest.main()
