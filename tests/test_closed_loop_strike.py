import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from closed_loop_expert import cosine_window,measured_context_gate
from shooting_control import StrikeResidual

class ClosedLoopContracts(unittest.TestCase):
    def test_window_matches_frozen_strike_including_plateau(self):
        frames=torch.linspace(60,100,81,dtype=torch.float64)
        actual=cosine_window(frames,[65.,75.,82.,91.]).numpy()
        ref=StrikeResidual([{'window':[65,75,82,91]}]*81,['j'],1.,0.,[[-2.,2.]])
        np.testing.assert_allclose(actual,ref.envelope(frames.numpy())[0][:,0],atol=1e-12)

    def test_measured_nominal_context_cannot_activate_learning(self):
        shift=torch.tensor([[0.,0.],[0.,.1],[.05,0.],[0.,0.],[1e-6,-1e-6]])
        dv=torch.tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,.2,0.],[0.,1e-6,0.]])
        g=measured_context_gate(shift,dv)
        self.assertEqual(float(g[0]),0.)
        self.assertEqual(float(g[-1]),0.)
        self.assertTrue(bool((g[1:4]>.99).all()))
        self.assertTrue(torch.equal(torch.randn(5,30)[[0,4]]*g[[0,4],None],torch.zeros(2,30)))


class RecurrentMaskContract(unittest.TestCase):
    def test_inactive_expert_memory_does_not_advance(self):
        from types import SimpleNamespace
        from closed_loop_expert import FrozenRecurrent
        torch.manual_seed(26)
        bank=FrozenRecurrent.__new__(FrozenRecurrent)
        bank.n=3;bank.device='cpu';bank.native=None
        bank.h=torch.randn(2,3,128);bank.c=torch.randn(2,3,128)
        recurrent=torch.nn.LSTM(4,128,num_layers=2)
        head=torch.nn.Linear(128,29)
        bank.model=SimpleNamespace(paid_obs_mean=torch.zeros(1,4),paid_obs_denominator=torch.ones(1,4),memory_a=SimpleNamespace(rnn=recurrent),actor=head)
        before=(bank.h.clone(),bank.c.clone());obs=torch.randn(3,4)
        with torch.no_grad():
            y,(h,c)=recurrent(obs[[0,2]].unsqueeze(0),(before[0][:,[0,2]].contiguous(),before[1][:,[0,2]].contiguous()))
            output=bank.act(obs,torch.tensor([True,False,True]),torch.zeros(3))
        torch.testing.assert_close(output[[0,2]],head(y.squeeze(0)))
        self.assertTrue(torch.equal(bank.h[:,1],before[0][:,1]))
        self.assertTrue(torch.equal(bank.c[:,1],before[1][:,1]))
        self.assertTrue(torch.equal(output[1],torch.zeros(29)))
        bank.reset(torch.tensor([2]))
        self.assertTrue(torch.equal(bank.h[:,2],torch.zeros(2,128)))
        self.assertTrue(torch.equal(bank.h[:,1],before[0][:,1]))

if __name__=='__main__':unittest.main()
