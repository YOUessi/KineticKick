"""Gate/IK tests catch early corrections, first-touch reopening and pose projection."""
import importlib
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_support_loss_and_first_touch_fade_are_continuous_and_never_reopen():
    m = importlib.import_module('early_path_controller')
    gate = m.PathResidualGate(2, 'cpu')
    request = torch.full((2, 29), .06)
    support = torch.tensor([[100., 0.], [0., 100.]])
    previous = torch.zeros_like(request)
    for tick in range(75):
        now = torch.full((2,), tick*.02)
        delta, info = gate.update(now, request, support, torch.zeros(2, dtype=torch.bool))
        if tick <= 30: assert torch.equal(delta, torch.zeros_like(delta))
        assert torch.equal(delta[1], torch.zeros(29))
        assert float((delta-previous).abs().max()) <= .015001
        previous = delta.clone()
    assert float(previous[0, 0]) == pytest.approx(.06)
    first = previous.clone()
    for tick in range(75, 110):
        touched = torch.tensor([tick == 75, False])  # One sample suffices to latch.
        delta, info = gate.update(torch.full((2,), tick*.02), -request, support, touched)
        assert float((delta-previous).abs().max()) <= .015001
        if tick == 75: assert torch.equal(delta, first)
        previous = delta.clone()
    assert torch.equal(delta, torch.zeros_like(delta))
    assert gate.exit_time[0] == pytest.approx(1.5)
    gate.reset(torch.tensor([0]))
    assert gate.exit_time.tolist() == [-1., pytest.approx(2.1)]


def test_support_switch_does_not_bypass_slew_and_bad_state_is_rejected():
    m = importlib.import_module('early_path_controller')
    gate = m.PathResidualGate(1, 'cpu')
    for tick in range(55):
        gate.update(torch.tensor([tick*.02]), torch.full((1,29), .06), torch.tensor([[100.,0.]]), torch.tensor([False]))
    previous = gate.actual.clone()
    delta, _ = gate.update(torch.tensor([1.1]), torch.full((1,29), -.06), torch.tensor([[0.,100.]]), torch.tensor([False]))
    assert float((delta-previous).abs().max()) <= .015001
    with pytest.raises(ValueError):
        gate.update(torch.tensor([1.12]), torch.full((1,29), float('nan')), torch.tensor([[100.,0.]]), torch.tensor([False]))


def test_cartesian_ik_zero_is_exact_and_small_request_preserves_orientation():
    m = importlib.import_module('early_path_controller')
    from placement_strike import BatchedLeg
    leg = BatchedLeg(ROOT/'configs/paid_original/paid_main_resolved.urdf', 'cpu')
    q = torch.tensor([[-.4, .1, .05, .8, -.3, .04]])
    rotation = torch.eye(3)[None]
    root = torch.zeros(1,3)
    local = torch.tensor([.06,0.,-.025])
    delta, errors = m.path_ik_delta(leg,q,rotation,root,torch.zeros(1,2),local)
    assert torch.equal(delta,torch.zeros_like(q))
    before, r0, _ = leg.forward(q,rotation,root)
    point0=before+(r0@local).squeeze(-1)
    delta, errors = m.path_ik_delta(leg,q,rotation,root,torch.tensor([[.003,0.]]),local)
    after,r1,_=leg.forward(q+delta,rotation,root)
    point1=after+(r1@local).squeeze(-1)
    assert torch.allclose(point1-point0,torch.tensor([[.003,0.,0.]]),atol=3e-4)
    assert float((r1-r0).abs().max()) < .003
    assert float(delta.abs().max()) <= .060001
    # A baseline PD target already outside the soft interval is not projected.
    q[:,4] = 1.1
    delta,_=m.path_ik_delta(leg,q,rotation,root,torch.zeros(1,2),local)
    assert torch.equal(q+delta,q)
