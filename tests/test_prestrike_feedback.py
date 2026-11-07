"""Protect nominal/early commands and one-time reward credit at task boundaries."""
import importlib

import pytest
import torch


def test_structural_nominal_and_prefix_zero_then_latched_fade():
    m=importlib.import_module('prestrike_feedback')
    gate=m.PreStrikeEnvelope(2,'cpu')
    previous=torch.zeros(2,29)
    raw=torch.full((2,29),100.)
    for tick in range(75):
        delta=gate.update(torch.full((2,),tick*.02),raw,torch.tensor([0.,1.]),torch.zeros(2,dtype=torch.bool))
        assert torch.equal(delta[0],torch.zeros(29))
        if tick<=30:assert torch.equal(delta,torch.zeros_like(delta))
        assert float((delta-previous).abs().max())<=.008001
        assert float(delta.abs().max())<=.040001
        previous=delta.clone()
    frozen=previous.clone()
    for tick in range(75,100):
        delta=gate.update(torch.full((2,),tick*.02),-raw,torch.tensor([1.,1.]),torch.tensor([tick==75,tick==75]))
        if tick==75:assert torch.equal(delta,frozen)
        assert float((delta-previous).abs().max())<=.008001
        previous=delta.clone()
    assert torch.equal(delta,torch.zeros_like(delta))
    assert gate.eligible.tolist()==[True,True]
    gate.reset(torch.tensor([1]))
    assert gate.eligible.tolist()==[True,False]
    assert gate.exit_time.tolist()==[pytest.approx(1.5),-1.]


def test_events_consume_once_but_reset_only_selected_environments():
    m=importlib.import_module('prestrike_feedback')
    events=m.OnceEvents(2,'cpu')
    events.add('goal',torch.tensor([True,False]),20.)
    events.add('goal',torch.tensor([True,True]),20.)
    events.add('speed',torch.tensor([False,True]),-20.)
    assert events.consume().tolist()==[20.,0.]
    assert events.consume().tolist()==[0.,0.]
    events.add('goal',torch.tensor([True,True]),20.)
    assert events.consume().tolist()==[0.,0.]
    events.reset(torch.tensor([1]))
    events.add('goal',torch.tensor([True,True]),20.)
    assert events.consume().tolist()==[0.,20.]


def test_pre_actor_calls_frozen_post_actor_once_with_none_and_keeps_early_base(monkeypatch):
    m=importlib.import_module('prestrike_feedback')
    from types import SimpleNamespace
    task=m.PreStrikeTask.__new__(m.PreStrikeTask)
    task.n=1;task.device='cpu';task.elapsed=torch.tensor([0.]);task.gate=torch.tensor([1.])
    task.pre_gate=m.PreStrikeEnvelope(1,'cpu');task.pre_actor=None
    task.pre_first_touch=torch.tensor([-1.]);task.pre_previous=torch.zeros(1,29)
    task.pre_last_base=torch.zeros(1,29);task.debug=[]
    task.actual_delta=torch.zeros(1,29)
    task.env=SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _:SimpleNamespace(_scale=.5)))
    base=torch.linspace(-10.,10.,29)[None]
    calls=[]
    def parent(self,actions=None):
        calls.append(actions);self.elapsed+=1.;return base
    monkeypatch.setattr(m.PostShotRecoveryTask,'commands',parent)
    output=task.commands(torch.full((1,29),100.))
    assert calls==[None]
    assert torch.equal(output,base)
    assert torch.equal(task.pre_gate.actual,torch.zeros_like(base))


def test_speed_event_uses_original_absolute_tolerance_not_a_ratio_tolerance():
    m=importlib.import_module('prestrike_feedback')
    velocity=torch.tensor([[20.000001907348633,-37.],[20.,37.]])
    limits=torch.tensor([[20.,37.],[20.,37.]])
    assert m.actual_speed_excess(velocity,limits).tolist()==[True,False]
