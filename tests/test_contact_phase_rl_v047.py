"""CPU functional checks for original 5ms G1 qdot-margin RL reward."""
import torch
import pytest
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import post_shot_objective_v047 as v47

def test_real_speed_margin_penalty_starts_before_original_joint_limit():
    state=dict(post_active=torch.tensor([True,True,True,False]),
               substep_qd_ratio_peak=torch.tensor([.40,.70,.87,1.5]))
    def fake_original(s,objective):
        z=torch.zeros(4)
        return z,dict(original_stance_reward=z)
    old=v47.original_recovery_objective
    v47.original_recovery_objective=fake_original
    try:
        value,parts=v47.recovery_objective_v047(state,{})
    finally:
        v47.original_recovery_objective=old
    torch.testing.assert_close(value[:2],torch.zeros(2),atol=0,rtol=0)
    assert value[2]<0 and value[2]>-20.
    assert value[3]==0
    torch.testing.assert_close(value,parts["preemptive_real_joint_speed_margin_cost"],atol=0,rtol=0)

def test_finite_guard_rejects_bad_physx_input(monkeypatch):
    monkeypatch.setattr(v47,"original_recovery_objective",
        lambda state,config:(torch.zeros(1),{}))
    state=dict(post_active=torch.tensor([True]),substep_qd_ratio_peak=torch.tensor([float("nan")]))
    with pytest.raises(ValueError):
        v47.recovery_objective_v047(state,{})

def test_source_shaping_cannot_modify_frozen_recovery_safety_threshold():
    assert v47.FIXED["warning_ratio"]==.70
    assert v47.FIXED["limit_ratio"]==1.
    assert v47.FIXED["weight"]==2.5
    assert not v47.FIXED["maximum_penalty_per_second"]<1.
