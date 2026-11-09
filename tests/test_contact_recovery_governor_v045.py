"""v45 contact-conditioned all-29-joint speed governor unit/safety interface tests.

A small new movement target is not an actual safety proof. All real original
qdot and contact checks must be independently tested by Isaac/PhysX.
"""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from contact_recovery_governor_v045 import braking_reference,ContactRecoveryGovernorTask
from coordination_style_v041 import CoordinationStyleReferenceTask

SETTINGS=dict(warning_ratio=.68,maximum_ratio=.98,
              maximum_position_brake_rad=.025,
              maximum_position_step_rad=.004,
              after_contact_min_s=.10,after_contact_max_s=1.25,
              separation_confirm_s=.015)

def test_measured_future_contact_guard_is_zero_before_contact():
    q=torch.tensor([[0.]*29],dtype=torch.float32)
    q[0,1]=30.
    original=torch.full((1,29),37.)
    for enabled in (False,True):
        result,info=braking_reference(q,original,torch.tensor([enabled]),SETTINGS)
        assert result.shape==(1,29)
        assert (result[0,0]==0)
        if enabled:
            assert float(result[0,1])<0
            assert float(result.abs().max())<=.0250001
        else:
            assert (result==0).all()
        assert info["actual_qdot_ratio"].shape==(1,29)

def test_control_direction_opposes_joint_motion_and_never_exceeds_reference_cap():
    original=torch.tensor([[20.]*29])
    q=torch.zeros(1,29)
    q[0,0]=10.
    q[0,1]=18.
    q[0,2]=-18.
    q[0,3]=40.
    action,info=braking_reference(q,original,torch.tensor([True]),SETTINGS)
    assert float(action[0,0])==0.
    assert float(action[0,1])<0.
    assert float(action[0,2])>0.
    assert abs(float(action[0,3])+SETTINGS["maximum_position_brake_rad"])<1e-7
    assert torch.isfinite(action).all()
    assert info["danger_actuator_count"].item()==3

def test_invalid_original_joint_safety_state_must_fail_closed():
    with pytest.raises(ValueError):
        braking_reference(torch.zeros(1,29),torch.zeros(1,29),torch.tensor([True]),SETTINGS)
    with pytest.raises(ValueError):
        braking_reference(torch.zeros(1,28),torch.ones(1,28),torch.tensor([True]),SETTINGS)
    with pytest.raises(ValueError):
        q=torch.ones(1,29);q[0,0]=float("nan")
        braking_reference(q,torch.ones(1,29),torch.tensor([True]),SETTINGS)

def test_disabled_physics_governor_returns_the_unchanged_v41_parent_object():
    task=object.__new__(ContactRecoveryGovernorTask)
    task.guard_cfg={**SETTINGS,"enabled":False}
    task.n=1
    task.debug=[]
    parent_action=torch.linspace(-1,1,29)[None]
    with patch.object(CoordinationStyleReferenceTask,"commands",return_value=parent_action) as parent:
        output=task.commands(None)
    parent.assert_called_once_with(None)
    assert output is parent_action
    torch.testing.assert_close(output,parent_action,atol=0,rtol=0)

def test_not_a_bogus_v44_machine_learned_safety_certificate():
    # The actual v44 model had 12 false-safe *correlated windows* and is
    # deliberately not loaded or imported by this new governor module.
    import inspect
    import contact_recovery_governor_v045 as g
    source=inspect.getsource(g)
    assert "torch.load" not in source
    assert "contact_risk_model_v044" not in source
    assert "write_root" not in source
    assert "set_joint_velocity_target" not in source
    assert "braking_reference" in source
