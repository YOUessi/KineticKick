"""v46 whole-body post-contact anticipatory landing reference, CPU contracts."""
import torch,pytest,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from contact_landing_transfer_v046 import (
    ContactLandingTransferTask,prepare_landing_weights,landing_phase_reference)
from coordination_style_v041 import CoordinationStyleReferenceTask

CFG=dict(early_contact_s=.06,late_contact_s=.62,ramp_s=.08,fade_s=.20,
 left_support_min_N=50.,right_unloaded_max_N=90.,
 right_capsule_max_height_m=.35,right_descent_min_mps=.04,
 reference_gain_rad=.028,maximum_delta_rad=.016,maximum_step_rad=.002)

def state(rel=.18,force=(180.,0.),height=.12,descent=.5):
    qd=torch.tensor([[6.]*29],dtype=torch.float32)
    limit=torch.tensor([[20.]*29])
    return landing_phase_reference(
       torch.tensor([rel]),qd,limit,torch.tensor([list(force)]),
       torch.tensor([height]),torch.tensor([descent]),CFG,torch.ones(29))

def test_true_postcontact_prelanding_changes_nonzero_fullbody_reference():
    out,diag=state()
    assert diag["qualified"].item()
    assert diag["stance"].item()
    assert diag["raw_phase_gain"].item()>.1
    assert out.shape==(1,29)
    assert bool((out.abs()>.002).any())
    assert (out<0).all()
    assert float(out.abs().max())<=.0160005

def test_no_motion_before_actual_foot_ball_contact_and_after_landing_window():
    for rel in (-10.,-.001,0.,.04,.63,1.01):
        delta,info=state(rel=rel)
        assert float(delta.abs().max())==0.
    out,diag=state(rel=.2,force=(0.,1000.),height=1.,descent=0.)
    assert not diag["qualified"].item()
    assert float(out.abs().max())==0.

def test_predicted_right_landing_can_gate_after_true_ball_contact():
    delta,diag=state(rel=.22,force=(0.,0.),height=.19,descent=.25)
    assert diag["anticipated_right_landing"].item()
    assert diag["qualified"].item() and float(delta.abs().max())>0
    zero,other=state(rel=.22,force=(0.,0.),height=.38,descent=.25)
    assert not other["qualified"].item() and float(zero.abs().max())==0.

def test_wholebody_joint_weights_cover_all_original_29_dof():
    import json
    r=json.loads((ROOT/"reports/v041_human_style_center.json").read_text())
    q=prepare_landing_weights(r["joint_names"],"cpu")
    assert q.shape==(29,) and float(q.min())>0 and float(q.max())==1.0
    names=r["joint_names"]
    ankle=names.index("left_ankle_roll_joint")
    elbow=names.index("right_elbow_joint")
    assert q[ankle]>q[elbow]
    with pytest.raises(ValueError):prepare_landing_weights(names[:-1],"cpu")

def test_nonfinite_and_wrong_motor_limits_do_not_create_safety_claim():
    with pytest.raises(ValueError):
        landing_phase_reference(torch.tensor([.2]),torch.ones(1,29),torch.zeros(1,29),
          torch.ones(1,2),torch.ones(1),torch.ones(1),CFG,torch.ones(29))
    with pytest.raises(ValueError):
        landing_phase_reference(torch.tensor([.2]),torch.full((1,29),float("nan")),
          torch.ones(1,29),torch.ones(1,2),torch.ones(1),torch.ones(1),CFG,torch.ones(29))

def test_disabled_landing_controller_returns_original_v41_action_identically():
    task=object.__new__(ContactLandingTransferTask)
    task.transfer_cfg={**CFG,"enabled":False}
    task.debug=[];task.n=1
    expected=torch.linspace(-1.,1.,29)[None]
    with patch.object(CoordinationStyleReferenceTask,"commands",return_value=expected) as parent:
        actual=task.commands(None)
    parent.assert_called_once_with(None)
    assert actual is expected
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
