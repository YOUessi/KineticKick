"""v41 real style activation independent from original soccer-error correction.

Unit tests use synthetic support/qdot and source-data contracts. Physical
success is assessed only by a separate unmodified original PhysX scorer.
"""
from pathlib import Path
from unittest.mock import patch
import torch,pytest,numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from coordination_style_v041 import CoordinationStyleReferenceTask,prespecified_style_safety_gate
from coordination_reference_v040 import CoordinationReferenceTask

CFG=dict(
 enabled=True,nominal_style_activation=1.,
 max_joint_rad=.006,max_step_rad=.0012,first_s=1.05,last_s=1.56,
 support_min_N=80.,swing_max_N=40.,max_qdot_ratio=.9,
 minimum_ball_distance_m=0.,maximum_ball_distance_m=1.15)

def test_nominal_soccer_context_zero_must_not_cancel_human_style_reference():
    # Explicit reproduction of v40 original zero-gate root cause:
    # center ball self.gate==0 but a loaded left-foot stance should permit style.
    legacy_error_gain=torch.tensor([0.])
    eligible,detail=prespecified_style_safety_gate(
        now=torch.tensor([1.42]),support=torch.tensor([[641.,0.]]),
        qd_ratio=torch.tensor([.315]),root_ball_distance=torch.tensor([.574]),
        contact=torch.tensor([False]),cfg=CFG)
    assert legacy_error_gain.item()==0.
    assert eligible.item()
    assert all(x.item() for x in detail.values())

def test_style_safety_blocks_unloaded_over_speed_post_contact_far_ball():
    cases=[
      (1.42,0.,0.,.2,.5,False),
      (1.42,300.,0.,1.02,.5,False),
      (1.42,300.,0.,.2,1.3,False),
      (1.42,300.,0.,.2,.5,True),
      (1.70,300.,0.,.2,.5,False),
      (1.42,300.,60.,.2,.5,False)]
    for now,left,right,qd,dist,touch in cases:
        a,_=prespecified_style_safety_gate(
           torch.tensor([now]),torch.tensor([[left,right]]),
           torch.tensor([qd]),torch.tensor([dist]),
           torch.tensor([touch]),CFG)
        assert not bool(a.item())

def test_separate_style_rejects_nans():
    with pytest.raises(ValueError):
        prespecified_style_safety_gate(
            torch.tensor([1.42]),torch.tensor([[641.,float("nan")]]),
            torch.tensor([.3]),torch.tensor([.5]),
            torch.tensor([False]),CFG)

def test_v41_zero_path_routes_exactly_one_parent_without_style_act():
    task=object.__new__(CoordinationStyleReferenceTask)
    task.style_config={**CFG,"enabled":False}
    task.coord_enabled=False
    task.n=1
    task.debug=[]
    parent_action=torch.randn(1,29)
    with patch.object(CoordinationReferenceTask,"commands",return_value=parent_action) as p:
        out=task.commands(None)
    p.assert_called_once_with(None)
    assert out is parent_action

def test_prespecified_reference_artifact_has_sufficient_phase_overlap():
    path=ROOT/"experiments/coordination_projection_v040/train_clip00001_center/g1_joint_delta_50hz.npz"
    if not path.exists():pytest.skip("Actual local human-proxy G1 reference only on Tang")
    with np.load(path,allow_pickle=False) as z:
        t=z["controller_time_s"]
        x=z["delta_joint_position_rad"]
    mask=(t>=1.40)&(t<=1.50)
    assert np.any(abs(x[mask])>1e-6)
    assert x.shape==(700,29)
    assert np.max(abs(x))<=CFG["max_joint_rad"]+1e-6
