import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from phase_coordinated_v063 import phase_trajectory,admissibility,WINDOW,AMPLITUDE,SLEW
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

def eligible_data(n=2):
    return (torch.tensor([1.2]*n),torch.tensor([[150.,20.]]*n),
        torch.tensor([.6]*n),torch.tensor([.7]*n),torch.tensor([.3]*n),
        torch.zeros(n,dtype=torch.bool),torch.zeros(n),torch.zeros(n,29),
        torch.stack((torch.full((n,29),-2.),torch.full((n,29),2.)),-1))
def test_prespecified_multi_joint_synthetic_not_low_mrad():
    a=phase_trajectory()
    assert a.shape==(700,29)
    assert np.isfinite(a).all()
    assert np.max(np.abs(a))>.025
    assert np.max(np.abs(a))<=AMPLITUDE+1e-6
    assert int(np.count_nonzero(a[int(1.2/.02)]))>=20
    assert np.all(a[0]==0) and np.all(a[200]==0)
    assert len(G1_JOINT_ORDER)==29
def test_zero_outside_window_and_smooth_boundary():
    a=phase_trajectory()
    assert np.all(a[np.arange(700)*.02<WINDOW[0]]==0)
    assert np.all(a[np.arange(700)*.02>WINDOW[1]]==0)
    assert np.max(np.abs(np.diff(a,axis=0)))<.025  # actuator slew cap enforced separately
def test_all_safety_gates_fail_closed():
    a=list(eligible_data())
    ok,flags=admissibility(*a)
    assert ok.all() and all(z.all() for z in flags.values())
    variants=[(0,torch.tensor([2.,2.])),(1,torch.zeros(2,2)),
        (2,torch.ones(2)),(3,torch.ones(2)*1.5),
        (4,torch.ones(2)*.1),(5,torch.ones(2,dtype=torch.bool)),
        (6,torch.ones(2)*10.),(7,torch.ones(2,29)*3.)]
    for index,new in variants:
        b=list(a);b[index]=new
        result,_=admissibility(*b)
        assert not result.any(),index
def test_finite_guards_reject_untrusted_input():
    a=list(eligible_data())
    a[2][0]=float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):
        admissibility(*a)
def test_protocol_is_not_human_training():
    import json
    p=Path(__file__).resolve().parents[1]
    cfg=json.loads((p/"configs/g1_phase_synthetic_candidate_v063.json").read_text())
    assert cfg["phase_coordinated_v063"]["source"]=="synthetic_temporal_ablation_not_human_learned"
    assert cfg["coordination_style_v041"]["enabled"] is False
    assert cfg["human_coordination_reference"]["enabled"] is False
