"""Reference transform respects the distinct pelvis/torso local frames."""
from types import SimpleNamespace
import math
import torch
from support_reference_frame import SupportReferenceFrame
from support_reference_observation import transform_running_reference


def rz(angle):
    c,s=math.cos(angle),math.sin(angle)
    return torch.tensor([[[c,-s,0.],[s,c,0.],[0.,0.,1.]]])


def test_world_transform_keeps_actual_state_and_handles_two_local_frames():
    t=SupportReferenceFrame(1,'cpu',dict(minimum_force_N=80.,support_confirmation_s=.015,duration_s=.2))
    forces=torch.tensor([[100.,0.]])
    for _ in range(3):t.after_physics(forces)
    feet=torch.tensor([[[1.,2.,.1],[2.,3.,.1]]])
    actual=feet+torch.tensor([[[2.,-1.,0.]]])
    args=(torch.tensor([True]),forces,feet,actual,torch.tensor([0.]),torch.tensor([math.pi/2]))
    t.update(torch.tensor([1.]),*args)
    t.update(torch.tensor([1.2]),*args)
    root=torch.tensor([[.4,-.2,.8]]);anchor=torch.tensor([[.5,-.1,1.1]])
    root_R=rz(.3);anchor_R=rz(-.4)
    obs=torch.randn(1,176)
    # All three reference points represent the same known world point [2,2,.4].
    world=torch.tensor([[2.,2.,.4]])
    for start in (154,157):obs[:,start:start+3]=(root_R.transpose(-1,-2)@(world-root)[...,None]).squeeze(-1)
    obs[:,160:163]=(anchor_R.transpose(-1,-2)@(world-anchor)[...,None]).squeeze(-1)
    target_R=rz(.2)
    obs[:,163:169]=(anchor_R.transpose(-1,-2)@target_R)[:,:,:2].reshape(1,6)
    obs[:,61:64]=torch.tensor([[1.,2.,3.]])
    before=obs.clone()
    out=transform_running_reference(obs,t,root,root_R,anchor,anchor_R,torch.tensor([True]))
    # Rotate [1,0] about [1,2], then shift pivot by [2,-1].
    expected=torch.tensor([[3.,2.,.4]])
    for start in (154,157):torch.testing.assert_close(root+(root_R@out[:,start:start+3,None]).squeeze(-1),expected)
    torch.testing.assert_close(anchor+(anchor_R@out[:,160:163,None]).squeeze(-1),expected)
    torch.testing.assert_close(anchor_R@out[:,163:169].reshape(1,3,2),rz(.2+math.pi/2)[:,:,:2],atol=2e-6,rtol=1e-6)
    torch.testing.assert_close(out[:,61:64],torch.tensor([[-2.,1.,3.]]),atol=2e-6,rtol=1e-6)
    unchanged=torch.ones(176,dtype=torch.bool);unchanged[61:64]=False;unchanged[154:169]=False
    assert torch.equal(out[:,unchanged],before[:,unchanged])
    assert torch.equal(obs,before)


def test_zero_transform_and_inactive_rows_are_exact_identity():
    t=SupportReferenceFrame(2,'cpu',dict(minimum_force_N=80.,support_confirmation_s=.015,duration_s=.2))
    obs=torch.randn(2,176);p=torch.zeros(2,3);r=torch.eye(3).expand(2,3,3)
    out=transform_running_reference(obs,t,p,r,p,r,torch.ones(2,dtype=torch.bool))
    assert out is obs
    t.alpha[:]=.5;t.yaw_rate[:]=2.
    # Inactive recurrent rows are not advanced or modified by this utility.
    out=transform_running_reference(obs,t,p,r,p,r,torch.tensor([True,False]))
    assert torch.equal(out[1],obs[1])
    torch.testing.assert_close(out[0,63],obs[0,63]+2.)
