"""Actual URDF kinematics parity and orientation preservation for swing reach."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from placement_strike import BatchedLeg
from cartesian_strike import LegKinematics


def test_batched_leg_matches_existing_urdf_geometry():
    urdf=Path(__file__).resolve().parents[1]/'configs/paid_original/paid_main_resolved.urdf'
    batched=BatchedLeg(urdf,'cpu');original=LegKinematics(urdf)
    q=np.array([[-.6,.2,.05,1.3,.4,-.1],[-.8,.25,-.05,1.5,.3,-.15]])
    root=np.eye(4);root[:3,:3]=Rotation.from_euler('xyz',[.04,.10,-1.55]).as_matrix();root[:3,3]=[.1,-2.,.8]
    rotation=torch.tensor(np.repeat(root[None,:3,:3],2,0),dtype=torch.float32)
    position=torch.tensor(np.repeat(root[None,:3,3],2,0),dtype=torch.float32)
    p,r,j=batched.forward(torch.tensor(q,dtype=torch.float32),rotation,position)
    for i in range(2):
        pose,jacobian=original.forward(q[i],root)
        np.testing.assert_allclose(p[i],pose[:3,3],atol=1e-6)
        np.testing.assert_allclose(r[i],pose[:3,:3],atol=1e-6)
        np.testing.assert_allclose(j[i],jacobian,atol=1e-6)
    new,error,angle=batched.translate(torch.tensor(q,dtype=torch.float32),rotation,position,torch.tensor([[.02,0.],[-.02,0.]]),torch.zeros(2))
    assert bool((error<.001).all())
    assert bool((angle<.005).all())
    # Different proximal angles require ankle corrections to preserve the world
    # foot orientation; holding the ankle angles would not satisfy this contract.
    assert bool(((new[:,4:]-torch.tensor(q[:,4:])).abs().amax(-1)>.005).all())


def test_normal_task_reaches_recorded_fast_pose_without_changing_other_row():
    urdf=Path(__file__).resolve().parents[1]/'configs/paid_original/paid_main_resolved.urdf'
    leg=BatchedLeg(urdf,'cpu',.4)
    actual=[0.03411770984530449, -3.653594732284546, 0.7318834066390991, -0.7013224363327026, -0.022375522181391716, 0.10279947519302368, 0.7050378322601318]
    nominal=[-0.104093536734581, -3.699066162109375, 0.7796681523323059, -0.7171198725700378, -0.0335208885371685, -0.04407443106174469, 0.6947464942932129]
    q=torch.tensor([[-.89117837,.27800646,.03943897,1.49902534,.45378640,-.18657124]],dtype=torch.float32)
    def rotation(pose):
        return torch.tensor(Rotation.from_quat([*pose[4:],pose[3]]).as_matrix()[None],dtype=torch.float32)
    r=rotation(actual);pos=torch.tensor([actual[:3]],dtype=torch.float32)
    desired,normal_frame,_=leg.forward(q,rotation(nominal),torch.tensor([nominal[:3]],dtype=torch.float32))
    unchanged,unchanged_frame,_=leg.forward(q,r,pos)
    out,error,angle=leg.translate(q.repeat(2,1),r.repeat(2,1,1),pos.repeat(2,1),torch.zeros(2,2),torch.zeros(2),
        reference=(torch.cat([desired,unchanged]),torch.cat([normal_frame,unchanged_frame])),normal_only=torch.tensor([True,False]))
    assert float(error[0])<.005 and float(angle[0])<.01
    assert float((out-q).abs().max())<=.400001
    assert bool((out>=leg.limits[:,0]-1e-6).all()) and bool((out<=leg.limits[:,1]+1e-6).all())
    torch.testing.assert_close(out[1],q[0],atol=1e-6,rtol=0)
