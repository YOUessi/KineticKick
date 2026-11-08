"""v38 physical joint geometry, sweep-direction and frozen-envelope unit tests."""
import math
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from foot_sweep_direction_v038 import (
    measured_point_velocity,lateral_sweep_target,damped_point_ik_delta,SweepEnvelope)
from placement_strike import BatchedLeg
from cartesian_strike import LegKinematics

def test_measured_point_velocity_includes_omega_cross_lever():
    q=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float32)
    p,v,lever=measured_point_velocity(
        torch.zeros(1,3),q,torch.zeros(1,3),
        torch.tensor([[0.,0.,10.]]),[.115,0.,-.025])
    torch.testing.assert_close(v,torch.tensor([[0.,1.15,0.]]),atol=1e-6,rtol=0)
    torch.testing.assert_close(p,lever,atol=0,rtol=0)

def test_sweep_target_rotates_with_goal_and_velocity():
    v=torch.tensor([[-1.5,-5.],[-1.5,-5.]])
    ball=torch.tensor([[-.10,-4.0],[-.10,-4.0]])
    goal=torch.tensor([[.7,-6.52],[.7,-6.52]])
    base,detail=lateral_sweep_target(v,ball,goal)
    assert base.shape==(2,2)
    assert (base.abs().max()<=.022+1e-8)
    assert float(detail["lateral_velocity_mps"][0])<0
    angle=0.63
    R=torch.tensor([[math.cos(angle),-math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]])
    turned,rot=lateral_sweep_target(v@R.T,ball@R.T,goal@R.T)
    torch.testing.assert_close(turned,base@R.T,atol=1e-7,rtol=1e-6)
    for values in (detail["lateral_velocity_mps"],rot["lateral_velocity_mps"]):
        torch.testing.assert_close(values,values[:1].repeat(2),atol=1e-7,rtol=0)

def test_real_urdf_jacobian_matches_independent_fk_and_small_displacement():
    urdf=ROOT/"configs/paid_original/paid_main_resolved.urdf"
    kin=BatchedLeg(urdf,"cpu")
    independent=LegKinematics(urdf)
    x=torch.tensor([[.2,-.08,.12,.5,-.15,.05]],dtype=torch.float32)
    rootR=torch.eye(3)[None]
    rootP=torch.zeros(1,3)
    p,R,J=kin.forward(x,rootR,rootP)
    T,jref=independent.forward(x.numpy()[0],np.eye(4))
    np.testing.assert_allclose(p.detach().numpy()[0],T[:3,3],atol=2e-7)
    np.testing.assert_allclose(R.detach().numpy()[0],T[:3,:3],atol=2e-7)
    np.testing.assert_allclose(J.detach().numpy()[0],jref,atol=2e-7)
    local=torch.tensor([.115,0.,-.025])
    target=torch.tensor([[.005,-.012]])
    dq,diagnostics=damped_point_ik_delta(kin,x,rootR,rootP,local,target,
                                           max_joint_rad=.022)
    assert dq.shape==(1,6)
    assert float(dq.abs().max())<=.022+1e-7
    p1,R1,_=kin.forward(x+dq,rootR,rootP)
    before=p+(R@local[None,:,None]).squeeze(-1)
    after=p1+(R1@local[None,:,None]).squeeze(-1)
    actual=after[:,:2]-before[:,:2]
    predicted=diagnostics["linearized_translation_xy_m"]
    torch.testing.assert_close(actual,predicted,atol=2e-4,rtol=0.1)
    # Strict zero input must produce exact zero output.
    z,_=damped_point_ik_delta(kin,x,rootR,rootP,local,torch.zeros_like(target))
    assert (z==0).all()

def test_slew_nominal_protection_latch_and_release():
    e=SweepEnvelope(2,"cpu")
    now=torch.tensor([.65,1.50])
    raw=torch.full((2,29),1.)
    active=torch.tensor([0.,1.])
    delta,info=e.update(now,raw,active,torch.tensor([False,False]))
    assert (delta[0]==0.).all()
    assert float(delta[1].abs().max())<=.004+1e-7
    assert info["exit_time_s"][1]<0
    e.update(torch.tensor([.75,1.6]),raw,active,torch.tensor([False,True]))
    assert e.exit_time[1]>0
    delta,_=e.update(torch.tensor([1.,1.81]),raw,active,torch.tensor([False,True]))
    assert (delta[1]==0).all()
    e.reset(torch.tensor([0,1]))
    assert (e.actual==0).all()
    assert (e.exit_time==-1).all()

def test_invalid_state_and_goal_fails_closed():
    try:
        lateral_sweep_target(torch.zeros(1,2),torch.zeros(1,2),torch.zeros(1,2))
    except ValueError:
        pass
    else:
        raise AssertionError("Cannot normalize nonexistent goal vector")


def test_disabled_task_returns_unchanged_parent_action():
    # Critical regression: old pre-v38 prototype ignored enabled=false and
    # actually applied 0.004 rad at t=1.44s, breaking physics parity.
    from unittest.mock import patch
    from foot_sweep_direction_v038 import FootSweepDirectionTask
    from post_shot_recovery import PostShotRecoveryTask
    task=object.__new__(FootSweepDirectionTask)
    task.sweep_cfg={"enabled":False}
    task.n=1
    task.debug=[]
    parent_action=torch.linspace(-1,1,29)[None].clone()
    with patch.object(PostShotRecoveryTask,"commands",return_value=parent_action) as parent:
        a=task.commands(None)
    parent.assert_called_once_with(None)
    assert a is parent_action
    torch.testing.assert_close(a,parent_action,atol=0,rtol=0)
