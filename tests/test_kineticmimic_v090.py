"""KineticMimic v90: real-PhysX provenance, learned contact and geometry losses."""
import sys,json,math
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from kineticmimic_physics_v090 import (
 G1ContactEpisode,stance_class,strict_physics,sample_contact_predictor,
 load_original_physics,sha)
from kineticmimic_loss_v090 import (
 TruthScope,validate_scopes,soft_dtw_path_cost,semantic_phase_cost,
 soft_phase_peak,phase_chain_hypothesis_loss,impact_prediction_loss,
 ContactModeHead,contact_prediction_loss,kineticmimic_multiobjective,
 GROUPS,IDS)

def fixture(n=2800,impact=313,success=True):
 t=np.arange(1,n+1)*.005
 q=np.zeros((n,29))
 feet=np.zeros((n,2))
 feet[:200,0]=300.
 feet[200:280,1]=350.
 feet[281:340,0]=350.
 if n>340:feet[340:,0]=300.
 force=np.zeros(n)
 force[impact:impact+2]=[300,50]
 return G1ContactEpisode(
  case="test",trace_source_sha256="f"*64,source_n_rows=n,true_physics_rows=n,
  terminal_annotation_count=0,times=t,q=q,dq=q,
  qdot_limits=np.ones(29)*20,
  qdot_ratio=np.ones(n)*.5,
  feet_normal_N=feet,stance_state=stance_class(feet),
  nonfoot_force_N=np.zeros(n),root_xyz=np.zeros((n,3)),
  com_xyz=np.zeros((n,3)),
  right_foot_point_velocity_w=np.zeros((n,3)),
  ball_xyz=np.zeros((n,3)),ball_velocity_xyz=np.zeros((n,3)),
  foot_ball_force_N=force,first_impact_s=float(t[impact]),
  measured_instep_impulse_fraction=1. if success else .72,
  original_ball_goal=True,original_recovered=True,original_termination=False,
  estimated_foot_ball_scalar_impulse_Ns=float(np.sum(force)*.005),
  observed_ball_speed_5_to_105ms_mps=4.8,
  original_reported_impact_Ns_by_surface={"instep":1.75},
  original_controller_verdict=True)

def test_stance_states_exact_four_classes_no_unsafe_labels():
 a=stance_class([[0,0],[150,0],[0,150],[150,150]])
 np.testing.assert_equal(a,[0,1,2,3])
 with pytest.raises(ValueError,match="finite"):
  stance_class([[float("nan"),3.]])
 with pytest.raises(ValueError,match="nonnegative"):
  stance_class([[-1.,0.]])

def test_actual_robot_hard_gates_cannot_be_replaced_with_soft_reward():
 x=fixture();assert strict_physics(x)["strict_pass"]
 from dataclasses import replace
 assert not strict_physics(replace(x,measured_instep_impulse_fraction=.72))["strict_pass"]
 assert not strict_physics(replace(x,original_recovered=False))["strict_pass"]
 assert not strict_physics(replace(x,original_termination=True))["strict_pass"]
 assert not strict_physics(replace(x,original_ball_goal=False))["strict_pass"]
 assert not strict_physics(replace(x,observed_ball_speed_5_to_105ms_mps=2.2))["strict_pass"]
 assert not strict_physics(replace(x,nonfoot_force_N=np.ones(2800)*25))["strict_pass"]
 assert not strict_physics(replace(x,qdot_ratio=np.ones(2800)*1.001))["strict_pass"]
 assert not strict_physics(replace(x,original_controller_verdict=False))["strict_pass"]
 assert not strict_physics(replace(x,times=x.times[:500],q=x.q[:500],dq=x.dq[:500],qdot_ratio=x.qdot_ratio[:500],feet_normal_N=x.feet_normal_N[:500],stance_state=x.stance_state[:500],nonfoot_force_N=x.nonfoot_force_N[:500],root_xyz=x.root_xyz[:500],com_xyz=x.com_xyz[:500],right_foot_point_velocity_w=x.right_foot_point_velocity_w[:500],ball_xyz=x.ball_xyz[:500],ball_velocity_xyz=x.ball_velocity_xyz[:500],foot_ball_force_N=x.foot_ball_force_N[:500]))["strict_pass"]

def test_contact_predictor_only_past_states_and_real_future_events():
 x=fixture()
 windows=sample_contact_predictor(x,stride=4,horizon=20,precontact_only=True)
 assert windows["x"].ndim==3 and windows["x"].shape[1:]==(12,71)
 assert windows["impact_next_100ms"].sum()>0
 assert np.any(windows["impact_next_100ms"]==0)
 assert windows["stance_after_100ms"].max()<=3
 assert windows["source_case"]=="test"
 assert np.all(np.diff(windows["query_time_s"])>0)

def test_softdtw_has_monotone_gradients_and_no_dummy_frame_copy():
 t=torch.linspace(0,1,20,dtype=torch.float64)
 a=torch.stack([torch.sin(2*math.pi*t+k*.1) for k in range(6)],dim=-1)[None]
 a.requires_grad_()
 b=a.detach().clone()
 c=semantic_phase_cost(a,b)
 assert torch.isfinite(c)
 c.backward()
 assert a.grad is not None and torch.isfinite(a.grad).all()
 other=torch.flip(b,dims=[1])
 assert float(semantic_phase_cost(a.detach(),other))>=float(c.detach())-1e-5

def test_collision_report_missing_scope_fails_closed():
 validate_scopes(TruthScope())
 with pytest.raises(ValueError,match="Unqualified human"):
  validate_scopes(TruthScope(human_true_foot_ball_contact=True))
 with pytest.raises(ValueError,match="certified"):
  validate_scopes(TruthScope(robot_installed_native_collider_pair_certified=True))

def test_event_stage_group_softpeak_timing_proxy_differentiable():
 d=torch.zeros((2,20,29),dtype=torch.float64,requires_grad=True)
 x=d.clone()
 for i,(group,t) in enumerate([("stance",3),("hip",6),("trunk",9),("swing",13),("counter_arm",9)]):
  x[:,t,IDS[group]]=1.+i*.05
 e,peaks=phase_chain_hypothesis_loss(x)
 assert torch.isfinite(e) and e>=0
 assert len(peaks)==5
 e.backward()
 assert d.grad is not None and torch.isfinite(d.grad).all()

def test_true_robot_impact_required_no_fake_human_impact():
 v=torch.zeros(2,3,requires_grad=True)
 a=torch.tensor([[0.,-4.5,0.],[0.,0.,0.]])
 p=torch.ones(2,requires_grad=True)
 t=torch.tensor([2.4,0.])
 loss=impact_prediction_loss(v,a,p,t,torch.tensor([True,False]))
 assert loss>0
 loss.backward()
 assert torch.isfinite(v.grad).all() and torch.isfinite(p.grad).all()
 with pytest.raises(ValueError,match="absent"):
  impact_prediction_loss(v.detach(),a,p.detach(),t,torch.tensor([False,False]))

def test_contact_head_lives_in_71d_real_physics_not_pose_only():
 m=ContactModeHead(hidden=16)
 x=torch.zeros(4,12,71)
 stance,hit=m(x)
 assert stance.shape==(4,4) and hit.shape==(4,)
 loss=contact_prediction_loss(stance,hit,torch.tensor([0,1,2,3]),torch.tensor([0.,0.,1.,0.]))
 loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
 with pytest.raises(ValueError,match="71"):
  m(torch.zeros(4,12,29))

def test_full_contact_coord_impact_objective_is_differentiable_not_hardware_certificate():
 x=torch.zeros(2,20,6,requires_grad=True)
 y=torch.zeros(2,20,6)
 dq=torch.rand(2,20,29,requires_grad=True)
 z=torch.randn(2,4,requires_grad=True)
 h=torch.randn(2,requires_grad=True)
 dv=torch.zeros(2,3,requires_grad=True)
 j=torch.ones(2,requires_grad=True)
 p=dict(predicted_ball_delta_v_xyz=dv,actual_ball_delta_v_xyz=torch.tensor([[0.,-4.,0.],[0.,0.,0.]]),
       predicted_scalar_foot_ball_impulse_Ns=j,actual_scalar_foot_ball_impulse_Ns=torch.tensor([2.2,0.]),
       has_true_robot_PhysX_contact=torch.tensor([True,False]))
 loss,terms=kineticmimic_multiobjective(x,y,dq,z,h,
      torch.tensor([1,2]),torch.tensor([0.,1.]),p)
 assert torch.isfinite(loss)
 loss.backward()
 for item in (x,dq,z,h,dv,j):assert item.grad is not None and torch.isfinite(item.grad).all()
 assert terms["human_contact_or_GRF_training_claim"] is False
 assert terms["physical_hardware_safety_certificate"] is False

def test_real_v85_raw_only_if_available_on_Tang():
 p=Path("/home/you/projects/kinetickick-wt-v085/reports/v085_center_cmu05_off.json")
 if not p.exists():pytest.skip("Tang historical original 29DoF physics only")
 episode=load_original_physics(p,sha(p),"historical_cmu05")
 assert episode.terminal_annotation_count==1
 assert episode.true_physics_rows==1028
 assert episode.first_impact_s is not None
 assert not strict_physics(episode)["strict_pass"]
 assert episode.estimated_foot_ball_scalar_impulse_Ns>0
