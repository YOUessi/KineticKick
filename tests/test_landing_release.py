"""State/geometry contracts for post-contact reference release; no simulator."""
import torch
import pytest
from landing_release import capsule_landing_state, LandingRelease

CFG=dict(mode='all',lookahead_s=.16,maximum_height_m=.16,predicted_margin_m=.015,minimum_descent_mps=.30,separation_s=.015,separation_gap_m=.005,ball_force_max_N=2.,maximum_ground_force_N=40.,duration_s=.08,phase_lag_scale_s=.08)

def state(n=1,**changes):
    result=dict(right_lowest_height_m=torch.full((n,),.08),min_vertical_velocity_mps=torch.full((n,),-1.),predicted_clearance_m=torch.full((n,),-.08),capsule_ball_gap_m=torch.full((n,),.02))
    for key,value in changes.items():result[key]=torch.full((n,),value)
    return result

def separated(gate,n=1):
    for gap in [.01,.02,.03,.04]:
        gate.after_physics(torch.ones(n,dtype=torch.bool),torch.zeros(n),torch.full((n,),gap),.005)

def decision(gate,t,geometry=None,pulse=True,ground=0.):
    n=gate.n
    return gate.command(torch.full((n,),t),state(n) if geometry is None else geometry,torch.full((n,),pulse,dtype=torch.bool),torch.full((n,),ground),phase_lag_s=torch.full((n,),.12))

def test_capsule_prediction_includes_angular_motion_and_surface_radii():
    points=torch.tensor([[-.1,0.,0.],[.1,0.,0.]])
    result=capsule_landing_state(points,torch.tensor([.02,.02]),torch.tensor([[0.,0.,.3]]),torch.eye(3)[None],torch.zeros(1,3),torch.tensor([[0.,2.,0.]]),torch.tensor([[.3,0.,.3]]),.11,torch.tensor([0.]),.16)
    assert float(result['right_lowest_height_m'][0])==pytest.approx(.28)
    assert float(result['min_vertical_velocity_mps'][0])==pytest.approx(-.2)
    assert float(result['predicted_clearance_m'][0])==pytest.approx(.248)
    assert float(result['capsule_ball_gap_m'][0])==pytest.approx(.07)

def test_no_release_before_contact_separation_pulse_completion_or_airborne_descent():
    g=LandingRelease(1,'cpu',CFG)
    assert float(decision(g,1.)[0])==1.
    separated(g)
    assert float(decision(g,1.,pulse=False)[0])==1.
    assert float(decision(g,1.,ground=50.)[0])==1.
    assert float(decision(g,1.,state(min_vertical_velocity_mps=.1))[0])==1.
    assert float(decision(g,1.,state(right_lowest_height_m=.3))[0])==1.
    assert float(g.trigger_time_s[0])==-1.

def test_smooth_release_never_rearms_when_foot_rises_after_trigger():
    g=LandingRelease(1,'cpu',CFG);separated(g)
    values=[float(decision(g,t)[0]) for t in [1.,1.02,1.04,1.06,1.08]]
    assert values==pytest.approx([1.,.896484375,.5,.103515625,0.],abs=2e-6)
    assert float(decision(g,1.2,state(right_lowest_height_m=.8,min_vertical_velocity_mps=2.))[0])==0.

def test_recontact_cancels_unconfirmed_separation_and_batch_reset_is_local():
    g=LandingRelease(2,'cpu',CFG);separated(g,2)
    g.after_physics(torch.ones(2,dtype=torch.bool),torch.tensor([20.,0.]),torch.tensor([.05,.05]),.005)
    values=decision(g,1.)
    assert g.trigger_time_s.tolist()==[-1.,1.]
    g.reset(torch.tensor([1]));assert g.trigger_time_s.tolist()==[-1.,-1.]
    assert torch.all(values==1.)

def test_bad_bounds_and_unexpected_physics_interval_fail():
    with pytest.raises(ValueError):LandingRelease(1,'cpu',{**CFG,'duration_s':0.})
    g=LandingRelease(1,'cpu',CFG)
    with pytest.raises(ValueError):g.after_physics(torch.tensor([True]),torch.tensor([0.]),torch.tensor([.1]),.02)


def test_release_blends_the_whole_projection_operator_continuously():
    from landing_release import retain_operator
    raw=torch.tensor([[1.2,-.7],[.2,.1]],dtype=torch.float64)
    projected=torch.tensor([[.5,-.3],[.25,.15]],dtype=torch.float64)
    assert torch.equal(retain_operator(raw,projected,torch.ones(2)),projected)
    assert torch.equal(retain_operator(raw,projected,torch.zeros(2)),raw)
    almost_off=retain_operator(raw,projected,torch.full((2,),1e-6))
    assert float((almost_off-raw).abs().max())<1e-6
    torch.testing.assert_close(retain_operator(raw,projected,torch.full((2,),.5)),(raw+projected)/2)


def test_unknown_ball_force_cannot_confirm_separation():
    g=LandingRelease(1,'cpu',CFG);separated(g)
    g.after_physics(torch.tensor([True]),torch.tensor([float('inf')]),torch.tensor([.1]),.005)
    decision(g,1.)
    assert float(g.separated_s[0])==0. and float(g.trigger_time_s[0])==-1.


def test_phase_lag_scales_release_and_zero_lag_preserves_the_original():
    g=LandingRelease(2,'cpu',CFG);separated(g,2)
    force=torch.zeros(2);pulse=torch.ones(2,dtype=torch.bool)
    r=g.command(torch.ones(2),state(2),pulse,force,phase_lag_s=torch.tensor([0.,.04]))
    assert torch.equal(r,torch.ones(2))
    assert g.trigger_time_s.tolist()==[-1.,1.]
    r=g.command(torch.full((2,),1.08),state(2),pulse,force,phase_lag_s=torch.tensor([0.,.12]))
    assert torch.equal(r,torch.tensor([1.,.5]))
    g.reset(torch.tensor([1]))
    assert float(g.release_strength[1])==0.


@pytest.mark.parametrize('field',['right_lowest_height_m','min_vertical_velocity_mps','predicted_clearance_m'])
def test_nonfinite_geometry_cannot_trigger_release(field):
    g=LandingRelease(1,'cpu',CFG);separated(g)
    decision(g,1.,state(**{field:float('-inf')}))
    assert float(g.trigger_time_s[0])==-1.

def test_nonfinite_ground_force_cannot_trigger_release():
    g=LandingRelease(1,'cpu',CFG);separated(g)
    decision(g,1.,ground=float('-inf'))
    assert float(g.trigger_time_s[0])==-1.
