"""Actual two-shank geometry, swept prediction and bounded command contracts."""
from pathlib import Path
import json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from running_reference_geometry import ReferenceGeometry,segment_distance
ROOT=Path(__file__).resolve().parents[1]

def states():
    d=json.loads((ROOT/'reports/v027_final_heldout_confirm_right_near_slower.json').read_text())
    rows=[min(d['traces'][0],key=lambda r:abs(r['time_s']-t)) for t in [1.50,1.54,1.55]]
    return d['joint_names'],torch.tensor([r['joint_pos'] for r in rows]),torch.tensor([r['joint_vel'] for r in rows])

def test_recorded_capsules_and_intermediate_path_match_independent_fk():
    from shank_clearance import ShankClearance
    names,q,v=states();m=ShankClearance(ROOT/'configs/paid_original/paid_main_resolved.urdf',names,'cpu')
    original=ReferenceGeometry(ROOT);expected=[]
    for state in q:
        shapes=original.primitives(state.numpy(),[0,0,0],[1,0,0,0]);a=next(x for x in shapes if x['name']=='left_knee_link');b=next(x for x in shapes if x['name']=='right_knee_link')
        expected.append(segment_distance(a['start'],a['end'],b['start'],b['end'])-a['radius']-b['radius'])
    np.testing.assert_allclose(m.gap(q),expected,atol=2e-7)
    np.testing.assert_allclose(expected,[.048587,.007500,.001440],atol=2e-6)
    predicted,when=m.swept_gap(q[1:2],v[1:2],.04)
    assert float(m.gap(q[1:2]+.04*v[1:2])[0])>.04
    assert float(predicted[0])<.005 and 0<float(when[0])<.04

def test_measured_closing_path_requests_only_bounded_active_abduction():
    from shank_clearance import ShankClearance
    names,q,v=states();m=ShankClearance(ROOT/'configs/paid_original/paid_main_resolved.urdf',names,'cpu')
    q=q[:1].repeat(3,1);v=v[:1].repeat(3,1)
    delta,info=m.command(q,v,torch.tensor([True,False,False]),torch.tensor([0.,0.,-.08]),margin=.012,horizon=.06,maximum=.10,maximum_step=.04)
    assert -.040001<=float(delta[0])<0.
    assert float(delta[1])==0.
    assert abs(float(delta[2])+.04)<1e-6  # Smooth release, not a new inactive correction.
    assert float(info['predicted_gap'][0])<.012
    target=q.clone();target[:,names.index('right_hip_roll_joint')]+=delta
    assert float(m.gap(target)[0])>float(m.gap(q)[0])


def test_segments_cover_interior_parallel_and_degenerate_cases():
    from shank_clearance import segment_gap
    a=torch.tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    b=torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,0.,0.]])
    c=torch.tensor([[.5,-1.,0.],[.2,.2,0.],[1.,0.,0.]])
    d=torch.tensor([[.5,1.,0.],[.8,.2,0.],[1.,0.,0.]])
    torch.testing.assert_close(segment_gap(a,b,c,d,.1),torch.tensor([-.1,.1,.9]),atol=1e-6,rtol=0)


def test_independently_clipped_ankle_never_starts_a_new_inactive_correction():
    from shank_clearance import ankle_compensation
    hip=torch.tensor([-.06,-.06,0.]);previous=torch.tensor([0.,0.,.08]);active=torch.tensor([False,True,False])
    actual=ankle_compensation(hip,previous,active,.04)
    torch.testing.assert_close(actual,torch.tensor([0.,.04,.04]),atol=1e-6,rtol=0)


def test_current_evaluation_does_not_include_reused_heldout_ids():
    from strike_cases import case_group
    assert case_group('heldout_confirm_right_near_slower','0.1.28')=='development'
    assert case_group('heldout_right_far_slower','0.1.28')=='development'
    assert case_group('heldout_v028_near_slow_13','0.1.28')=='evaluation'
    assert case_group('heldout_v028_left_near_slow','0.1.28')=='evaluation'
