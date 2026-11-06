import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import torch
from approach_adapter import approach_features, approach_command


def test_measured_capture_error_changes_command_and_protects_nominal():
    ref=torch.zeros(4,2);root=torch.tensor([[-.05,0.],[.05,0.],[-.05,0.],[-.05,0.]])
    dv=torch.tensor([.15,.15,0.,-.20]);velocity=torch.zeros(4,2);shift=torch.zeros(4,2)
    feature=approach_features(ref,velocity,root,velocity,shift,torch.ones(4),dv,.18,.10)
    assert torch.allclose(feature,torch.tensor([.375,-.375,0.,0.]))
    faster_left=velocity.clone();faster_left[:,0]=-.1
    assert approach_features(ref,velocity,root,faster_left,shift,torch.ones(4),dv,.18,.10)[0]>feature[0]
    result=approach_command(torch.ones(4,3),torch.ones(4,3,10)*.16,feature,torch.zeros(4,10),.2,.04)
    assert torch.allclose(result[0],torch.ones(10)*.04)
    assert torch.allclose(result[1],torch.ones(10)*-.04)
    assert torch.count_nonzero(result[2:])==0


def test_adapter_window_end_releases_without_sign_flip():
    last=torch.full((1,10),.07)
    out=approach_command(torch.zeros(1,3),torch.ones(1,3,10)*.16,torch.ones(1),last,.2,.04)
    assert torch.allclose(out,torch.full_like(out,.03))


def test_independent_audit_rejects_modified_actuator_request():
    import pytest
    from analyze_placement_strike import verify_approach
    names=[side+'_'+part+'_joint' for side in ['left','right'] for part in ['hip_pitch','hip_roll','hip_yaw','ankle_pitch','ankle_roll']]
    cfg={'approach_adapter':{'coefficients':[[.16]*10]*3,'maximum_rad':.2,'maximum_step_rad':.04},'fast_prefix_windows_s':[[0.,.05,.15,.35],[.2,.3,.5,.65],[.5,.65,.8,1.]]}
    row={'time_s':.1,'approach_feature':.75,'approach_correction_rad':[.02]*10,'approach_requested_rad':[.04]*10,'approach_target_before_rad':[.98]*10,'approach_soft_limits_rad':[[-1.,1.]]*10}
    assert verify_approach(row,cfg,[0.]*10,names,.15,.75)==[.02]*10
    row['approach_requested_rad'][0]=.06
    with pytest.raises(ValueError,match='projection'):
        verify_approach(row,cfg,[0.]*10,names,.15,.75)


def test_audit_rejects_self_consistent_stale_feature():
    import pytest
    from analyze_placement_strike import verify_approach
    cfg={'approach_adapter':{'coefficients':[[0.]*10]*3,'maximum_rad':.2,'maximum_step_rad':.04},'fast_prefix_windows_s':[[0.,.05,.15,.35]]*3}
    names=[side+'_'+part+'_joint' for side in ['left','right'] for part in ['hip_pitch','hip_roll','hip_yaw','ankle_pitch','ankle_roll']]
    row={'time_s':.1,'approach_feature':.3,'approach_correction_rad':[0.]*10,'approach_requested_rad':[0.]*10,'approach_target_before_rad':[0.]*10,'approach_soft_limits_rad':[[-1.,1.]]*10}
    with pytest.raises(ValueError,match='measured state'):
        verify_approach(row,cfg,[0.]*10,names,.15,.4)
