"""29DoF URDF proposal gate is conservative, test fixtures cannot authorize motor commands."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from v097_candidate_safety_audit import qualified_limits,verify_candidate_screen
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

def test_joint_urdf_limits_name_bound(tmp_path):
    urdf=tmp_path/'robot.urdf'
    urdf.write_text('<robot name="a">'+''.join(
      f'<joint name="{j}" type="revolute"><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
       for j in G1_JOINT_ORDER)+'</robot>')
    low,high=qualified_limits(urdf,G1_JOINT_ORDER)
    assert low.shape==high.shape==(29,)
    assert np.all(low==-1) and np.all(high==1)
    with pytest.raises(ValueError):
        qualified_limits(urdf,list(G1_JOINT_ORDER)[::-1])
def test_fails_closed_on_angle_exceed_even_when_qdot_within_limit():
    pred=np.full((2,20,29),.5)
    q0=np.zeros((2,29));q0[1,0]=.98
    low=np.full(29,-1.);high=np.ones(29)
    result=verify_candidate_screen(pred,q0,np.ones(29),low,high,np.zeros((2,29)))
    assert result['original_all29_urdf_angle_violating_windows']==1
    assert result['maximum_original_urdf_angle_excess_rad']>.15
    assert result['qdot_limit_violating_samples']==0
    assert not result['physical_actuation_authorized']
def test_fails_closed_even_with_zero_movement():
    result=verify_candidate_screen(np.zeros((1,20,29)),np.zeros((1,29)),
      np.ones(29),np.full(29,-1.),np.ones(29),np.zeros((1,29)))
    assert result['original_all29_urdf_angle_violating_windows']==0
    assert not result['physical_actuation_authorized']
    assert result['requires_authenticated_native_collider_distance_gradient']
def test_rejects_bad_all29_state_shapes():
    with pytest.raises(ValueError):
        verify_candidate_screen(np.zeros((1,20,28)),np.zeros((1,29)),
          np.ones(29),np.full(29,-1.),np.ones(29),np.zeros((1,29)))
