"""Geometry and temporal support checks requiring Tang's existing Pinocchio ABI."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from v162_human_support import Geometry, contiguous, difference, source


def test_contact_run_boundaries_are_half_open_and_do_not_bridge_motion():
    assert contiguous([False, True, True, False, True])==[(1,3),(4,5)]
    assert contiguous([False,False])==[]
    assert contiguous([True,True])==[(0,2)]


def test_derivative_penalties_preserve_constant_or_linear_movement():
    t=np.arange(5.)
    np.testing.assert_allclose(difference(len(t),1) @ np.ones_like(t),0.)
    np.testing.assert_allclose(difference(len(t),2) @ (2.3+t*4.1),0.,atol=1e-12)


def test_capsule_position_jacobian_includes_articulation_and_world_root_translation():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/v162_human_support.json').read_text())
    assets=Path('/home/you/projects/kinetickick')
    robot=source.Robot(assets/cfg['urdf'],json.loads((assets/cfg['schema']).read_text()))
    geom=Geometry(robot)
    state=np.r_[np.clip(np.zeros(29),robot.low+.03,robot.high-.03),.08,.03,.8]
    rot=Rotation.from_euler('xyz',[.2,.13,-.4]).as_matrix()
    _,_,old,height,jac=geom.at(state,rot)
    inc=np.random.default_rng(123).normal(size=32)*1e-7
    _,_,moved,new_height=geom.at(state+inc,rot,False)
    for j in range(2):
        np.testing.assert_allclose(moved[j]-old[j],jac[j] @ inc,atol=2e-13)
        np.testing.assert_allclose(new_height[j]-height[j],(jac[j] @ inc)[:,2],atol=2e-13)
        np.testing.assert_allclose(jac[j][:,:,-3:],np.broadcast_to(np.eye(3),(len(jac[j]),3,3)))
