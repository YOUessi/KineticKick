"""Offline retarget conventions and analytic original-G1 Jacobian checks."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
# Pinocchio 2.7 and ezc3d are optional Tang-local research dependencies.
pytest.importorskip('pinocchio')
pytest.importorskip('ezc3d')
from v160_human_reference import Robot, angle_quaternions, interpolate_markers, marker_frame, source_grid
from scipy.spatial.transform import Rotation


def test_resampling_retains_both_endpoints_and_marks_missing_input():
    t,st=source_grid(436,120.,50.)
    assert t[0]==st[0]==0. and st[-1]==435/120
    np.testing.assert_allclose(np.diff(t),.02)
    xyz=np.zeros((436,2,3));xyz[:,0,0]=np.arange(436);mask=np.ones((436,2),bool);mask[-1,0]=False
    y,observed=interpolate_markers(xyz,mask,np.arange(436)/120,st)
    assert not observed[-1,0] and y[-1,0,0]==434 and observed[-1,1]
    mask[30:34,0]=False
    with pytest.raises(ValueError,match='three-frame'):
        interpolate_markers(xyz,mask,np.arange(436)/120,st)


def test_axis_handedness_and_temporal_quaternion_hemisphere():
    R=marker_frame(np.array([[2.,0,0]]),np.array([[.3,1.,0]]))
    np.testing.assert_allclose(R[0],np.eye(3))
    turns=Rotation.from_euler('z',[178,180,182],degrees=True).as_matrix()
    q=angle_quaternions(turns);assert np.all((q[1:]*q[:-1]).sum(-1)>0)


def test_analytic_robot_directions_match_joint_finite_difference():
    import json
    master=Path('/home/you/projects/kinetickick');urdf=master/'configs/paid_original/paid_main_resolved.urdf';schema=master/'reports/paid_physics_parity_clean_paid.json'
    if not urdf.is_file() or not schema.is_file():pytest.skip('requires original Tang URDF and runtime link order')
    model=Robot(urdf,json.loads(schema.read_text()));q=np.clip(np.linspace(-.12,.15,29),model.low+.01,model.high-.01)
    _,jac=model.features(q);direction=np.linspace(.2,.8,29);h=1e-6
    actual=(model.features(q+h*direction)[0]-model.features(q-h*direction)[0])/(2*h)
    np.testing.assert_allclose(jac@direction,actual,atol=1e-6)
