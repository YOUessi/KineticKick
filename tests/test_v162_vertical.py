import numpy as np
from scipy.spatial.transform import Rotation
from v162_vertical_audit import measured_pose
from v162_vertical import correction


def test_first_high_toe_cannot_conceal_flat_loaded_following_samples():
    t=np.arange(1,25)*.005
    r=Rotation.from_euler('y',[80.]*11+[35.]*13,degrees=True)
    p=np.tile([0.,0.,.28],(24,1));mask=np.zeros(24,dtype=bool);mask[[10,11]]=True
    points=[(11,None,None,1.),(12,None,None,9.)]
    result=measured_pose(t,p,r,mask,points,[1.,0.,0.])
    assert result['checks']['first_toe_down']
    assert not result['checks']['weighted_toe_down']
    assert not result['pass']


def test_padding_is_checked_across_precontact_steps_and_plant_ends():
    t=np.arange(1,25)*.005;r=Rotation.from_euler('y',[85.]*24,degrees=True)
    p=np.tile([0.,0.,.28],(24,1));p[8,2]=.13
    mask=np.zeros(24,dtype=bool);mask[12]=True
    result=measured_pose(t,p,r,mask,[(13,None,None,2.)],[1.,0.,0.])
    assert not result['checks']['padded_clearance']
    assert result['checks']['first_dorsal_forward']
    np.testing.assert_array_equal(correction(1.,np.ones(18)),0.)
    np.testing.assert_array_equal(correction(3.,np.ones(18)),0.)
