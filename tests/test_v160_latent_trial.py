import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from v160_latent_trial import (ConditionalVAE, Motion, clip_splits, count, hold_and_extrapolation,
                               make_windows, matching_width, mlp, reference_violations)


def trajectory():
    frames=np.arange(38,dtype=float)
    q=np.repeat((frames*.01)[:,None],29,axis=1)
    p=np.stack((frames*.03,np.zeros_like(frames),np.ones_like(frames)),axis=1)
    return Motion('example','left',q,p,Rotation.from_euler('z',frames*.005),'fixture')


def test_future_edits_cannot_change_prefix_and_pelvis_frame():
    motion=trajectory();before=make_windows(motion,8,10,10)
    motion.joints[8:]+=1.;motion.root_p[8:]+=7.
    after=make_windows(motion,8,10,10)
    assert np.array_equal(before.context[0],after.context[0])
    assert not np.array_equal(before.target[0],after.target[0])
    assert before.anchors.tolist()==[7,17,27]
    fresh=trajectory();world=Rotation.from_euler('zyx',[.7,.3,-.1])
    fresh.root_p=world.apply(fresh.root_p)+np.array([1.,.4,-.8])
    fresh.root_r=world*fresh.root_r
    transformed=make_windows(fresh,8,10,10)
    assert np.allclose(before.context,transformed.context,atol=1e-12)
    assert np.allclose(before.target,transformed.target,atol=1e-12)


def test_kinematic_reference_crossing_includes_first_predicted_step():
    q=np.zeros((1,10,35));q[:,:,3]=.11
    limits=np.tile([-1.,1.,5.],(29,1))
    position,velocity=reference_violations(q,np.zeros((1,29)),limits)
    assert not position.any()
    assert np.argwhere(velocity).tolist()==[[0,0,3]]
    w=make_windows(trajectory(),8,10,10);hold,linear=hold_and_extrapolation(w,8,10)
    assert not hold.any()
    assert np.allclose(linear,w.target,atol=1e-12)


def test_overlapping_complete_clip_split_is_rejected_and_models_are_matched():
    with pytest.raises(ValueError,match='one nonempty partition'):
        clip_splits({'train':['kickA'],'validation':['kickB'],'test':['kickA']})
    cv=ConditionalVAE(282,350,64,8)
    width=matching_width(282,350,count(cv));det=mlp(282,width,350)
    assert abs(count(cv)-count(det))/count(cv)<.01
    h,mu,lv=cv.zprior(torch.zeros(2,282))
    assert cv.decode(h,mu).shape==(2,350)
    assert lv.shape==(2,8)
