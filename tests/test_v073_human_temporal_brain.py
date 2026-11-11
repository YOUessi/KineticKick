"""v73 self-supervised human 3D motion training: strict provenance and leakage tests."""
from pathlib import Path
import json,sys
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from train_human_temporal_brain_v073 import (FEATS,LANDMARKS,COS_COUNT,
 sha,load_verified,clip_windows,TemporalBrain,persistence,linear_velocity,
 human_metrics,weighted_loss,train_net,infer)

def example_pose(T=180):
 t=np.arange(T,dtype=float)/120
 xyz=np.zeros((T,19,3))
 for i in range(19):
  xyz[:,i,0]=.04*np.sin(1.3*t+i*.2)
  xyz[:,i,1]=.02*np.cos(1.9*t+i*.1)
  xyz[:,i,2]=(i/20)+.03*np.cos(t*2+i*.1)
 cos=np.column_stack([.75*np.sin(t*2+i*.3) for i in range(8)])
 return t,xyz,cos

def fake_source(path, subject="10",fps=120,impact=False,force=False,bad=False):
 t,xyz,cos=example_pose()
 valid=np.ones((len(t),19),bool)
 if bad:valid[38,4]=False;xyz[38,4,:]=np.nan
 np.savez_compressed(path,time_s=t,pelvis_relative_scale_normalized_xyz=xyz,
   marker_derived_human_8_cosine_relations=cos,primary_marker_valid=valid,
   original_human_subject=np.asarray(subject),original_frame_rate_hz=np.asarray(fps),
   true_impact_event_available=np.asarray(impact),
   force_plate_data_available=np.asarray(force),
   primary_landmark_names=np.asarray(["body_%02d"%i for i in range(19)]))
 return sha(path)

def test_read_original_120hz_19_and_8_channels(tmp_path):
 p=tmp_path/"10_01.npz"
 h=fake_source(p)
 f,v,t,n=load_verified(p,h,"10")
 assert f.shape==(180,65) and v.all()
 assert len(n)==19 and np.all(np.isfinite(f))
 assert abs(np.diff(t).mean()-1/120)<1e-8

@pytest.mark.parametrize("mutation,explain",[
  ({"subject":"11"},"subject"),
  ({"fps":60},"120Hz"),
  ({"impact":True},"impact"),
  ({"force":True},"GRF"),
])
def test_reject_untrusted_foot_impact_subject_and_rate(tmp_path,mutation,explain):
 p=tmp_path/"fake.npz"
 kw=dict(subject="10",fps=120,impact=False,force=False)
 kw.update(mutation)
 h=fake_source(p,**kw)
 with pytest.raises(ValueError,match="schema"):
  load_verified(p,h,"10")

def test_tampered_original_raw_source_SHA_rejected(tmp_path):
 p=tmp_path/"raw.npz"
 h=fake_source(p)
 with p.open("ab") as f:f.write(b"UNREGISTERED EDIT")
 with pytest.raises(ValueError,match="SHA drift"):
  load_verified(p,h,"10")

def test_missing_human_marker_not_zero_filled(tmp_path):
 p=tmp_path/"raw.npz"
 h=fake_source(p,bad=True)
 f,v,_,_=load_verified(p,h,"10")
 assert not v[38] and np.isnan(f[38]).any()
 a,b,_=clip_windows(f,v,24,12,3,np.zeros(65),np.ones(65))
 assert a.shape[1:]==(24,65) and b.shape[1:]==(12,65)
 assert np.isfinite(a).all() and np.isfinite(b).all()

def test_no_cross_sample_or_artificial_clip_boundaries():
 x=np.arange(95*65,dtype=float).reshape(95,65)*1e-5
 mask=np.ones(95,bool)
 a,b,starts=clip_windows(x,mask,24,12,12,np.zeros(65),np.ones(65))
 assert a.shape[0]==5 and b.shape[0]==5
 for z,u,st in zip(a,b,starts):
  np.testing.assert_allclose(z[-1],x[st+23],rtol=1e-7)
  np.testing.assert_allclose(u[0],x[st+24],rtol=1e-7)

def test_true_prediction_baselines_have_no_magic_future_access():
 rng=np.random.default_rng(14)
 x=rng.standard_normal((3,24,65)).astype("float32")
 p=persistence(x,12)
 v=linear_velocity(x,12)
 assert p.shape==v.shape==(3,12,65)
 np.testing.assert_allclose(p[:,0],x[:,-1])
 np.testing.assert_allclose(v[:,0],x[:,-1]+(x[:,-1]-x[:,-5])/4)
 assert not np.allclose(v,p)

def test_model_is_65d_wholehuman_and_gradients_finite():
 torch.manual_seed(99)
 m=TemporalBrain(24,12,hidden=24)
 x=torch.randn(4,24,65)
 y=m(x)
 assert y.shape==(4,12,65)
 assert torch.allclose(y,x[:,-1:,:].expand(-1,12,-1),atol=1e-6)
 loss=weighted_loss(y,torch.randn_like(y))
 loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_future_metrics_are_source_geometry_not_false_kinetics():
 t=np.zeros((2,12,65),dtype=np.float32)
 e=human_metrics(t,t,np.zeros(65),np.ones(65))
 assert e["normalized_xyz_landmark_euclidean_MAE"]==0.
 assert e["observed_8_segment_cosine_MAE"]==0.
 p=t.copy();p[:,:,0]=.01;p[:,:,57]=.2
 e=human_metrics(p,t,np.zeros(65),np.ones(65))
 assert e["normalized_xyz_landmark_euclidean_MAE"]>0
 assert e["observed_8_segment_cosine_MAE"]>0
 assert e["predicted_cosine_bound_violation_rate"]==0

def test_train_micro_two_models_no_GPU_no_test_subject_use():
 rng=np.random.default_rng(6)
 x=rng.standard_normal((24,24,65)).astype("float32")*.1
 y=np.tile(x[:,-1:,:],(1,12,1))
 cfg=dict(seed=7301,history_frames=24,future_frames=12,gru_hidden=24,
   learning_rate=.002,weight_decay=.001,epochs=3,early_stop_patience=2,batch_size=8)
 for shuffled in (False,True):
  m,meta=train_net(x,y,x[:8],y[:8],cfg,shuffled)
  assert meta["selection_uses_test_subject"] is False
  assert meta["negative_control_input_shuffled"] is shuffled
  assert meta["best_epoch"]<=3
  out=infer(m,x[:2])
  assert np.isfinite(out).all() and out.shape==(2,12,65)

def test_protocol_split_not_slide_window_heldout_leakage():
 p=ROOT/"configs/v073_human_temporal_learning.json"
 if not p.exists():pytest.skip("Research protocol not yet frozen")
 c=json.loads(p.read_text())
 assert c["train_clips"]==["10_01","10_02","10_03","10_05"]
 assert c["validation_clips"]==["10_06"]
 assert c["test_clips"]==["11_01"]
 assert c["professional_instestep_contact_groundtruth"] is False
 assert c["test_subject_seen_in_training"] is False
 assert c["no_CUDA_no_robot_policy_no_PhysX"] is True
 assert len(c["original_human_source"])==6

def test_reject_zero_gradient_loss_or_wrong_29dof_implicit_claim():
 with pytest.raises(ValueError,match="Invalid"):
  TemporalBrain(0,12)
 with pytest.raises(ValueError,match="65D|19x3"):
  TemporalBrain(24,12)(torch.zeros(1,24,29))
