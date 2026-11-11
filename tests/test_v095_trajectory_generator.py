"""Synthetic-only math contracts; no invented human GRF or actual G1 physics."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v095_trajectory_generator as g
def names():
 return tuple(x for v in g.GROUPS.values() for x in v)
def synthetic(case):
 n=150; rng=np.random.default_rng(40 if case=="train" else 50)
 x=np.zeros((n,73))
 x[:,35:64]=rng.uniform(-.4,.4,(n,29))
 c=np.zeros((n,7));c[:,0]=1
 return dict(case=case,certification=True,joint_names=names(),base=x,contact=c,
  source_sha256="a"*64)
def test_all29_partition():
 ix=g.group_indices(names())
 assert sum(len(v) for v in ix.values())==29
 with pytest.raises(ValueError):g.group_indices(names()[:-1])
def test_forecast_no_future_feature_and_200hz_relabel():
 z=synthetic("train"); x,y,s=g.make_windows(z,"full")
 assert x.shape[1]==80 and y.shape[1:]==(20,29)
 assert np.array_equal(x[0,:73],z["base"][0])
 assert np.array_equal(y[0,0],z["base"][1,35:64])
 assert s[-1]+20<len(z["base"])
def test_actual_learned_full29_predictions_vs_different_controls():
 z=synthetic("train")
 model=g.learn([z],"full")
 pred,truth,_=g.infer(model,z)
 ix=g.group_indices(z["joint_names"])
 assert pred.shape==truth.shape and pred.shape[2]==29
 assert not np.array_equal(pred,truth)
 assert not np.array_equal(pred,g.apply_ablation(pred,"time_reversed",ix))
 assert np.all(g.apply_ablation(pred,"freeze_arms",ix)[:,:,ix["left_counterarm"]]==0)
 assert np.all(g.apply_ablation(pred,"strike_leg_only",ix)[:,:,ix["trunk"]]==0)
 assert g.score(pred,truth,ix)["windows"]==len(pred)
def test_never_train_on_heldout_and_no_overwrite(tmp_path):
 a=synthetic("train")
 with pytest.raises(ValueError,match="EPISODE_LEAKAGE"):
  g.experiment([a],[a],tmp_path/"x.json")
 with pytest.raises(FileExistsError):
  o=tmp_path/"existing.json";o.write_text("untouched")
  g.experiment([a],[synthetic("test")],o)
 assert o.read_text()=="untouched"
def test_strict_features_and_ablation_guard():
 z=synthetic("train")
 with pytest.raises(ValueError):g.make_windows(z,"future_impulse")
 with pytest.raises(ValueError):g.apply_ablation(np.zeros((1,20,29)),"unregistered",g.group_indices(names()))


def test_integrate_only_current_measured_q0_and_speed_limits():
 v=np.zeros((2,20,29),dtype=float)
 v[:,:,0]=.2
 q0=np.ones((2,29),dtype=float)*.3
 lim=np.ones(29)
 q=g.integrate_measured_q0_29dof(v,q0,lim)
 assert q.shape==(2,20,29)
 assert np.isclose(q[0,-1,0],.38)
 assert np.array_equal(q[0,:,1],np.ones(20)*.3)
 with pytest.raises(ValueError):
  g.integrate_measured_q0_29dof(v,q0,np.ones(28))
 with pytest.raises(ValueError):
  g.integrate_measured_q0_29dof(v,q0,lim,dt_s=.01)
