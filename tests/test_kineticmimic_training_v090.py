"""v90 Contact+Coord+Impact: trained head and immutable PhysX split tests."""
import sys,json
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from kineticmimic_physics_v090 import sha,load_original_physics,sample_contact_predictor
from train_kineticmimic_v090 import (
 ContactImpactHead,performance,physical_outcome,rows_to_training,
 loss_for_network,inference,fit,score)

def test_contact_impact_head_outputs_future_4state_physx_and_robot_force():
 torch.manual_seed(90)
 m=ContactImpactHead(hidden=24)
 x=torch.randn(5,12,71)
 stance,hit,v,J=m(x)
 assert stance.shape==(5,4) and hit.shape==(5,)
 assert v.shape==(5,3) and J.shape==(5,)
 assert torch.isfinite(stance).all() and (J>0).all()

def test_contact_impact_head_joint_backward_respects_real_PhysX_mask():
 torch.manual_seed(44)
 m=ContactImpactHead(hidden=16)
 x=np.random.default_rng(80).normal(size=(12,12,71)).astype("float32")
 batch=dict(x=x,stance=np.arange(12)%4,
            impact=np.asarray([1.]*3+[0.]*9,dtype="float32"),
            delta_v=np.tile(np.asarray([0.,-4.,.2],dtype="float32"),(12,1)),
            impulse=np.asarray([2.2]*12,dtype="float32"))
 loss=loss_for_network(m,batch,np.arange(12),torch.tensor([3.]),False)
 assert loss>0 and torch.isfinite(loss)
 loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_training_negative_history_shuffles_only_past_not_future():
 torch.manual_seed(51)
 rng=np.random.default_rng(44)
 batch=dict(x=rng.normal(size=(8,12,71)).astype("float32"),
   stance=np.arange(8)%4,
   impact=np.asarray([0.,1.]*4,dtype="float32"),
   delta_v=rng.normal(size=(8,3)).astype("float32"),
   impulse=np.ones(8,dtype="float32"))
 m=ContactImpactHead(hidden=16)
 a=loss_for_network(m,batch,np.arange(8),torch.tensor([1.]),False)
 b=loss_for_network(m,batch,np.arange(8),torch.tensor([1.]),True,np.random.default_rng(9001))
 assert a.shape==b.shape==() and float(abs(a-b).detach())>1e-7
 np.testing.assert_allclose(batch["impact"],[0,1]*4)

def test_f1_never_contact_baseline_is_not_success():
 labels=np.asarray([0,0,1,0,1],dtype=bool)
 p=np.zeros(5)
 metrics=performance(labels,p)
 assert metrics["tp"]==0 and metrics["fn"]==2 and metrics["f1"]==0

def test_f1_perfect_contact_prediction():
 labels=np.asarray([0,1,1,0],dtype=bool)
 p=np.asarray([.1,.8,.95,.05])
 metric=performance(labels,p)
 assert metric["f1"]==1

def test_freeze_training_split_source_manifest():
 p=ROOT/"configs/kineticmimic_v090_study.json"
 if not p.exists():pytest.skip("First source/code SHA not yet precommitted")
 c=json.loads(p.read_text())
 assert c["version"]=="v0.1.90"
 assert (c["train_scene"],c["validation_scene"],c["test_scene"])==(
   "v084_C","v084_D","v085_center")
 assert not c["train_on_center_scene"] and not c["human_true_impact_available"]
 assert not c["model_is_29dof_actor"] and not c["elite_foot_kick_claim"]
 assert c["no_new_PhysX_GPU_training"] is True
 assert len(c["source_physx"])==9
 assert sum(x["scene"]=="v084_C" for x in c["source_physx"].values())==3
 assert sum(x["scene"]=="v084_D" for x in c["source_physx"].values())==3
 assert sum(x["scene"]=="v085_center" for x in c["source_physx"].values())==3
 assert c["epochs"]<=150
 for f,expected in c["source_sha256"].items():assert sha(ROOT/f)==expected

def test_real_source_physical_impact_has_g1_not_human_proxy():
 p=Path("/home/you/projects/kinetickick-wt-v084/reports/v084_C_v32_off.json")
 if not p.exists():pytest.skip("Tang original measured G1 200Hz archive only")
 e=load_original_physics(p,sha(p),"v084_C_v32")
 x=sample_contact_predictor(e)
 dv,J=physical_outcome(e)
 assert x["x"].shape[1:]==(12,71)
 assert dv.shape==(3,) and np.isfinite(dv).all()
 assert J>0 and e.first_impact_s is not None
 # Actual 100ms hazard label comes exclusively from *after* the final input
 # frame's real first foot-ball contact; no post-contact state in model input.
 assert x["query_time_s"][-1]<e.first_impact_s
 assert int(x["impact_next_100ms"].sum())>0

def test_end_to_end_tiny_contact_fitting_on_synthetic_labels_no_fake_success():
 torch.set_num_threads(2)
 rng=np.random.default_rng(42)
 N=40
 a=dict(x=rng.normal(size=(N,12,71)).astype("float32"),
   stance=(np.arange(N)%4).astype("int64"),
   impact=np.asarray([1 if i%5==0 else 0 for i in range(N)],dtype="float32"),
   delta_v=np.tile(np.asarray([0.,-4.,0.],dtype="float32"),(N,1)),
   impulse=np.ones(N,dtype="float32")*2.2)
 cfg=dict(seed=9001,hidden=16,epochs=4,early_stop_patience=2,batch_size=8,
   learning_rate=.001,weight_decay=.001)
 model,diag=fit(a,a,cfg,False)
 assert diag["best_validation_epoch"]>0
 result=score(model,a,False)
 assert result["n_windows"]==N and result["not_29dof_motor_trajectory_generated"] is True
 assert result["subject_human_contact_truth_unavailable"] is True
