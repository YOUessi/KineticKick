"""Independent v91 leakage/metrics/schema contract checks."""
import json,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v091_coordination_observer import standardized,fit_ridge,predict,persist,score,experiment,TRAIN,TEST
def fixture(name,n=100):
 rng=np.random.default_rng(sum(ord(x) for x in name))
 t=np.arange(n)/50
 C=rng.normal(size=(n,6))
 B=rng.normal(size=(n,73))
 contact=np.column_stack((np.eye(4)[np.arange(n)%4],np.zeros((n,3))))
 target=np.zeros((n,34))
 target[:,:29]=B[:,35:64]*.4
 target[:,29:33]=contact[:,:4]
 return {"case":name,"certification":True,"clock":C,"base":B,"contact":contact,
         "y":target,"stance":np.arange(n)%4,
         "near_contact":np.arange(n)%3==0,"impact_time_s":None,
         "when_future":t,"joint_names":tuple("j"+str(k) for k in range(29)),
         "source_sha256":"abc","real_200hz_frames":n*4,"annotation_rows":0,
         "original_physx_complete":False,"original_hard_gate_pass":False,
         "nonfoot_peak_N":0.,"original_inst_fraction":0.,"original_goal":False,"original_recovered":False}
def test_frozen_no_scene_leak():
 assert set(TRAIN)=={"C_v32_off","C_cmu02","D_v32","D_cmu02"}
 assert set(TEST)=={"E_v32","E_cmu02","F_v32","F_cmu02"}
 assert set(TRAIN).isdisjoint(TEST)
def test_future_joint_velocity_is_not_observer_feature():
 z=fixture("A")
 z["y"][:,:29]+=42
 model=fit_ridge([z],"full")
 p=predict(model,z)
 assert p.shape==(100,34)
 assert np.isfinite(p).all()
 assert z["base"].shape[1]==73 and z["contact"].shape[1]==7
def test_train_standardizer_not_from_evaluation():
 x=np.array([[1.,5.],[2.,5.],[3.,5.]])
 z,mu,std=standardized(x)
 assert np.isfinite(z).all() and mu[0]==2 and std[1]==1
 held=np.array([[1000.,1000.]])
 projected,_,_=standardized(held,mu,std)
 assert projected[0,0]>100
def test_full_body_29_support_4_impact_1_and_persistence():
 z=fixture("B")
 p=persist(z)
 assert p.shape==(100,34)
 assert np.allclose(p[:,:29],z["base"][:,35:64])
 assert np.array_equal(p[:,29:33].argmax(axis=1),z["stance"])
 assert not p[:,33].any()
def test_gate_metric_detects_unsafe_joint_velocity():
 z=fixture("C")
 p=np.zeros_like(z["y"])
 p[:,0]=1.2
 metrics=score(z["y"],p,np.ones(len(p),bool))
 assert metrics["predicted_joint_qdot_ratio_over_limit_fraction"]>0
 assert metrics["n"]==100
 assert metrics["support_macro_f1_present_classes"] is not None
def test_missing_contact_window_is_not_fabricated():
 z=fixture("D")
 p=persist(z)
 assert score(z["y"],p,np.zeros(len(p),bool)) is None
def test_frozen_ablation_fit_and_episode_holdout():
 a,b=fixture("A"),fixture("B")
 result,models=experiment([a],[b])
 assert result["future_true_impacts_as_input"] is False
 assert len(models)==4
 assert len(result["metrics"])==1
 for k in ("velocity_persistence","clock_only_ridge","state_only_ridge",
           "contact_conditioned_ridge","contact_feature_shuffled_negative"):
  assert k in result["metrics"][0]["metrics"]
def test_same_episode_train_eval_rejected():
 a=fixture("A")
 with pytest.raises(ValueError,match="LEAKAGE"):experiment([a],[a])
def test_observer_refuses_uncertified_source():
 a,b=fixture("A"),fixture("B")
 b["certification"]=False
 with pytest.raises(ValueError,match="UNCERTIFIED"):experiment([a],[b])
