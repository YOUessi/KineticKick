"""v44 trained/contact-aware physical qdot forecaster and negative safety audit."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from contact_risk_model_v044 import ActionConditionedRiskMLP,conditional_risk_score,risk_metrics
from build_contact_future_v044 import feature_from_sample

STUDY=ROOT/"configs/contact_future_forecast_v044.json"
RESULT=ROOT/"experiments/contact_future_v044/seed4401_group_trial_holdout/result.json"
DATA=ROOT/"data/research_human_kicks/contact_future_v044/planned_action_future_qdot_v044.npz"

def test_67d_original_robot_state_and_precommitted_future_action():
    q=np.zeros(29,dtype=np.float32)
    v=np.full(29,.5,dtype=np.float32)
    limits=np.full(29,20.,dtype=np.float32)
    loads=[100.,0.]
    pos=[.6,-.4]
    ball=[0.,-.2,0.]
    base=conditional_risk_score(q,v,limits,loads,pos,ball,1.3,False)
    style=conditional_risk_score(q,v,limits,loads,pos,ball,1.3,True)
    assert base.shape==(67,) and style.shape==(67,)
    np.testing.assert_array_equal(base[:66],style[:66])
    assert style[-1]==1 and base[-1]==0
    assert np.isclose(base[29],.025)
    with pytest.raises(ValueError):
        conditional_risk_score(q,v,np.zeros(29),loads,pos,ball,1.3,True)

def test_safe_tensor_model_forward_and_training_gradient():
    torch.manual_seed(4401)
    model=ActionConditionedRiskMLP()
    x=torch.zeros(4,67)
    xm=torch.zeros(67);xs=torch.ones(67)
    ym=torch.zeros(29);ys=torch.ones(29)
    out=model(x,xm,xs,ym,ys)
    assert out.shape==(4,29)
    assert torch.isfinite(out).all() and (out>0).all()
    out.mean().backward()
    assert any(p.grad is not None for p in model.parameters())
    with pytest.raises(ValueError):
        model(torch.zeros(4,66),xm,xs,ym,ys)

def test_29_joint_risk_metrics_validate_input():
    z=np.ones((2,29))
    x=risk_metrics(z,z)
    assert x["mae"]==0. and x["mse"]==0.
    assert x["all_future_unsafe_window_count"]==0
    with pytest.raises(ValueError):risk_metrics(np.ones((2,28)),z)

@pytest.mark.skipif(not DATA.exists(),reason="Pinned original 12 G1 PhysX source traces only on Tang")
def test_original_source_groups_do_not_mix_144_correlated_windows():
    cfg=json.loads(STUDY.read_text())
    meta=json.loads((ROOT/"data/research_human_kicks/contact_future_v044/build_manifest.json").read_text())
    assert cfg["model_uses_future_physics_at_inference"] is False
    assert cfg["source_group_split"] if "source_group_split" in cfg else cfg["group_split"]
    for path,sha in cfg["source_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==cfg["original_dataset_sha256"]
    assert meta["observation_count"]==144
    assert meta["independent_samples_cannot_be_counted_as_144"]
    with np.load(DATA,allow_pickle=False) as z:
        assert z["features"].shape==(144,67)
        assert z["targets_max_qdot_ratio_per_joint"].shape==(144,29)
        assert int(z["future_any_original_qdot_exceeded"].sum())==60
        x=z["features"];groups=z["episode_id"];splits=z["split"]
        assert set(groups[splits=="train"])==set(cfg["group_split"]["train"])
        assert set(groups[splits=="val"])==set(cfg["group_split"]["val"])
        assert set(groups[splits=="test"])==set(cfg["group_split"]["test"])
        assert set(groups[splits=="train"]).isdisjoint(set(groups[splits=="val"]))
        assert set(groups[splits=="test"]).isdisjoint(set(groups[splits=="train"]))
        assert set(groups[splits=="test"]).isdisjoint(set(groups[splits=="val"]))

@pytest.mark.skipif(not RESULT.exists(),reason="Actual Tang trained risk model missing")
def test_learned_model_is_not_promoted_due_to_unsafe_false_safes():
    result=json.loads(RESULT.read_text())
    assert result["trained_on_retrospective_v42_data"]
    assert result["not_blind_to_prior_v42_research"]
    assert result["completed_updates"]==1000
    assert result["selected_training_step"]==50
    assert result["model_selection_only_by_val_mae"]
    assert result["n_physics_episode_groups"]=={"train":4,"val":1,"test":1}
    assert result["safety_certificate_obtained"] is False
    assert result["model_allowed_to_replace_existing_robot_policy"] is False
    test=result["metrics"]["test"]
    assert test["planned_action_mlp"]["underpredicted_unsafe_window_count"]==12
    assert test["planned_action_mlp"]["all_future_unsafe_window_count"]==12
    assert test["planned_action_mlp"]["mae"]<test["training_only_ridge"]["mae"]
    assert test["training_mean_future"]["mae"]<test["planned_action_mlp"]["mae"]
    assert not result["heldout_triple_test_MLP_beats_train_mean"]

@pytest.mark.skipif(not RESULT.exists(),reason="Actual Tang original source checkpoint missing")
def test_serialization_recovery_keeps_exact_same_pretest_learned_weights():
    cfg=json.loads(STUDY.read_text())
    assert cfg["serialization_only_retry"]["first_test_evaluation_completed_before_fix"] is False
    p=ROOT/"experiments/contact_future_v044/seed4401_first_attempt_serialization_failed/best_validation_model.pt"
    q=ROOT/"experiments/contact_future_v044/seed4401_group_trial_holdout/best_validation_model.pt"
    if not p.exists():pytest.skip("Original no-promotion failed checkpoint not on public Git clone")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        a=torch.load(p,map_location="cpu",weights_only=True)
    b=torch.load(q,map_location="cpu",weights_only=True)
    assert a["trained_steps"]==b["trained_steps"]==50
    for k in a["model_state_dict"]:
        torch.testing.assert_close(a["model_state_dict"][k],b["model_state_dict"][k],atol=0,rtol=0)
    assert type(a["torch_version"]).__name__=="TorchVersion"
    assert isinstance(b["torch_version"],str)
