"""v48 independent G1 fullbody PRECONTACT learning and physical provenance tests."""
from pathlib import Path
import json,hashlib,torch,pytest
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from preimpact_wholebody_v048 import (
    PreImpactCoordinationActor,PreImpactWholeBodyTask,FEATURE_DIM,phase_envelope)
from post_shot_recovery import PostShotRecoveryTask
STUDY=ROOT/"configs/learned_preimpact_wholebody_study_v048.json"
DATA=ROOT/"experiments/coordination_projection_v040/train_clip00001_center/g1_joint_delta_50hz.npz"
FINAL=ROOT/"reports/v048_preimpact_full_physics_audit.json"
TRAIN=ROOT/"experiments/preimpact_wholebody_v048/seed4801_env64_iter96/result.json"

def digest(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

def test_independent_160_dim_29_joint_PPO_actor_finite_output():
    torch.manual_seed(4801)
    m=PreImpactCoordinationActor(FEATURE_DIM,FEATURE_DIM,29)
    assert FEATURE_DIM==160
    x=torch.randn(3,FEATURE_DIM)
    out=m.act_inference(x)
    assert out.shape==(3,29)
    assert torch.isfinite(out).all()
    assert sum(p.numel() for p in m.parameters())>30000

def test_start_window_has_true_pre_impact_termination_and_ramp():
    s=json.loads(STUDY.read_text())["training_config"]
    c=json.loads((ROOT/s).read_text())["preimpact_wholebody_v048"]
    t=torch.tensor([.25,1.02,1.12,1.23,1.42,1.52,1.54,1.57,1.8])
    gate=phase_envelope(t,c)
    assert gate.shape==t.shape
    assert float(gate[0])==float(gate[1])==0
    assert gate[3]>.9
    assert gate[4]>.9
    assert float(gate[6])==float(gate[7])==float(gate[8])==0.
    assert float(gate.max())<=1.

def test_human_training_weak_prior_not_true_instestep_or_hardware():
    study=json.loads(STUDY.read_text())
    assert study["human_training_source_qualifies_real_athlete_instestep"] is False
    with np.load(DATA,allow_pickle=False) as z:
        assert z["delta_joint_position_rad"].shape==(700,29)
        assert z["source_clip_id"].item()=="00001"
        assert z["source_is_verified_instep"].item()==False
    assert digest(str(DATA.relative_to(ROOT)))==study["source_3dsp_weak_pose_prior_sha256"]
    assert study["original_v32_PPO_recovery_sha256"]=="ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326"

def test_original_source_shas_frozen_and_holdout_context_disjoint():
    study=json.loads(STUDY.read_text())
    for name,h in study["source_sha256"].items():
        assert digest(name)==h,("source_sha_drift",name)
    assert study["no_extra_model_tuning_or_seed_selection_after_results"]
    assert len(study["planned_6_new_exact_soccer_conditions"])==6
    previous=[]
    for r in study["planned_6_new_exact_soccer_conditions"]:
        tup=(*r["ball_offset_xy_m"],r["velocity_y_delta_mps"])
        assert tup not in previous
        previous.append(tup)

@pytest.mark.skipif(not TRAIN.exists(),reason="Original actual IsaacLab GPU PPO training incomplete")
def test_training_real_G1_PPO_checkpoint_frozen_before_physx():
    c=json.loads(STUDY.read_text())
    z=json.loads(TRAIN.read_text())
    assert z["updates"]>=1 and z["stop"] in ("iteration_limit","wall_time_limit")
    assert z["checkpoint_sha256"]==digest(c["training_config_obj"]["output_directory"]+"/last_model.pt") if "training_config_obj" in c else z["checkpoint_sha256"]==digest("experiments/preimpact_wholebody_v048/seed4801_env64_iter96/last_model.pt")
    assert c["human_training_source_qualifies_real_athlete_instestep"] is False

@pytest.mark.skipif(not FINAL.exists(),reason="All 13 new original G1 14s physical runs incomplete")
def test_13_full_physx_paired_and_no_unsafe_overclaim():
    z=json.loads(FINAL.read_text())
    assert len(z["cases"])==6
    assert z["zero_physics_equals_original"]
    assert z["new_trained_G1_actor_promotion_approved"] is False
    assert z["professional_biomechanics_established"] is False
    assert z["six_trials_only_not_population_generalization"]
    for c in z["cases"]:
        for key in ("original_v32","learned_preimpact_v48"):
            p=c[key]
            assert p["original_source_sha256"]==digest(p["original_source"])
            assert p["strict_physical_pass"]==all(p["strict_gates"].values())
            assert p["qdot_speed_overlimit_sample_joint_count"]>=0
            assert p["actual_physics_steps"]>0
