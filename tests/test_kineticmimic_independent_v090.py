"""KineticMimic v90 independent original PhysX, optical evidence and hard gates."""
from pathlib import Path
import json,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from kineticmimic_physics_v090 import sha,strict_physics,load_original_physics

def audit():
 f=ROOT/"reports/v090_kineticmimic_independent_audit.json"
 if not f.exists():pytest.skip("Requires Tang original 200Hz PhysX artifact audit")
 return json.loads(f.read_text())
def training():
 p=ROOT/"experiments/kineticmimic_v090_contact_head/kineticmimic_contact_training_audit.json"
 if not p.exists():pytest.skip("Requires Tang-only original PhysX contact head")
 return json.loads(p.read_text())

def test_precommitted_method_source_registry_all_sha_match():
 p=ROOT/"configs/kineticmimic_v090_study.json"
 c=json.loads(p.read_text())
 assert c["version"]=="v0.1.90"
 for rel,digest in c["source_sha256"].items():
  assert sha(ROOT/rel)==digest
 for meta in c["source_physx"].values():
  assert sha(Path(meta["path"]))==meta["sha256"]
 assert len(c["source_physx"])==9
 assert not c["train_on_center_scene"]
 assert not c["human_true_impact_available"]
 assert not c["model_is_29dof_actor"]

def test_original_measured_physics_full_gates_and_censor():
 a=audit();phys=a["original_G1_CUDA_200hz_strict_physics_per_source"]
 assert len(phys)==9
 assert a["historical_strict_pass_among_9"]==8
 assert a["real_new_PhysX_episodes"]==0
 bad=phys["center_cmu05"]
 assert bad["independent_all_gates_pass"] is False
 assert bad["original_200hz_true_steps"]==1028
 assert bad["nonphysics_terminal_annotation"]==1
 assert bad["original_recorded_14s_full"] is False
 assert bad["real_foot_ball_contact_time_s"] is not None
 assert bad["original_measured_instep_impulse_fraction"]>=.95
 assert "original_physics_full_14s" in bad["gate_details"] and not bad["gate_details"]["original_physics_full_14s"]

def test_real_contact_predictor_misses_first_impact_and_not_promoted():
 a=audit();x=training()
 assert x["model_has_NO_action_output"]
 assert x["real_G1_PhysX_new_rollouts"]==0
 assert a["no_model_reliably_predicts_contact_on_center"] is True
 for name,v in a["contact_head_validation_and_center_independent_reloads"].items():
  s=v["recomputed_scores"]
  assert s["n_windows"]==228
  assert s["hazard_next_100ms_F1"]["f1"]==0.
  assert s["hazard_next_100ms_F1"]["fn"]==15
  assert s["contact_positive_windows"]==15
  assert s["predicted_stance_accuracy"]>=.90
  assert v["actually_generated_a_29dof_controller"] is False
 assert x["nonlearned_test_baselines"]["contact_never_predicted_F1"]["f1"]==0
 assert x["nonlearned_test_baselines"]["constant_training_scene_impact_delta_v_mae_mps"]<1.2

def test_human_optical_geometry_tiny_better_but_real_nonfoot_worse():
 a=audit();t=a["CMU_human_optical_six_relative_geometry"]
 assert not a["human_foot_ball_true_contact_or_GRF_truth_available"]
 assert t["center_v32"]["human10_02_relative_geometry_MSE"]>t["center_cmu05"]["human10_02_relative_geometry_MSE"]
 assert a["original_G1_CUDA_200hz_strict_physics_per_source"]["center_v32"]["independent_all_gates_pass"]
 assert not a["original_G1_CUDA_200hz_strict_physics_per_source"]["center_cmu05"]["independent_all_gates_pass"]
 assert not a["full_body_generated_29dof_real_physics_tested"]

def test_real_human_geometry_source_two_clips_same_one_subject_not_elite():
 p=ROOT/"configs/kineticmimic_style_v090.json"
 c=json.loads(p.read_text())
 assert not c["human_impact_truth_available"] and not c["human_GRf_truth_available"]
 assert c["human_subjects"]==1
 assert c["humans"]["10_02"]["source_subject"]==c["humans"]["10_05"]["source_subject"]==10
 assert set(c["robots"])=={"center_v32","center_cmu02","center_cmu05"}
 assert c["protocol_requires_no_G1_GPU"] is True

def test_not_all_positive_surrogate_implies_fatal_nonfoot_pass():
 a=audit();z=a["original_G1_CUDA_200hz_strict_physics_per_source"]
 assert z["center_cmu05"]["original_measured_instep_impulse_fraction"]==1.
 assert z["center_cmu05"]["ball_speed_5_to_105ms_mps"]>3.5
 assert z["center_cmu05"]["nonfoot_max_net_force_N"]>20
 assert z["center_cmu05"]["independent_all_gates_pass"] is False

def test_local_original_figures_are_valid_shas_only():
 p=ROOT/"reports/v090_figures_audit.json"
 if not p.exists():pytest.skip("Only Tang local original 200Hz evidence figures")
 v=json.loads(p.read_text())
 assert len(v["plots"])==3 and v["new_PhysX"]==0
 for r in v["plots"]:
  assert sha(ROOT/r["file"])==r["SHA256"]
