"""v47 true G1 fullbody PPO transfer / original PhysX evaluation consistency.

All raw physically observed 14s output is only on Tang. Public clones skip
raw-only tests, never treat one static human video proxy as professional
explosive motion or 6 original physics conditions as population reliability.
"""
import hashlib,json,sys
from pathlib import Path
import torch,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from post_shot_objective_v047 import FIXED

STUDY=ROOT/"configs/learned_contact_recovery_study_v047.json"
FINAL=ROOT/"reports/v047_independent_original_physx_audit.json"
SOURCE=ROOT/"experiments/contact_phase_recovery_v047/seed4701_env64_iter72/last_model.pt"
RESULT=ROOT/"experiments/contact_phase_recovery_v047/seed4701_env64_iter72/result.json"

def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

def test_registered_training_data_and_original_g1_limits_not_changed():
    s=json.loads(STUDY.read_text())
    for path,expected in s["source_sha256"].items():
        assert sha(path)==expected,(path,"SOURCE_DRIFT")
    assert s["training"]["seed"]==4701
    assert s["training"]["num_envs"]==64
    assert s["training"]["iterations"]==72
    assert s["no_hyperparameter_sweep"]
    assert s["no_multi_checkpoint_selection"]
    assert s["original_29dof_floating_G1_real_PhysX"]
    assert s["original_joint_actual_velocity_and_torque_limits_unchanged"]
    assert s["human_qualified_professional_explosive_data_used"] is False
    assert s["frozen_recovery_actor_start_SHA256"]=="ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326"
    assert FIXED["warning_ratio"]==.70

@pytest.mark.skipif(not RESULT.exists() or not SOURCE.exists(),reason="Real PPO training only on Tang")
def test_real_72_ppo_updates_sha_finite_checkpoint_and_no_test_fitting():
    s=json.loads(STUDY.read_text())
    result=json.loads(RESULT.read_text())
    assert result["updates"]==72 and result["stop"]=="iteration_limit"
    assert result["checkpoint_sha256"]==sha(str(SOURCE.relative_to(ROOT)))
    assert s["actual_final_trained_checkpoint_sha256"]==sha(str(SOURCE.relative_to(ROOT)))
    model=torch.load(SOURCE,map_location="cpu",weights_only=False)
    # Original RSL-RL runner stores the zero-indexed last completed iteration.
    assert model["iter"]==71
    assert all(torch.isfinite(v).all() for v in model["model_state_dict"].values())
    prior=torch.load(ROOT/"experiments/post_shot_v032/ppo_s3201/last_model.pt",
                     map_location="cpu",weights_only=False)
    shared=set(prior["model_state_dict"])&set(model["model_state_dict"])
    assert len(shared)==17
    assert any(not torch.equal(prior["model_state_dict"][k],model["model_state_dict"][k]) for k in shared)
    assert s["frozen_after_completed_training_before_any_new_physx"]["test_shots_executed_before_freeze"] is False
    assert s["frozen_after_completed_training_before_any_new_physx"]["training_update_count"]==72

def test_no_new_ball_triplet_reuses_v42_v45_v46_or_original_training():
    s=json.loads(STUDY.read_text())
    triplets=[tuple([*r["ball_offset_xy_m"],r["velocity_y_delta_mps"]])
              for r in s["predeclared_eval_exact_six_ball_conditions"]]
    assert len(set(triplets))==6
    for fname,key in (
      ("configs/novel_ball_generalization_v042.json","trials"),
      ("configs/contact_recovery_governor_v045.json","six_new_parameter_conditions"),
      ("configs/landing_transfer_study_v046.json","six_new_ball_conditions")):
      old=json.loads((ROOT/fname).read_text())
      prior={(*r["ball_offset_xy_m"],r["velocity_y_delta_mps"]) for r in old[key]}
      assert not set(triplets)&prior

@pytest.mark.skipif(not FINAL.exists(),reason="Six paired original 14s G1 PhysX evaluations incomplete")
def test_paired_12_real_physx_require_original_contact_speed_limit_and_recovery():
    x=json.loads(FINAL.read_text())
    assert x["version"]=="0.1.47"
    assert x["actual_full14s_new_test_conditions"]==6
    assert len(x["records"])==6
    assert x["no_test_checkpoint_selection"]
    assert x["original_joint_vel_and_torque_limits_unchanged"]
    assert x["original_foot_geometry_and_soccer_ball_unchanged"]
    assert x["only_six_once_each_not_robust_population_statistics"]
    assert x["model_not_deployed"]
    for case in x["records"]:
      for key in ("old","learned"):
        r=case[key]
        assert sha(r["original_source"])==r["original_source_sha256"]
        assert r["strict_original_full_task_pass"]==all(r["original_physical_gates"].values())
        assert r["original_qdot_speed_exceedance_sample_joints"]>=0
        assert r["physical_steps"]>0
        assert r["first_actual_ball_contact_s"] is None or r["first_actual_ball_contact_s"]>=0.
        assert r["actual_qdot_peak_over_original_limit_ratio"]>=0
    assert sum(x["old"]["strict_original_full_task_pass"] for x in x["records"])==x["original_v32_strict_success"]
    assert sum(x["learned"]["strict_original_full_task_pass"] for x in x["records"])==x["new_finetuned_recovery_strict_success"]

@pytest.mark.skipif(not FINAL.exists(),reason="No original physical final evaluation")
def test_frozen_complete_six_case_selection_and_no_hardware_claim():
    z=json.loads(FINAL.read_text())
    assert z["training_not_human_kick_policy"]
    assert z["model_not_deployed"]
    assert z["no_test_checkpoint_selection"]
    assert z["PPO_warmstart_original_sha256"]=="ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326"
    assert len(set(tuple(r["exact_parameter_tuple"]) for r in z["records"]))==6
