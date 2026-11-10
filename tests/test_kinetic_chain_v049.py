"""v49 empirical G1 kinetic-chain and original contact physics contracts.

No human biomechanics truth; six known exact ball initial conditions are NOT
a newly blinded experiment. Read-only 12 original Isaac/PhysX traces.
"""
from pathlib import Path
import hashlib,json,sys
import pytest,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from extract_kinetic_chain_v049 import REL,analyze_one
CONFIG=ROOT/"configs/kinetic_chain_observability_v049.json"
META=ROOT/"data/research_human_kicks/kinetic_chain_v049/kinetic_chain_audit_v049.json"
DATA=ROOT/"data/research_human_kicks/kinetic_chain_v049/kinetic_chain_physx_v049.npz"
COMP=ROOT/"data/research_human_kicks/kinetic_chain_v049/paired_kinetic_chain_comparison.json"

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def test_contact_window_has_true_uninterpolated_original_200hz():
    assert REL.shape==(131,)
    assert REL[0]==-.3 and REL[-1]==.35
    np.testing.assert_allclose(np.diff(REL),.005,atol=3e-16,rtol=0)
    assert 0. in REL

def test_frozen_original_12_real_physics_report_source_sha():
    study=json.loads(CONFIG.read_text())
    assert study["version"]=="0.1.49" and study["no_new_PhysX_episodes"]
    assert study["no_model_tuning_or_new_rl_training_on_old_v48_test_states"]
    assert len(study["source_sha256"])==18
    for name,dig in study["source_sha256"].items():
        assert sha(ROOT/name)==dig,name
    assert study["human_expert_action_naturalness_rating_not_collected"]
    assert not study["no_velocity_contribution_as_energy_fraction"] is False

@pytest.mark.skipif(not DATA.exists(),reason="Original complete G1 PhysX only available on Tang")
def test_exact_29_joint_5ms_physical_velocity_and_torque_data():
    source=json.loads(META.read_text())
    assert sha(DATA)==source["v49_physics_npz_sha256"]
    assert source["original_source_episode_count"]==12
    assert source["independent_ball_cases"]==6
    with np.load(DATA,allow_pickle=False) as z:
        assert z["qdot"].shape==(12,131,29)
        assert z["tau"].shape==(12,131,29)
        assert z["original_power_W_estimated"].shape==(12,131,29)
        assert z["ball_velocity"].shape==(12,131,3)
        assert z["left_right_support"].shape==(12,131,2)
        assert z["original_PhysX_data_only"].item()
        assert not z["physical_human_foot_biomechanics_verified"].item()
        combined=z["base_contribution"]+z["hip_contribution"]+z["knee_contribution"]+z["ankle_contribution"]
        np.testing.assert_allclose(combined,z["foot_point_velocity"],rtol=0,atol=4e-6)
        np.testing.assert_allclose(z["qdot"]*z["tau"],z["original_power_W_estimated"],
                                   atol=6e-4,rtol=3e-5)

@pytest.mark.skipif(not COMP.exists(),reason="Real G1/PhysX original comparisons retained on Tang")
def test_original_and_new_foot_contact_velocity_are_physical_not_human_power():
    z=json.loads(COMP.read_text())
    assert z["source_is_valid_human_proximal_to_distal_biomechanics"] is False
    assert z["statistical_population_naturalness_inference_supported"] is False
    assert z["original_fullbody_physical_episodes"]==12
    assert z["original_hip_world_goal_direction_positive_count"]==6
    assert z["original_knee_world_goal_direction_negative_count"]==6
    assert z["original_ankle_world_goal_direction_negative_count"]==6
    assert z["learned_hip_world_goal_direction_positive_count"]==6
    assert z["learned_knee_world_goal_direction_negative_count"]==6
    assert z["learned_ankle_world_goal_direction_negative_count"]==6
    assert z["original_hip_knee_ankle_joint_qdot_peak_nondecreasing_examples"]==0
    assert z["learned_hip_knee_ankle_joint_qdot_peak_nondecreasing_examples"]==0
    assert z["mean_abs_paired_foot_forward_speed_change_mps"]<.03
    assert not z["innovation_claim_allowed"]

@pytest.mark.skipif(not META.exists(),reason="Original full G1 PhysX raw samples missing")
def test_recomputed_source_foot_decomposition_and_first_true_impact_one_case():
    meta=json.loads(META.read_text())
    row=meta["episodes"][0]
    original,metrics,names,limit=analyze_one(row["source"],"configs/state_conditioned_strike_v9.json","original_v32")
    assert row["source_sha256"]==original["source_sha256"]
    assert len(names)==29 and len(limit)==29
    assert original["foot_velocity_world_decomposition_max_numerical_error_mps"]<1e-9
    assert sum(original["original_forward_velocity_from_parts_mps"].values())==pytest.approx(
        original["original_forward_foot_contact_speed_mps"],abs=1e-9)
    assert original["actual_ball_translation_energy_delta_25ms_J"]>0
    assert original["true_original_inst_angle_not_verified"] is True
