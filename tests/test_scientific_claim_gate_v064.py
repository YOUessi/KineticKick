"""Scientific-claim regression against ball-only and null-controller false positives."""
from __future__ import annotations
import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from scientific_claim_gate_v064 import classify_pair, assess


def pair(name, mode="human_ordered"):
    return {"context": name, "policy": mode, "valid_real_PhysX_pair": True,
            "preaction_parity": {"matches_at_abs_tol_1e_6": True},
            "full_25_field_200hz_physics_parity_vs_original": {
                "matches_at_abs_tol_1e_6": True, "max_abs_numeric_difference": 0},
            "candidate_20ms_actual_steps": 0,
            "candidate_actual_max_command_delta_rad": 0.0,
            "actual_measured_g1_29dof_joint_position_difference_max_rad": 0.0,
            "original_strict_pass": True, "candidate_strict_pass": True}


def fixture():
    return {"version": "v0.1.64", "actual_original_G1_PhysX_attempts": 12,
            "actual_original_PhysX_all_gate_pass": 12,
            "all_episodes": [{"id": str(i)} for i in range(12)],
            "preregistered_pair_results": [pair(g, p) for g in
                ("center_seen", "unseen_6407", "unseen_6419", "unseen_6431")
                for p in ("human_ordered", "human_reversed")],
            "professional_high_power_true_foot_ball_contact": False,
            "measured_human_ground_reaction_force": False,
            "independent_human_subject_test": False}


def test_all_goal_passes_never_certify_inert_human_model():
    result = assess(fixture())
    assert result["original_task_passes"] == 12
    assert result["inert_pairs"] == 8
    assert result["candidate_effect_pairs"] == 0
    assert result["scientific_activation_gate"] == "REJECT_ALL_CANDIDATES_INERT"
    assert not result["professional_kinetic_claim_allowed"]
    assert not result["naturalness_claim_allowed"]


def test_candidate_moved_without_issued_action_cannot_be_claimed():
    x = pair("center_seen")
    x["actual_measured_g1_29dof_joint_position_difference_max_rad"] = 0.0005
    assert classify_pair(x) == "INCONSISTENT_INERT_CANDIDATE_STATE_CHANGED"


def test_reference_only_without_actual_joint_motion_is_not_effect():
    x = pair("center_seen")
    x["candidate_20ms_actual_steps"] = 3
    x["candidate_actual_max_command_delta_rad"] = .01
    assert classify_pair(x) == "UNVERIFIED_ACTUATION_NO_OBSERVABLE_CAUSAL_MOTION"


def test_actual_displacement_requires_original_same_seed_state_parity():
    x = pair("center_seen")
    x["candidate_20ms_actual_steps"] = 3
    x["candidate_actual_max_command_delta_rad"] = .01
    x["actual_measured_g1_29dof_joint_position_difference_max_rad"] = .003
    x["full_25_field_200hz_physics_parity_vs_original"] = {
        "matches_at_abs_tol_1e_6": False, "max_abs_numeric_difference": 0.003}
    assert classify_pair(x) == "ROBOT_MOTION_EFFECT_OBSERVED_NOT_NATURALNESS"
    x["preaction_parity"]["matches_at_abs_tol_1e_6"] = False
    assert classify_pair(x) == "UNSCORABLE_PREACTION_STATE_DRIFT"


def test_missing_or_repeated_physics_is_not_twelve_independent_trials():
    x = fixture()
    x["all_episodes"].pop()
    with pytest.raises(ValueError, match="complete"):
        assess(x)
    y = fixture()
    y["preregistered_pair_results"][1] = copy.deepcopy(y["preregistered_pair_results"][0])
    with pytest.raises(ValueError, match="complete"):
        assess(y)


def test_uncertain_contact_and_safety_cannot_license_professional_claims():
    x = fixture()
    x["professional_high_power_true_foot_ball_contact"] = False
    x["measured_human_ground_reaction_force"] = True
    x["independent_human_subject_test"] = True
    result = assess(x)
    assert not result["professional_kinetic_claim_allowed"]
    assert not result["naturalness_claim_allowed"]
