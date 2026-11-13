"""v144 independent raw-C3D atan2 and strict original G1 PhysX provenance tests."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v144_human_temporal_source_and_physical_audit as v
import v144_independent_temporal_falsification as independent


@pytest.fixture(scope="module")
def evidence():
    return v.run()


@pytest.fixture(scope="module")
def cert(evidence):
    return independent.independently_verify_report(evidence, v.frozen())


def test_sha_locked_twelve_original_physics_traces_and_v54_real_steps(evidence):
    p = evidence["historical_v53_strict_physics"]
    assert p["historical_real_PhysX_trials_sha_verified"] == 12
    assert p["historical_source_bytes_sha_verified"] > 0
    assert evidence["historical_v54_requalification"]["historical_physics_5ms_frames"] == 6628
    assert evidence["new_real_PhysX_episodes"] == 0


def test_original_strict_gate_matrix_preserves_every_negative(evidence):
    p = evidence["historical_v53_strict_physics"]
    assert set(p["strict_success_by_policy"]) == {"frozen_v32", "cmu_10_02", "cmu_10_05"}
    assert all(x == 0 for x in p["strict_success_by_policy"].values())
    for policy in p["strict_success_by_policy"]:
        assert p["failed_gates_by_policy"][policy]["legal_right_instep_contact"] == 4


def test_independent_atan2_agrees_with_arccos_real_marker_motion(cert):
    # arccos loses a few 1e-10 near zero-angle samples; atan2 is stable.
    assert cert["max_abs_segment_rate_reconstruction_difference_rad_per_s"] < 1e-8


def test_both_same_subject_observed_proximal_distal_timings(cert):
    assert cert["descriptive_proximal_to_distal_right_segment_peak_order_by_clip"] == {
        "10_02": True, "10_05": True,
    }
    assert not any(cert["counterfactual_reverse_pre_window_order_preserved_by_clip"].values())
    assert cert["different_subjects_in_event_cohort"] == 1
    assert cert["no_valid_population_p_value_or_elite_power_conclusion"]


def test_phase_shift_null_is_reproducible_and_not_population_inference(cert):
    assert cert["deterministic_4096_independent_phase_shift_both_order_rate"] == 104 / 4096


def test_synthetic_rate_agrees_between_independent_methods():
    t = np.arange(140) / 120
    v3 = np.column_stack((np.cos(5 * t), np.sin(5 * t), np.zeros_like(t)))
    a, _ = v.observed_rate(v3)
    b = independent.independent_rate(v3)
    assert np.allclose(a[2:-2], b[2:-2], atol=1e-10)


@pytest.mark.parametrize("key,bad", [
    ("new_real_PhysX_episodes", 1),
    ("new_motor_training_or_actuation", 1),
    ("strict_physical_shot_success_claimed", True),
    ("professional_human_power_chain_learned", True),
])
def test_forged_live_training_and_success_fails_closed(evidence, key, bad):
    forged = copy.deepcopy(evidence)
    forged[key] = bad
    with pytest.raises(ValueError):
        independent.independently_verify_report(forged, v.frozen())


def test_forged_subject_holdout_rejected(evidence):
    bad = copy.deepcopy(evidence)
    bad["human_real_C3D_marker_temporal_pilot"][
        "subject_disjoint_temporal_event_model_evaluation_eligible"] = True
    with pytest.raises(ValueError, match="HELDOUT"):
        independent.independently_verify_report(bad, v.frozen())


def test_hidden_g1_strict_success_rejected(evidence):
    bad = copy.deepcopy(evidence)
    bad["historical_v53_strict_physics"]["strict_success_by_policy"]["cmu_10_05"] = 1
    with pytest.raises(ValueError, match="FAKE_PHYSICAL_SUCCESS"):
        independent.independently_verify_report(bad, v.frozen())


def test_missing_a_human_clip_rejected(evidence):
    bad = copy.deepcopy(evidence)
    bad["human_real_C3D_marker_temporal_pilot"]["per_clip"].pop("10_01")
    with pytest.raises(ValueError, match="DENOMINATOR"):
        independent.independently_verify_report(bad, v.frozen())


def test_invented_human_contact_or_grf_rejected(evidence):
    for field in ("measured_human_impact", "measured_human_GRF",
                  "professional_footballer_certified"):
        bad = copy.deepcopy(evidence)
        bad["human_real_C3D_marker_temporal_pilot"]["per_clip"]["10_02"][field] = True
        with pytest.raises(ValueError, match="FALSE_HUMAN"):
            independent.independently_verify_report(bad, v.frozen())


def test_unanchored_segment_peak_hallucination_rejected(evidence):
    bad = copy.deepcopy(evidence)
    bad["human_real_C3D_marker_temporal_pilot"]["per_clip"]["11_01"][
        "segments"]["pelvis"]["pre_ball_marker_launch_descriptor"] = {
            "peak_pre_ball_marker_onset_s": 0.01,
        }
    with pytest.raises(ValueError, match="UNQUALIFIED_EVENT"):
        independent.independently_verify_report(bad, v.frozen())


def test_time_and_peak_tampering_rejected(evidence):
    bad = copy.deepcopy(evidence)
    bad["human_real_C3D_marker_temporal_pilot"]["per_clip"]["10_02"][
        "segments"]["right_shank"]["pre_ball_marker_launch_descriptor"][
        "peak_pre_ball_marker_onset_s"] = 0.5
    with pytest.raises(ValueError, match="MISMATCH"):
        independent.independently_verify_report(bad, v.frozen())


def test_native_contact_pair_not_fabricated(evidence):
    assert evidence["native_original_G1_physx_pair_manifold_truth"] == "UNKNOWN"
    assert not evidence["professional_human_power_chain_learned"]
    assert not evidence["strict_physical_shot_success_claimed"]
