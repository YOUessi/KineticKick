"""v0.1.41: strict archived PhysX results and provenance regression.

The local JSON is ignored in a GitHub clone. Report requires actual original
G1 physics, real joint speed limits, whole-ball goal, 14s and 29DoF policy.
Do not turn known development cases into independent generalization evidence.
"""
from pathlib import Path
import hashlib,json,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"reports/v041_coordination_style_comparison.json"

@pytest.mark.skipif(not DATA.exists(),reason="Exact 14s PhysX records only on Tang")
def test_archived_study_shas_gates_and_no_fresh_task_overclaim():
    study=json.loads((ROOT/"configs/g1_coordination_style_physics_v041.json").read_text())
    report=json.loads(DATA.read_text())
    for f,h in study["source_sha256"].items():
        assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,(f,"drift")
    assert report["zero_pass"]
    assert report["n_evaluated_fixed_conditions"]==3
    assert report["prespecified_conditions_total"]==3
    assert report["all_evaluated_physically_pass"]
    assert report["candidate_full_pass_count"]==3
    assert report["original_goal_count"]==1
    assert report["candidate_goal_count"]==3
    assert report["original_failures_recovered_in_tested_cases"]==2
    assert report["stopping_reason"]=="all_prespecified_conditions_completed"
    assert report["any_unseen_fresh_conditions_used"] is False
    assert report["cross_task_generalization_estimated"] is False
    assert report["not_paper_naturalness_evidence"] is True
    assert report["human_source_is_verified_explosive_instep"] is False

@pytest.mark.skipif(not DATA.exists(),reason="Tang-only physics")
def test_all_old_known_conditions_have_legal_shot_qdot_contact_recovery():
    raw=json.loads(DATA.read_text())
    for row in raw["records"]:
        assert row["new_physics_pass"]
        assert row["candidate_goal"]
        assert row["full_physical_steps"]==2800
        assert row["full_policy_steps"]==700
        assert row["actual_joint_speed_exceedances"]==0
        assert row["actual_joint_speed_ratio_max"]<1.
        assert row["continuous_standing_s"]>=1.
        assert row["new_speed_retention"]>=.95
        assert row["additional_29joint_control_decisions"]>0
        assert row["peak_added_joint_reference_rad"]<=.00601
        assert row["failed_gates"]==[]

@pytest.mark.skipif(not DATA.exists(),reason="Tang-only physics")
def test_nominal_error_gate_is_independent_of_style_prior():
    raw=json.loads(DATA.read_text())
    center=next(x for x in raw["records"] if x["case"]=="center")
    assert center["style_despite_zero_old_error_on_center"]>0
    for other in raw["records"]:
        if other["case"]!="center":
            assert other["style_despite_zero_old_error_on_center"] is None
            assert other["legacy_error_gate_criterion_applicable"] is False
    invalid=json.loads((ROOT/"reports/v041_human_style_near_slow_17_audit_initial_inapplicable_gate.json").read_text())
    corrected=json.loads((ROOT/"reports/v041_human_style_near_slow_17_audit.json").read_text())
    assert invalid["passed"] is False
    assert invalid["failed_gates"]==["new_style_active_despite_zero_original_error"]
    assert corrected["passed"] and not corrected["failed_gates"]
    assert corrected["criteria"]["new_style_active_despite_zero_original_error"] is None

@pytest.mark.skipif(not DATA.exists(),reason="Tang-only physics")
def test_source_3dsp_is_training_broadcast_and_morphology_is_not_proven():
    report=json.loads(DATA.read_text())
    assert report["the_only_human_pose_prior"].startswith("3DSP")
    assert not report["model_is_learned_human_conditional_policy"]
    assert report["coordination_groundtruth_not_calibrated"]
    c=json.loads((ROOT/"configs/g1_coordination_style_physics_v041.json").read_text())
    assert not c["human_source_qualified_professional_instep"]
    assert not c["actual_human_impact_time_available"]
    assert c["one_human_training_clip_used"]=="00001"

@pytest.mark.skipif(not DATA.exists(),reason="Tang-only physics")
def test_actual_body_coordination_effect_is_finite_but_tiny():
    mapping={
       "center":"reports/v041_human_style_center_physical_coordination.json",
       "heldout_v028_near_slow_17":"reports/v041_human_style_near_slow_17_effect.json",
       "heldout_v028_left_near_slow":"reports/v041_human_style_left_near_slow_effect.json"}
    for name,p in mapping.items():
        r=json.loads((ROOT/p).read_text())
        assert r["case"]==name
        assert r["human_impact_event_verified"] is False
        assert r["robot_impact_event_verified"] is True
        assert 0.<r["relative_proxy_error_reduction"]<.02
        assert 0.<r["mean_absolute_robot_relation_change"]<.005
        assert r["full_joint_actual_difference_peak_rad"]>0.
