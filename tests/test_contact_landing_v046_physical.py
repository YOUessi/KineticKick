"""v46 real G1/PhysX landing-transfer validation and original source pins."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_contact_landing_v046 import compare_zero
STUDY=ROOT/"configs/landing_transfer_study_v046.json"
ZERO=ROOT/"reports/v046_zero_original_center_audit.json"
FINAL=ROOT/"reports/v046_landing_transfer_final_audit.json"

def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

def test_frozen_physics_source_and_new_unique_exact_parameter_conditions():
    study=json.loads(STUDY.read_text())
    assert study["version"]=="0.1.46"
    assert study["total_prespecified_physx_runs"]==15
    assert study["original_foot_ball_contact_and_actuator_limits_unchanged"]
    assert study["no_gain_fitting_after_physical_results"]
    assert study["rejected_v044_forecaster_NOT_used"]
    assert study["rejected_v045_non_engagement_controller_NOT_combined"]
    assert not study["no_new_policy_PPO_training"] is False
    assert len(study["six_new_ball_conditions"])==6
    for path,digest in study["source_sha256"].items():
        assert sha(path)==digest,path
    new={(c["ball_offset_xy_m"][0],c["ball_offset_xy_m"][1],c["velocity_y_delta_mps"])
         for c in study["six_new_ball_conditions"]}
    known={(c["ball_offset_xy_m"][0],c["ball_offset_xy_m"][1],c["velocity_y_delta_mps"])
         for c in study["old_diagnostics"]}
    old=json.loads((ROOT/"configs/novel_ball_generalization_v042.json").read_text())
    used={(c["ball_offset_xy_m"][0],c["ball_offset_xy_m"][1],c["velocity_y_delta_mps"]) for c in old["trials"]}
    assert len(new)==6 and not (new&known) and not(new&used)
    for a,b,c in new:
        assert abs(a)<=.15 and abs(b)<=.15 and abs(c)<=.3

@pytest.mark.skipif(not ZERO.exists(),reason="Need raw original disabled-landing G1 14s PhysX on Tang")
def test_disabled_governor_protects_original_28_physics_fields():
    cfg=json.loads(STUDY.read_text())
    saved=json.loads(ZERO.read_text())
    assert saved["zero_parity"]["shared_physical_fields_checked"]==28
    assert saved["zero_parity"]["original_action_stream_identical"]
    assert saved["zero_parity"]["all_shared_original_physics_identical"]
    assert saved["zero_parity"]["zero_additional_actuator_reference_steps"]==0
    assert saved["zero_parity"]["full_json_byte_identical"] is False
    # Inspect all original 2800x5ms samples again instead of trusting summary.
    calc=compare_zero(cfg)
    assert calc==saved["zero_parity"]

@pytest.mark.skipif(not FINAL.exists(),reason="Frozen 15 original G1 physical episodes incomplete")
def test_paired_original_physics_has_valid_residuals_and_true_qdot_gate():
    r=json.loads(FINAL.read_text())
    assert len(r["results"])==6 and len(r["old_diagnostics"])==2
    assert r["zero_parity"]["all_shared_original_physics_identical"]
    assert r["existing_default_v5_not_replaced"]
    assert r["new_professional_biomechanics_data"] is False
    assert r["has_robot_hardware_validation"] is False
    assert r["new_landing_controller_not_certified_safe"]
    assert r["once_only_each_condition"]
    assert r["fixed_original_v41_full_pass_count"]==sum(
        c["original_v41"]["original_strict_pass"] for c in r["results"])
    assert r["new_landing_v46_full_pass_count"]==sum(
        c["landing_v46"]["original_strict_pass"] for c in r["results"])
    for c in r["results"]:
        for k in ("original_v41","landing_v46"):
            v=c[k]
            assert v["source_sha256"]==sha(v["source_path"])
            assert v["original_strict_pass"]==all(v["original_task_gates"].values())
            assert v["actual_physics_steps"]>0
            assert v["original_qdot_excess_samples"]>=0
        landing=c["landing_v46"]
        assert landing["peak_landing_joint_position_target_delta_rad"]<=.016021
        assert landing["peak_landing_joint_target_slew_per20ms_rad"]<=.002021
        assert not landing["landing_command_contract_errors"]
        if landing["landing_actual_nonzero_control_steps"]>0:
            assert landing["first_landing_delta_time_s"]>=landing["first_ball_contact_s"]
    assert r["new_landing_actual_nonzero_case_count"]==sum(
        c["landing_v46"]["landing_actual_nonzero_control_steps"]>0 for c in r["results"])

@pytest.mark.skipif(not FINAL.exists(),reason="Need independent physical strict scoring")
def test_no_new_style_promotion_from_single_physx_without_qualified_human_data():
    r=json.loads(FINAL.read_text())
    assert not r["new_landing_controller_not_certified_safe"] is False
    assert r["existing_default_v5_not_replaced"]
    assert r["no_success_filtering"]
    assert r["new_exact_ball_parameters_not_new_physics_domain"]
    assert len(r["results"])==6
