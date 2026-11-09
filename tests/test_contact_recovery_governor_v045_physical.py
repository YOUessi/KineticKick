"""v45 registered whole-body post-contact governor vs frozen original style in actual PhysX.

Entire test class requires original ignored 2800x5ms Tang physics files. No goal
means pass unless every original motion/contact/qdot/recovery gate passed.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from contact_recovery_governor_v045 import braking_reference
from audit_contact_recovery_v045 import compare_zero

STUDY=ROOT/"configs/contact_recovery_governor_v045.json"
ZERO=ROOT/"reports/v045_governor_zero_center_audit.json"
FINAL=ROOT/"reports/v045_governor_final_independent_audit.json"

def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def test_study_preserves_original_physics_actuator_and_model():
    p=json.loads(STUDY.read_text())
    assert p["no_ball_or_robot_state_injections"] and p["original_floating_root_physx"]
    assert p["original_29dof_PD_and_joint_speed_torque_limits_unchanged"]
    assert p["v44_future_risk_model_used_in_controller"] is False
    assert p["v44_model_rejected_for_unsafe_false_safes"] is True
    assert p["no_controller_PPO_retraining"] and p["no_gain_tuning_after_observing_cases"]
    assert p["total_prespecified_physx_runs"]==15
    assert len(p["six_new_parameter_conditions"])==6
    assert len(p["known_development_diagnostics"])==2
    assert len(p["source_sha256"])>=18
    for src,dig in p["source_sha256"].items():
        assert sha(src)==dig,src

def test_six_exact_new_ball_cases_differ_from_prior_test_conditions():
    p=json.loads(STUDY.read_text())
    old=json.loads((ROOT/"configs/novel_ball_generalization_v042.json").read_text())
    new={(*c["ball_offset_xy_m"],c["velocity_y_delta_mps"])
         for c in p["six_new_parameter_conditions"]}
    prior={(*c["ball_offset_xy_m"],c["velocity_y_delta_mps"])
           for c in old["trials"]}
    assert len(new)==6 and not(new&prior)
    for triple in new:
        assert max(abs(triple[0]),abs(triple[1]))<.15
        assert abs(triple[2])<.30

@pytest.mark.skipif(not ZERO.exists(),reason="Tang-original 14s zero-physics data unavailable")
def test_zero_governor_has_exact_original_shared_physics_and_actions():
    study=json.loads(STUDY.read_text())
    zero=json.loads(ZERO.read_text())["zero_exact"]
    assert zero["original_actions_identical"]
    assert zero["original_shared_28_physical_channels_identical_every_5ms"]
    assert zero["zero_added_governor_control_steps"]==0
    assert zero["entire_json_byte_identical"] is False
    assert len(zero["original_instrumented_only_7_channels_absent_in_new_core_runner"])==7
    # Check actual 2800-sample physics once, not merely logged booleans.
    equivalent=compare_zero(study)
    assert equivalent["original_actions_identical"] and equivalent["original_shared_28_physical_channels_identical_every_5ms"]

@pytest.mark.skipif(not FINAL.exists(),reason="All prespecified v45 new PhysX trials not yet available")
def test_original_g1_strict_gates_and_actual_governor_effect_across_all_cases():
    r=json.loads(FINAL.read_text())
    assert r["original_V44_unreliable_forecaster_not_used"]
    assert r["physical_data_is_six_once_each"]
    assert len(r["results"])==6
    assert r["fixed_new_physical_condition_count"]==6
    assert len(r["known_old_development_diagnostics"])==2
    assert r["zero_exact"]["original_actions_identical"]
    assert r["default_robot_controller_not_replaced"]
    assert r["population_robustness_established"] is False
    for row in r["results"]:
        for control in ("original_style","contact_governor"):
            v=row[control]
            assert v["source_sha256"]==sha(v["path"])
            assert v["original_strict_pass"]==all(v["original_g1_task_gates"].values())
            assert v["actual_physx_samples"]>0
            assert v["real_joint_velocity_overlimit_sample_joints"]>=0
            assert v["real_joint_velocity_max_ratio"]>=0
            assert v["real_outgoing_ball_speed_xy_mps"] is None or v["real_outgoing_ball_speed_xy_mps"]>=0
        gov=row["contact_governor"]
        assert gov["governor_max_reference_rad"]<=.02501
        assert gov["governor_max_joint_reference_change_rad"]<=.00403
        assert not gov["governor_source_errors"]
        if gov["actual_governor_nonzero_steps"]>0:
            assert gov["governor_first_nonzero_s"]>=gov["first_contact_s"]

@pytest.mark.skipif(not FINAL.exists(),reason="No full-v45 PhysX output yet")
def test_no_faked_improvement_if_original_full_task_fails():
    z=json.loads(FINAL.read_text())
    old=sum(v["original_style"]["original_strict_pass"] for v in z["results"])
    now=sum(v["contact_governor"]["original_strict_pass"] for v in z["results"])
    assert old==z["paired_original_style_strict_pass"]
    assert now==z["paired_contact_governor_strict_pass"]
    assert now<=6
    assert z["paired_change"]["style_to_governor_improved"]>=0
    assert z["paired_change"]["style_to_governor_regressed"]>=0
