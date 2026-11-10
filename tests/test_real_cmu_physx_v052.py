"""v52 research regressions: physics failures must remain failures, even if ball enters goal."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

AUDITS={c:ROOT/("reports/v052_cmu_"+c+"_physics_audit.json") for c in ["10_02","10_05"]}
RAW={c:ROOT/("reports/v052_cmu_"+c+"_center_physx.json") for c in AUDITS}
ZERO=ROOT/"reports/v052_cmu_10_02_center_zero_physx.json"


@pytest.mark.skipif(not all(x.exists() for x in AUDITS.values()),
                    reason="Original G1 full PhysX outcomes stored only on Tang")
def test_both_original_real_cmu_clips_evaluated_not_selected_by_offline_mse():
    scores=[json.loads(AUDITS[c].read_text()) for c in ("10_02","10_05")]
    for c,trial in zip(("10_02","10_05"),scores):
        assert trial["clip"]==c
        assert not trial["human_true_foot_ball_impact_available"]
        assert trial["nonzero_applied_reference_control_steps"]>0
        assert trial["actual_robot_joint_pose_peak_delta_from_base_rad"]>0
        assert trial["source_reference_sha256"]==sha(ROOT/"experiments/cmu_real_120hz_phase_v052"/c/"g1_joint_delta_50hz.npz")
        assert trial["physics_source_sha256"]==sha(RAW[c])
        assert trial["real_5ms_joint_speed_limit_exceedance_samples"]==0
        assert trial["ball_xy_outgoing_speed_mps"]>4.0
        assert trial["first_true_robot_instep_impulse_Ns"]>0
    assert scores[0]["task_success_all_gates"] is True
    assert scores[0]["original_full_14sec_5ms_steps"]==2800
    assert scores[1]["task_success_all_gates"] is False
    assert scores[1]["original_full_14sec_5ms_steps"]<2800
    assert scores[1]["gates"]["complete_14s_original_PhysX"] is False
    assert scores[1]["gates"]["whole_sphere_goal_and_original_task"] is False
    assert scores[1]["gates"]["recovered_standing"] is False
    assert scores[1]["gates"]["correct_real_instep_and_no_illegal_ball_contact"] is True
    assert scores[0]["physically_realized_proxy_MSE_change"]<0
    assert scores[1]["physically_realized_proxy_MSE_change"]<0


@pytest.mark.skipif(not all(x.exists() for x in RAW.values()),
                    reason="Tang real unmodified PhysX simulation not available")
def test_failure_cannot_be_hidden_by_scored_goal_or_optical_proxy():
    good=json.loads(RAW["10_02"].read_text())
    bad=json.loads(RAW["10_05"].read_text())
    assert good["records"][0]["goal"] is True
    assert bad["records"][0]["goal"] is True
    assert bad["records"][0]["recovered"] is False
    assert bad["records"][0]["standing_seconds"]==0
    assert bad["termination_reasons"]==[["physical_nonfoot"]]
    assert len(bad["kinematics"][0])==len(bad["joint_strike"]["feedback"])==257
    assert len(bad["traces"][0])==1029
    assert len(good["kinematics"][0])==700
    assert good["termination_reasons"]==[[]]


@pytest.mark.skipif(not ZERO.exists(),reason="Tang standalone zero-control physical check not yet run")
def test_disabling_new_CMU_controller_keeps_all_shared_G1_physics_fields_identical():
    z=json.loads(ZERO.read_text())
    baseline=json.loads((ROOT/"reports/v032_learned_center.json").read_text())
    assert z["joint_strike"]["mode"]=="coordination_style_v041"
    assert z["root_fixed"] is False and z["post_initialization_state_writes"] is False
    assert z["termination_reasons"]==[[]]
    assert len(z["kinematics"][0])==700
    # v32's archived trace included seven auxiliary diagnostic fields that
    # are absent from the new unchanged runner, but every SHARED 5ms physical
    # measurement must remain bitwise identical; do not fabricate omitted data.
    missing=set(baseline["traces"][0][0])-set(z["traces"][0][0])
    assert missing=={"segment_quaternions_wxyz","segment_angular_velocity_w_rad_s",
       "support_foot_ground_force_w_N","support_foot_normal_force_w_N",
       "support_foot_friction_force_w_N","support_contact_capacity_exhausted",
       "root_velocity_w"}
    for a,b in zip(z["traces"][0],baseline["traces"][0]):
        assert set(a)<=set(b)
        assert all(b[k]==v for k,v in a.items())
    for key in ("contacts","kinematics","records","sequence_records",
                "running_evidence","stride_shot_evidence","stride_shot_success"):
        assert z[key]==baseline[key],key
    for step in z["joint_strike"]["feedback"]:
        assert step["coordination_style_v041"]["enabled"] is False
        assert step["human_coordination_reference"]["enabled"] is False
        assert np.max(np.abs(step["human_coordination_reference"]["applied_delta_rad"]))==0
