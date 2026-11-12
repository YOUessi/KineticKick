"""CPU-only v121 correctness / adversarial regression; no simulation or CUDA."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "tools/v121_29dof_power_chain.py"
spec = importlib.util.spec_from_file_location("v121_29dof_power_chain", MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CFG = json.loads((Path(__file__).resolve().parents[1] /
                  "configs/v121_frozen_cpu_cohort.json").read_text())

NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


def make_trace(n=100):
    rows = []
    for i in range(n):
        rows.append({
            "time_s": (i + 1) * .005, "physics_dt_s": .005,
            "joint_pos": [0] * 29, "joint_vel": [1] * 29,
            "joint_torque_estimate": [2] * 29,
            "support_foot_force_N": {"left": 100., "right": 20.},
            "whole_body_com_xyz": [i*.001, 0., .7],
            "nonfoot_contact_N": 0.,
        })
    return {
        "root_fixed": False, "joint_names": NAMES[:],
        "traces": [rows], "joint_velocity_limits": [[20.] * 29],
        "records": [{"contact_time_s": .405}],
    }


def test_groups_disjoint_cover_all_original29():
    g = m.group_indices(NAMES)
    assert len(g["waist"]) == 3
    assert len(g["right_hip"]) == 3
    assert len(g["right_knee"]) == 1
    assert len(g["right_ankle"]) == 2
    assert len(g["left_arm"]) == 7
    assert len(g["left_support_leg"]) == 6
    assert sum(map(len, g.values())) == 29


@pytest.mark.parametrize("field", ["joint_names", "traces"])
def test_missing_or_corrupt_evidence_fail_closed(field):
    raw = make_trace()
    if field == "joint_names":
        raw[field].pop()
    else:
        raw["traces"][0][0]["joint_vel"].pop()
    with pytest.raises(ValueError):
        m.analyze_physical(raw, 100, CFG)


def test_postimpact_first_knee_violation_not_mislabeled_prestrike():
    raw = make_trace()
    knee = NAMES.index("right_knee_joint")
    raw["traces"][0][90]["joint_vel"][knee] = 30.
    out = m.analyze_physical(raw, 100, CFG)
    assert out["full_29dof_qdot"]["first_hard_limit_violation_joint"] == "right_knee_joint"
    assert out["full_29dof_qdot"]["first_hard_limit_violation_phase"] == "post1250"
    assert out["full_29dof_qdot"]["first_hard_limit_violation_time_s"] == .455
    assert out["post1250_censored"] is True
    assert out["pre400_observed_rows"] == 80
    assert out["post1250_observed_rows"] == 20


def test_no_false_hazard_when_below_limit_and_support_qualified():
    out = m.analyze_physical(make_trace(), 100, CFG)
    assert out["full_29dof_qdot"]["first_hard_limit_violation_time_s"] is None
    assert out["left_support"]["three_consecutive_frames_before_hip_onset"] is None  # hip onset is left-censored
    assert out["identifiable_waist_hip_knee_ankle_order"] is None
    assert out["window_boundary_prevents_transfer_order_identification"]
    assert out["not_human_muscle_power_not_athlete_GRF_not_causal_transfer"]
    assert out["no_qualified_wrist_hip_pair_or_link_distance"]


def test_corrupted_time_does_not_appear_as_certified_200hz():
    raw = make_trace()
    raw["traces"][0][52]["time_s"] += .005
    with pytest.raises(ValueError, match="NON_200HZ"):
        m.analyze_physical(raw, 100, CFG)


def test_tampered_sha_rejected(tmp_path):
    path = tmp_path/"one.json"
    path.write_text('{"physics":true}')
    expected = m.sha256(path)
    m.verify_file(path, expected)
    path.write_text('{"physics":false}')
    with pytest.raises(ValueError, match="SOURCE_SHA256_DRIFT"):
        m.verify_file(path, expected)


def row(control, observer=False, seed=3):
    return {
        "source_version": 116, "scene": "AG",
        "control": control, "name": control+str(observer),
        "observer": observer, "seed": seed, "ball_point": [1., 2.],
        "hard_gate_pass": control == "v32",
        "observed": {"full_29dof_qdot": {"maximum_ratio": 0.5},
          "group_estimated_motor_positive_power": {
              g: {"estimated_positive_motor_work_pre400_J": 1., "onset_s": .3, "onset_left_censored_by_pre400_boundary": False}
              for g in m.GROUP_PREFIXES}},
    }


def test_pair_matching_excludes_observer_duplicate():
    matches = m.matched_pairs([row("v32"), row("v32", True), row("cmu02")])
    assert len(matches) == 1
    assert matches[0]["baseline_pass"] is True


def test_pair_matching_rejects_unmatched_seed():
    with pytest.raises(ValueError, match="UNMATCHED_FROZEN_PAIR"):
        m.matched_pairs([row("v32"), row("cmu02", seed=11)])


def test_pair_matching_rejects_duplicate_independent():
    with pytest.raises(ValueError, match="DUPLICATE_INDEPENDENT_CONDITION"):
        m.matched_pairs([row("v32"), row("v32"), row("cmu02")])


def test_low_power_returns_absent_onset_not_fabrication():
    t = np.arange(1, 81) * .005
    p = np.zeros(80)
    assert m.sustained_onset(t, p, np.arange(80)) is None
    p[10:13] = 10.
    assert m.sustained_onset(t, p, np.arange(80)) == .055


def test_support_is_established_when_hip_activation_starts_later():
    raw = make_trace()
    hip = m.group_indices(NAMES)["right_hip"]
    for row in raw["traces"][0][:10]:
        for idx in hip: row["joint_torque_estimate"][idx] = 0.0
    result = m.analyze_physical(raw, 100, CFG)
    assert result["group_estimated_motor_positive_power"]["right_hip"]["onset_s"] == .055
    assert result["left_support"]["three_consecutive_frames_before_hip_onset"] is True
