"""Offline tests for v123 first-state evidence, without GPU or mock robot success."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v123_initial_runup_observability as mod

CFG = json.loads(mod.CONFIG.read_text())
NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def fake_ref(n=10):
    return {"names": NAMES, "t": np.arange(1, n + 1) * 0.005,
            "q": np.zeros((n, 29)), "v": np.zeros((n, 29))}


def fake_physics(n=400, first_speed=0.0):
    traces = []
    for i in range(n):
        traces.append({
            "time_s": (i + 1) * 0.005, "physics_dt_s": 0.005,
            "joint_pos": [0.0] * 29,
            "joint_vel": [0.0] * 29,
            "joint_torque_estimate": [0.0] * 29,
            "root_xyz": [0.0, i * 0.005, 0.8],
            "root_speed_xy": first_speed if i == 0 else 0.0,
            "root_tilt_deg": 3.0,
            "whole_body_com_xyz": [0.0, i * 0.005, 0.7],
            "support_foot_force_N": {"left": 60.0 if i >= 3 else 0.0, "right": 0.0},
            "nonfoot_contact_N": 0.0,
            "coordination_phase": "idle",
        })
    return {"root_fixed": False, "joint_names": NAMES,
            "joint_velocity_limits": [10.0] * 29, "traces": [traces],
            "records": [{"contact_time_s": 0.5}]}


def test_first_sustained_requires_three_actual_frames():
    t = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    assert mod.first_sustained(t, [60, 60, 0, 60, 0, 0], 50)["onset_s"] is None
    assert mod.first_sustained(t, [0, 60, 60, 60, 0, 0], 50)["onset_s"] == 0.01


def test_first_sustained_left_censor():
    assert mod.first_sustained([.005, .01, .015], [55] * 3, 50) == {
        "onset_s": .005, "left_censored": True
    }


def test_empty_or_ill_formed_support_rejected():
    with pytest.raises(ValueError, match="BAD_SUPPORT_TRACE"):
        mod.first_sustained([0.005], [1, 2], 50)


def test_first_frame_motion_not_rest_like():
    assert not mod.rest_like(1.8, [0.0] * 29, CFG)


def test_single_of_29_moving_disqualifies_rest():
    v = [0.0] * 29
    v[-1] = .11
    assert not mod.rest_like(0, v, CFG)


def test_zero_state_is_rest_like_sentinel_only():
    assert mod.rest_like(0.0, [0.0] * 29, CFG)


def test_shape_or_nan_rejected():
    with pytest.raises(ValueError, match="BAD_29DOF_VELOCITY"):
        mod.rest_like(0.0, [0.0] * 28, CFG)
    with pytest.raises(ValueError, match="BAD_ROOT_SPEED"):
        mod.rest_like(float("nan"), [0.0] * 29, CFG)


def test_first_hard_limit_occurs_at_row_zero():
    t = np.array([0.005, 0.01])
    vel = np.zeros((2, 29))
    vel[0, 28] = 11
    r = mod.first_hard_limit(t, vel, np.ones(29) * 10, NAMES)
    assert r["first_time_s"] == 0.005
    assert r["first_joint"] == "right_wrist_yaw_joint"


def test_first_effect_exact_200hz_boundary():
    a, b = fake_ref(), fake_ref()
    b["q"][3, 12] = 2e-5
    r = mod.first_effect(a, b, 1e-5)
    assert r["first_s"] == .02
    assert r["common_prefix_200hz_frames"] == 3


def test_first_effect_exact_noise_below_threshold():
    a, b = fake_ref(), fake_ref()
    b["v"][0, 2] = 5e-6
    r = mod.first_effect(a, b, 1e-5)
    assert r["first_s"] is None
    assert r["common_prefix_200hz_frames"] == 10


def test_first_effect_does_not_pair_different_grid_or_joint_order():
    a, b = fake_ref(), fake_ref()
    b["t"][4] += .005
    with pytest.raises(ValueError, match="MISMATCHED_TIME_GRID"):
        mod.first_effect(a, b, 1e-5)
    a, b = fake_ref(), fake_ref()
    b["names"] = list(reversed(NAMES))
    with pytest.raises(ValueError, match="MISMATCHED_29DOF_ORDER"):
        mod.first_effect(a, b, 1e-5)


def test_synthetic_initial_moving_and_sustained_support():
    x = fake_physics(first_speed=1.8)
    r, _ = mod.audit_trace(x, 400, CFG)
    assert r["initial_start_time_left_censored_if_already_moving"]
    assert r["first_measured_sustained_support_left"]["onset_s"] == .02
    assert r["root_xy_precontact_path_m"] > .4
    assert r["postimpact_1250ms_measured_frames"] == 250


def test_synthetic_first_joint_qdot_nonzero():
    x = fake_physics()
    x["traces"][0][0]["joint_vel"][1] = .2
    r, _ = mod.audit_trace(x, 400, CFG)
    assert not r["first_frame_all29_and_root_rest_like"]


def test_non200hz_source_fails_closed():
    x = fake_physics()
    x["traces"][0][25]["time_s"] += 0.001
    with pytest.raises(ValueError, match="NON_200HZ"):
        mod.audit_trace(x, 400, CFG)


def test_write_once_never_overwrites_existing_audit(tmp_path):
    p = tmp_path / "audit.json"
    mod.write_once_json(p, {"success": False})
    with pytest.raises(FileExistsError):
        mod.write_once_json(p, {"success": True})
    assert json.loads(p.read_text()) == {"success": False}


def test_source_sha_mismatch_rejected(tmp_path):
    p = tmp_path / "immutable.json"
    p.write_text("original data")
    with pytest.raises(ValueError, match="SOURCE_SHA256_DRIFT"):
        mod.prior.verify_file(p, "0" * 64)
