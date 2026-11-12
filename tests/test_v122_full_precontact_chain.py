"""CPU-only adversarial tests: no new PhysX runs, no CUDA initialization."""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import v122_full_precontact_chain as new
import v121_29dof_power_chain as old

CFG = json.loads((ROOT / "configs/v122_frozen_full_precontact.json").read_text())
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


def synthetic(n=350, impact=1.2):
    rows = []
    for i in range(n):
        rows.append({
            "time_s": round((i + 1) * 0.005, 8),
            "physics_dt_s": 0.005,
            "joint_pos": [0.] * 29,
            "joint_vel": [1.] * 29,
            "joint_torque_estimate": [0.] * 29,
            "support_foot_force_N": {"left": 100., "right": 20.},
            "whole_body_com_xyz": [0.001 * i, 0., .7],
            "nonfoot_contact_N": 0.,
        })
    return {
        "root_fixed": False, "joint_names": NAMES[:],
        "joint_velocity_limits": [[20.] * 29], "traces": [rows],
        "records": [{"contact_time_s": impact}],
    }


def power_group(raw, name, start_s, stop_s=1.19, value=2.):
    ix = old.group_indices(raw["joint_names"])[name][0]
    for row in raw["traces"][0]:
        if start_s - 1e-9 <= row["time_s"] <= stop_s + 1e-9:
            row["joint_torque_estimate"][ix] = value


def test_full_precontact_observation_is_not_equivalent_to_last400():
    r = synthetic()
    power_group(r, "waist", .2)
    o = new.analyze_physics(r, 350, CFG)
    w = o["estimated_robot_motor_power_groups"]["waist"]
    assert o["full_precontact_rows"] == 239
    assert o["pre400_rows"] == 80
    assert w["last400_primary20"]["left_censored"] is True
    assert w["full_primary20"]["left_censored"] is False
    assert w["full_primary20"]["onset_s"] == .2
    assert o["source_initial_state_may_precede_first_sample"] is True
    assert o["no_biological_power_transfer_causality_certified"] is True


def test_full_sim_initial_active_still_left_censored():
    r = synthetic()
    for group in ("waist", "right_hip", "right_knee", "right_ankle"):
        power_group(r, group, .005)
    o = new.analyze_physics(r, 350, CFG)
    assert o["full_primary20_sequence_order_observable"] is None
    assert o["full_sensitivity40_sequence_order_observable"] is None
    assert o["estimated_robot_motor_power_groups"]["waist"]["full_primary20"]["left_censored"] is True


def test_all_zero_power_never_makes_up_an_onset_or_work():
    o = new.analyze_physics(synthetic(), 350, CFG)
    assert o["full_primary20_sequence_order_observable"] is None
    assert o["estimated_robot_motor_power_groups"]["waist"]["full_primary20"]["onset_s"] is None
    assert o["estimated_robot_motor_power_groups"]["waist"]["full_work_J"] == 0.


def test_observed_order_is_only_kinematic_power_proxy_not_biological_cause():
    r = synthetic()
    for name, t in zip(("waist", "right_hip", "right_knee", "right_ankle"), (.2, .3, .4, .5)):
        power_group(r, name, t)
    o = new.analyze_physics(r, 350, CFG)
    assert o["full_primary20_sequence_order_observable"] is True
    assert o["no_biological_power_transfer_causality_certified"] is True
    assert o["sequence_threshold_robust_if_both_identifiable"] is True


def test_reversed_power_sequence_falsifies_observed_order():
    r = synthetic()
    for name, t in zip(("waist", "right_hip", "right_knee", "right_ankle"), (.5, .4, .3, .2)):
        power_group(r, name, t)
    assert new.analyze_physics(r, 350, CFG)["full_primary20_sequence_order_observable"] is False


def test_fixed_sensitivity_changes_onset_without_choosing_better_threshold():
    r = synthetic()
    power_group(r, "waist", .2, .25, 1.)
    power_group(r, "waist", .7, .85, 4.)
    o = new.analyze_physics(r, 350, CFG)
    w = o["estimated_robot_motor_power_groups"]["waist"]
    assert w["full_primary20"]["onset_s"] == .2
    assert w["full_sensitivity40"]["onset_s"] == .7


def test_prior_support_identifiable_only_when_hip_onset_not_left_censored():
    r = synthetic()
    power_group(r, "right_hip", .3)
    o = new.analyze_physics(r, 350, CFG)
    assert o["left_support_full_precontact"]["already_established_before_non_censored_hip_onset"] is True
    assert o["left_support_full_precontact"]["onset_left_censored_by_first_sim_frame"] is True
    power_group(r, "right_hip", .005, .295)
    o2 = new.analyze_physics(r, 350, CFG)
    assert o2["left_support_full_precontact"]["already_established_before_non_censored_hip_onset"] is None


def test_tampered_time_grid_fails_closed():
    r = synthetic()
    r["traces"][0][125]["time_s"] += .005
    with pytest.raises(ValueError, match="NON_200HZ"):
        new.analyze_physics(r, 350, CFG)


def test_incomplete_joint_dimension_fails_closed():
    r = synthetic()
    r["traces"][0][12]["joint_torque_estimate"].pop()
    with pytest.raises(ValueError):
        new.analyze_physics(r, 350, CFG)


def test_source_tamper_SHA_fails_closed(tmp_path):
    path = tmp_path / "historical.json"
    path.write_text('{"value":1}')
    h = new.sha256(path)
    old.verify_file(path, h)
    path.write_text('{"value":2}')
    with pytest.raises(ValueError, match="SOURCE_SHA256_DRIFT"):
        old.verify_file(path, h)


def test_matched_triplets_exclude_OFF_ON_replicates():
    def row(c, observer=False, seed=21):
        return {
            "version": 116, "scene": "AG", "control": c,
            "observer": observer, "seed": seed, "ball_point": [1., 0.],
            "strict_pass": c == "v32",
            "analysis": {"estimated_robot_motor_power_groups": {
                k: {"full_work_J": 1.,
                    "full_primary20": {"onset_s": .5, "left_censored": False}}
                for k in old.GROUP_PREFIXES}},
        }
    result = new.matched([row("v32"), row("cmu02"), row("cmu05"), row("v32", True)])
    assert len(result) == 2
    assert all(r["baseline_strict_pass"] for r in result)


def test_mismatched_seed_cannot_be_falsely_paired():
    def row(c, seed):
        return {
            "version": 116, "scene": "AG", "control": c,
            "observer": False, "seed": seed, "ball_point": [1., 0.],
            "strict_pass": True,
            "analysis": {"estimated_robot_motor_power_groups": {
                k: {"full_work_J": 1.,
                    "full_primary20": {"onset_s": .5, "left_censored": False}}
                for k in old.GROUP_PREFIXES}},
        }
    with pytest.raises(ValueError, match="UNMATCHED_SEED"):
        new.matched([row("v32", 1), row("cmu02", 2), row("cmu05", 1)])

def test_write_once_is_atomic_for_serialization_and_refuses_overwrite(tmp_path):
    dest = tmp_path / "audit.json"
    with pytest.raises(TypeError, match="UNSERIALIZABLE"):
        new.write_once_json(dest, {"unserializable": object()})
    assert not dest.exists()
    new.write_once_json(dest, {"numpy_scalar": np.bool_(True)})
    assert json.loads(dest.read_text())["numpy_scalar"] is True
    with pytest.raises(FileExistsError):
        new.write_once_json(dest, {"anything": "else"})
    assert json.loads(dest.read_text()) == {"numpy_scalar": True}
