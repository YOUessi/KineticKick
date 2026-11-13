"""v134 authentic SHA-qualified G1 29DoF event window, negative cases."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v134_real_29dof_coupled_window as audit


def test_original_group_partition_exactly_29():
    groups = audit.group_indices()
    assert sum(len(v) for v in groups.values()) == 29
    assert "waist_trunk" in groups
    assert "left_shoulder_elbow_wrist" in groups
    assert "right_hip_knee_ankle" in groups


def synthetic(case, *, steps=None, start=.005):
    if steps is None:
        steps = 1028 if case == "cmu_10_05" else 2800
    t = start + np.arange(steps) * .005
    q = np.zeros((steps, 29))
    dq = np.zeros((steps, 29))
    return {
        "t": t, "q": q, "dq": dq, "com": np.zeros((steps,3)),
        "support": np.ones((steps,2)) * 50,
        "nonfoot": np.zeros(steps),
        "limits": np.ones(29) * 10,
    }


@pytest.mark.parametrize("case,post", [
    ("frozen_v32", 251), ("cmu_10_02", 251), ("cmu_10_05", 0),
])
def test_window_size_and_missing_post_explicit(case, post):
    result = audit.frame_window_audit(case, synthetic(case))
    assert result["real_pre_0p4s_frames"] == 80
    assert result["real_post_1p25s_frames"] == post
    assert result["missing_post_frames"] == 251-post
    assert not result["missing_post_failure_trajectory_imputed"]


def test_400ms_and_1250ms_window_uses_exact_frozen_physics_clock():
    a = audit.frame_window_audit("frozen_v32", synthetic("frozen_v32"))
    assert a["pre_0p4s_left_right_foot_support_force_proxy_integrals_Ns"] == pytest.approx([20.,20.])
    assert a["post_1p25s_left_right_support_proxy_integrals_Ns"] == pytest.approx([62.75,62.75])


def test_failure_never_populates_missing_post_event_support():
    r = audit.frame_window_audit("cmu_10_05", synthetic("cmu_10_05"))
    assert r["post_1p25s_left_right_support_proxy_integrals_Ns"] is None
    assert r["post_1p25s_joint_speed_exceedance_frames"] is None
    assert r["post_1p25s_nonfoot_peak_N"] is None


@pytest.mark.parametrize("modify,err", [
    (lambda x: x.update(dq=np.zeros((100, 29))), "SHAPE"),
    (lambda x: x.update(support=np.zeros((2800, 1))), "SHAPE"),
    (lambda x: x.update(com=np.zeros((2800, 2))), "SHAPE"),
    (lambda x: x.update(q=np.zeros((2800, 28))), "SHAPE"),
    (lambda x: x.update(limits=np.zeros(29)), "SHAPE"),
])
def test_dimension_or_limit_corruption_fails_closed(modify, err):
    x = synthetic("frozen_v32")
    modify(x)
    with pytest.raises(ValueError, match=err):
        audit.frame_window_audit("frozen_v32", x)


def test_missing_pre_window_is_not_interpolated():
    with pytest.raises(ValueError, match="MISSING_FROZEN_400MS"):
        audit.frame_window_audit("frozen_v32",
                                  synthetic("frozen_v32", steps=2800, start=4.80))


def test_200hz_gap_rejected():
    x = synthetic("frozen_v32")
    x["t"][910] += .002
    with pytest.raises(ValueError, match="TIME_GAP"):
        audit.frame_window_audit("frozen_v32", x)


def test_corrupted_nan_rejected():
    x = synthetic("frozen_v32")
    x["nonfoot"][99] = float("nan")
    with pytest.raises(ValueError, match="SHAPE"):
        audit.frame_window_audit("frozen_v32", x)


def test_nonfoot_bad_can_coexist_with_no_qdot_exceedance():
    x = synthetic("cmu_10_05")
    x["nonfoot"][1026] = 25.89
    r = audit.frame_window_audit("cmu_10_05", x)
    assert r["pre_0p4s_nonfoot_over_20N_frames"] == 1
    assert r["pre_0p4s_joint_speed_exceedance_frames"] == 0
    assert r["pre_0p4s_max_nonfoot_net_force_N"] == 25.89


def test_source_original_3_episode_6628_frames_sha():
    r = audit.analyze_three_original_v54()
    assert r["source_sha_frozen"] and r["g1_true_physx_states_reexamined"] == 6628
    assert r["new_real_physx_episodes"] == 0
    fail = r["cases"]["cmu_10_05"]
    assert fail["real_post_1p25s_frames"] == 0
    assert fail["missing_post_frames"] == 251
    assert fail["pre_0p4s_named_sensor_force_peak_s"] == 5.135
    assert fail["pre_0p4s_max_nonfoot_net_force_N"] > 25.89
    assert not r["native_actor_collider_pair_verified"]


def test_original_off_on_all_nine_phyx_common_fields_SHA_locked():
    parity = audit.reverify_original_v52_v54_off_on()
    assert set(parity) == set(audit.CASES)
    for case,result in parity.items():
        assert result["all_common_physical_fields_exact"]
        assert result["common_fields_compared"] == 9
        assert result["this_is_not_current_v134_runtime_observer_invariance"]
        assert len(result["off_original_v52_sha256"]) == 64


def test_original_off_on_one_changed_joint_q_is_detected():
    off = {key: [1] for key in audit.v128.ORIGINAL_SHARED_PHYSICAL_FIELDS}
    on = copy.deepcopy(off)
    on["traces"] = [{"joint_pos": [2]}]
    result = audit.exact_legacy_off_on_common_field_parity(
        off, on, field_names=audit.v128.ORIGINAL_SHARED_PHYSICAL_FIELDS)
    assert not result["all_common_physical_fields_exact"]
    assert result["nonidentical_fields"] == ["traces"]


def test_original_off_on_missing_field_cannot_pass():
    off = {key: [1] for key in audit.v128.ORIGINAL_SHARED_PHYSICAL_FIELDS}
    on = copy.deepcopy(off)
    del on["contacts"]
    with pytest.raises(ValueError, match="FIELD_MISSING"):
        audit.exact_legacy_off_on_common_field_parity(
            off, on, field_names=audit.v128.ORIGINAL_SHARED_PHYSICAL_FIELDS)
