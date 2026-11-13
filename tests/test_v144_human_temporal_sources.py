"""v144 actual-human source and unsigned segment orientation-rate tests (CPU)."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v144_human_temporal_source_and_physical_audit as v


@pytest.fixture(scope="module")
def cfg():
    return v.frozen()


@pytest.fixture(scope="module")
def rows(cfg):
    return v.fetch_qualified_sources(cfg)[0]


def test_frozen_commit_sha_and_no_physical_actuation(cfg):
    assert v.FREEZE_SHA == "4c220f2967fa6d9440e5159f4e4b6d30b911f537"
    assert cfg["max_new_gpu_physx_runs"] == 0
    assert cfg["evidence_policy"]["true_instestep_impact_label"] is False


def test_all_six_real_original_120hz_clips_sha_verified(cfg, rows):
    assert set(rows) == set(cfg["source_clips"])
    assert sum(x["original_C3D_frames"] for x in rows.values()) == 3358


def test_original_manifest_has_subject_disjoint_raw_but_not_event_cohort(cfg, rows):
    assert {v["subject"] for v in rows.values()} == {"10", "11"}
    assert {rows[x]["subject"] for x in cfg["strong_launch_anchor_clips"]} == {"10"}


def test_original_frozen_events_are_unverified_raw_marker_onsets(cfg, rows):
    assert rows["10_02"]["inferred_event"]["inferred_ball_launch_onset_frame"] == 353
    assert rows["10_05"]["inferred_event"]["inferred_ball_launch_onset_frame"] == 303
    assert all(rows[name]["inferred_event"]["verified_true_foot_ball_contact"] is False
               for name in cfg["strong_launch_anchor_clips"])


@pytest.mark.parametrize("omega", [0., 0.5, 2., 5.])
def test_constant_and_rotating_synthetic_vector_rate_is_exact(omega):
    t = np.arange(140) / 120.
    v3 = np.stack([np.cos(omega*t), np.sin(omega*t),
                   np.zeros_like(t)], axis=-1)
    rate, good = v.observed_rate(v3)
    assert np.all(good)
    assert np.all(np.isnan(rate[:2])) and np.all(np.isnan(rate[-2:]))
    assert np.allclose(rate[2:-2], omega, atol=5e-7)


def test_nonfinite_marker_never_interpolated():
    t = np.arange(140) / 120.
    samples = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1)
    samples[71] = [np.nan, 0., 0.]
    rate, valid = v.observed_rate(samples)
    assert not valid[71]
    assert np.isnan(rate[69]) and np.isnan(rate[73])


def test_zero_vector_never_means_zero_motion():
    samples = np.tile([1., 0., 0.], (140, 1))
    samples[70] = [0., 0., 0.]
    rate, valid = v.observed_rate(samples)
    assert not valid[70]
    assert np.isnan(rate[68]) and np.isnan(rate[72])


def test_bad_segment_shape_rejected():
    with pytest.raises(ValueError, match="INVALID_SEGMENT"):
        v.observed_rate(np.ones((4, 4)))


@pytest.mark.parametrize("bad_anchor", [True, -1, 2, 500])
def test_unknown_or_fabricated_impact_window_rejected(bad_anchor):
    with pytest.raises(ValueError, match="MISSING_FROZEN_PRE"):
        v.pre_onset_descriptor(np.ones(150), bad_anchor)


def test_missing_more_than_22_of_72_window_frames_fails_closed():
    rate = np.ones(150)
    rate[80:105] = np.nan
    with pytest.raises(ValueError, match="INSUFFICIENT_UNIMPUTED"):
        v.pre_onset_descriptor(rate, 120)


def test_exact_72_frame_window_excludes_onset():
    x = np.arange(150, dtype=float)
    result = v.pre_onset_descriptor(x, 120)
    assert result["window_frames"] == 72
    assert result["valid_pre_frames"] == 72
    assert result["first_order_time_index_pre"] == 71
    assert result["max_orientation_rate_proxy_rad_per_s"] == 119.
    assert result["peak_pre_ball_marker_onset_s"] == round(1 / 120, 6)


def test_original_real_six_clip_extraction_does_not_invent_events(cfg, rows):
    x = v.six_clip_pilot(cfg, rows)
    assert x["raw_source_clips"] == 6
    assert x["raw_original_c3d_frames"] == 3358
    assert x["qualified_weak_ball_marker_launch_events"] == 2
    assert x["distinct_subjects_with_weak_event"] == 1
    assert not x["subject_disjoint_temporal_event_model_evaluation_eligible"]
    assert x["clips_without_valid_ball_marker_anchor"] == 4
    for name in ("10_01", "10_03", "10_06", "11_01"):
        assert x["per_clip"][name]["unverified_ball_marker_launch_anchor_frame"] is None
        assert all(row["pre_ball_marker_launch_descriptor"] is None
                   for row in x["per_clip"][name]["segments"].values())


def test_relabel_weak_short_burst_as_reliable_contact_is_rejected(cfg, rows):
    fake = copy.deepcopy(rows)
    fake["10_03"]["observed_ball_kinematics_quality"] = v.QUALIFIED
    with pytest.raises(ValueError, match="UNDECLARED_EVENT"):
        v.six_clip_pilot(cfg, fake)


def test_marker_label_missing_or_ambiguous_fails_closed():
    pairs = {"pelvis": ["LHJ0", "RHJ0"]}
    with pytest.raises(ValueError, match="AMBIGUOUS_ANATOMICAL"):
        v.named_markers(["S:LHJ0"], np.zeros((4, 1, 3)), pairs)
    with pytest.raises(ValueError, match="AMBIGUOUS_ANATOMICAL"):
        v.named_markers(["S:LHJ0", "A:LHJ0", "S:RHJ0"],
                        np.zeros((4, 3, 3)), pairs)


def test_provenance_digest_tampering_is_rejected(cfg, monkeypatch):
    monkeypatch.setattr(v, "sha256", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="SOURCE_SHA256_DRIFT"):
        v.fetch_qualified_sources(cfg)


def test_write_once_refuses_replacement_and_symlink(tmp_path):
    dest = tmp_path / "existing.json"
    dest.write_text("{}")
    with pytest.raises(ValueError, match="OVERWRITE"):
        v.write_once(dest, {"bad": True})
    link = tmp_path / "ln.json"
    link.symlink_to(dest)
    with pytest.raises(ValueError, match="OVERWRITE"):
        v.write_once(link, {"bad": True})


def test_write_once_and_json_nonfinite_rejected(tmp_path):
    dest = tmp_path / "new.json"
    v.write_once(dest, {"count": 1})
    assert json.loads(dest.read_text()) == {"count": 1}
    with pytest.raises(ValueError, match="OVERWRITE"):
        v.write_once(dest, {"count": 2})
    bad = tmp_path / "nan.json"
    with pytest.raises(ValueError):
        v.write_once(bad, {"not_number": float("nan")})
    assert not bad.exists()
