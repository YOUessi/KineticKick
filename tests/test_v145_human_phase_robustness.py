"""v145 CPU-only frozen posthoc source and full-grid independent tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v144_human_temporal_source_and_physical_audit as v144
import v145_phase_order_robustness as v
import v145_independent_phase_audit as independent
from v142_native_pair_ledger import safe_write_once


@pytest.fixture(scope="module")
def cfg():
    return v.frozen()


@pytest.fixture(scope="module")
def real_quality(cfg):
    return v144.fetch_qualified_sources(v144.frozen())[0]


@pytest.fixture(scope="module")
def grid(cfg, real_quality):
    return v.compute_grid(cfg, real_quality)


def test_preimplementation_sha_and_all_negative_claims(cfg):
    assert v.FREEZE_SHA == "6a277845fd858a67361721b94a87f119756d68a0"
    assert cfg["counts"]["all_chain_evaluations"] == 720
    assert all(cfg["evidence_policy"].values())


def test_all_original_human_sources_sha_and_subject_split(real_quality, cfg):
    assert set(real_quality) == set(cfg["source_clips"])
    assert sum(x["original_C3D_frames"] for x in real_quality.values()) == 3358
    assert {real_quality[x]["subject"] for x in cfg["weak_anchor_clips"]} == {"10"}


@pytest.mark.parametrize("width", [1, 5, 9])
def test_nan_median_no_filled_mocap_gap(width):
    a = np.arange(31, dtype=float)
    a[15] = np.nan
    smoothed = v.median_without_imputation(a, width)
    assert np.isnan(smoothed[15])
    if width > 1:
        assert all(np.isnan(smoothed[i]) for i in range(15-width//2, 16+width//2))


@pytest.mark.parametrize("width", [0, 2, 6, 11, -1])
def test_unregistered_smoothing_fails(width):
    with pytest.raises(ValueError, match="INVALID_SMOOTH"):
        v.median_without_imputation(np.arange(20.),width)


@pytest.mark.parametrize("span", [1,2,3,4])
@pytest.mark.parametrize("omega", [0.0,0.8,3.0])
def test_independent_atan2_known_synthetic_angular_rate(span, omega):
    t=np.arange(140)/120
    q=np.array([np.cos(omega*t),np.sin(omega*t),np.zeros(len(t))]).T
    rate,valid=independent.atan2_observed_rate(q,span=span)
    assert valid.all()
    np.testing.assert_allclose(rate[span:-span],omega,atol=1e-11)


def test_independent_atan2_not_imputing_invalid_marker():
    t=np.arange(140)/120
    q=np.array([np.cos(t),np.sin(t),np.zeros(len(t))]).T
    q[43]=np.nan
    rate,valid=independent.atan2_observed_rate(q,span=2)
    assert not valid[43]
    assert np.isnan(rate[41]) and np.isnan(rate[45])


@pytest.mark.parametrize("anchor", [-1, True, 200])
def test_bad_anchor_rejected_not_promoted(anchor):
    rates={"pelvis":np.ones(100),"right_thigh":np.ones(100),
           "right_shank":np.ones(100)}
    result=v.phase_order_test(rates,("pelvis","right_thigh","right_shank"),anchor,.6)
    assert result["status"]=="INVALID_ANCHOR_OR_WINDOW"


def test_margin_strict_and_reverse_controls():
    n=200
    rates={k:np.zeros(n) for k in ("p","t","s")}
    for i,k in zip((80,83,87),("p","t","s")):
        rates[k][i]=10.
    assert v.phase_order_test(rates,("p","t","s"),100,.4)["status"]=="ORDER_MARGIN_PASS"
    rates["s"][:]=0
    rates["s"][80]=10
    assert v.phase_order_test(rates,("p","t","s"),100,.4)["status"]=="REVERSED_OR_TIED"


def test_incomplete_window_not_silently_successful():
    a={k:np.ones(200) for k in ("p","t","s")}
    a["p"][70:100]=np.nan
    res=v.phase_order_test(a,("p","t","s"),100,.4,min_fraction=.75)
    assert res["status"]=="INSUFFICIENT_VALID_MARKERS"


def test_actual_frozen_grid_denominator_and_statuses(grid,cfg):
    assert len(grid)==720
    assert set((r["clip"],r["chain"]) for r in grid)=={
        ("10_02","right"),("10_02","left"),("10_05","right"),("10_05","left")}
    assert all(r["status"] in ("ORDER_MARGIN_PASS","STRICT_ORDER_SMALL_MARGIN",
                               "REVERSED_OR_TIED","INSUFFICIENT_VALID_MARKERS",
                               "INVALID_ANCHOR_OR_WINDOW","INVALID_SOURCE_LENGTH")
               for r in grid)


def test_original_v144_right_order_not_turned_into_true_contact(grid,cfg):
    out=v.summarize(grid,cfg)["per_chain_event"]
    assert all(out["right"][clip]["baseline_v144_choice"]["status"]=="ORDER_MARGIN_PASS"
               for clip in cfg["weak_anchor_clips"])
    assert all(out["left"][clip]["baseline_v144_choice"]["status"]!="ORDER_MARGIN_PASS"
               for clip in cfg["weak_anchor_clips"])


def test_all_720_robustness_decisions_and_paired_control(grid,cfg):
    summary=v.summarize(grid,cfg)
    expected={("right","10_02"):129,("right","10_05"):108,
              ("left","10_02"):0,("left","10_05"):4}
    for (side,clip),n in expected.items():
        assert summary["per_chain_event"][side][clip]["margin_pass_count"]==n
    pair=summary["bilateral_specificity_control"]
    assert pair["10_02"]==dict(right_only_pass=129,left_only_pass=0,both_pass=0,neither_pass=51)
    assert pair["10_05"]==dict(right_only_pass=105,left_only_pass=1,both_pass=3,neither_pass=71)


def test_original_qualifications_not_allowing_anchor_creation(cfg,real_quality):
    fake=copy.deepcopy(real_quality)
    fake["10_02"]["inferred_event"]["verified_true_foot_ball_contact"]=True
    with pytest.raises(ValueError,match="WEAK_ANCHOR_MISREPRESENTED"):
        v.compute_grid(cfg,fake)


def test_write_once_no_rewrite_existing_real_evidence(tmp_path):
    p=tmp_path/"raw.json"
    safe_write_once(p,{"unmodified":True})
    with pytest.raises(FileExistsError):
        safe_write_once(p,{"unmodified":False})
    assert json.loads(p.read_text())["unmodified"] is True


def test_real_full_run_preserves_historical_negative_and_reports_only_cpu(tmp_path):
    result,_=v.run(tmp_path/"grid.json")
    assert result["new_physical_rollouts"]==0
    assert result["historical_physics"]["v54_actual_5ms_states"]==6628
    assert result["historical_physics"]["v53"]["historical_real_PhysX_trials_sha_verified"]==12
    assert result["g1_native_left_wrist_hip_pair_manifold"]=="UNKNOWN"
    assert not result["g1_motor_action_authorized"]
    assert result["weak_event_subjects"]==["10"]


def test_full_720_independent_atan2_verified_on_real_source(tmp_path):
    primary_path=tmp_path/"main.json"
    independently_checked=tmp_path/"independent.json"
    v.run(primary_path)
    result,_=independent.review(primary_path,independently_checked)
    assert result["exact_matched_decisions"]==720
    assert result["exact_matched_3_body_peak_frame_indices"]==720
    assert result["maximum_absolute_peak_rate_numerical_difference_rad_per_s"]<1e-7
    assert result["discrepancies"]==[]


def test_tampered_primary_refuses_false_physx_success(tmp_path):
    a=tmp_path/"tampered.json"
    a.write_text(json.dumps({"version":"v0.1.145","protocol_commit_sha":v.FREEZE_SHA,
                            "source_frame_count_unmodified":3358,
                            "new_physical_rollouts":12,
                            "g1_native_left_wrist_hip_pair_manifold":"CERTIFIED"}))
    with pytest.raises(ValueError,match="PRIMARY_EVIDENCE_PROVENANCE"):
        independent.review(a,tmp_path/"shouldnotexist.json")
    assert not (tmp_path/"shouldnotexist.json").exists()
