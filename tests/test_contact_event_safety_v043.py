"""v43 strict original-G1/PhysX event-aligned qdot safety evidence tests.

Run on Tang local ignored 12 physics episodes; public clones skip raw-data parts.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from build_contact_event_safety_v043 import indices
from analyze_contact_event_safety_v043 import stats

STUDY=ROOT/"configs/contact_event_safety_v043.json"
DATA=ROOT/"data/research_human_kicks/contact_event_safety_v043/event_aligned_qdot_dynamics.npz"
META=ROOT/"data/research_human_kicks/contact_event_safety_v043/event_aligned_audit.json"
RISK=ROOT/"data/research_human_kicks/contact_event_safety_v043/retrospective_qdot_guard_audit.json"

def test_strict_original_5ms_event_alignment_ignores_interpolated_hypotheticals():
    times=np.arange(0.005,3.005,.005)
    original=[{"time_s":float(x)} for x in times]
    relative=np.arange(-50,201,dtype=float)*.005
    idx=indices(original,1.57,relative)
    assert len(idx)==251
    assert np.all(np.diff(idx)==1)
    np.testing.assert_allclose(times[idx]-1.57,relative,atol=4e-16,rtol=0)
    with pytest.raises(ValueError):
        indices(original,1.572,relative)

def test_small_sample_risk_confusion_with_false_negative_is_preserved():
    result=stats([True,False,True,False],[True,True,False,False])
    assert result["tp"]==1 and result["fp"]==1
    assert result["fn"]==1 and result["tn"]==1
    assert result["recall"]==.5
    assert result["precision"]==.5

def test_prespecified_first_contact_window_is_not_human_event_truth():
    protocol=json.loads(STUDY.read_text())
    assert protocol["version"]=="0.1.43"
    assert protocol["no_new_physics"] is True
    assert protocol["no_new_training"] is True
    assert protocol["v42_data_is_development_not_future_holdout"] is True
    assert protocol["precontact_warn_ratio"]==.9
    assert protocol["postcontact_failure_ratio"]==1.
    assert protocol["bins"]==251
    assert protocol["window_start_s"]==-.25
    assert protocol["window_end_s"]==1.
    assert len(protocol["source_sha256"])==17

@pytest.mark.skipif(not DATA.exists(),reason="Original G1/PhysX binary data retained on Tang only")
def test_actual_event_dataset_and_original_physical_sources_are_sha_certified():
    study=json.loads(STUDY.read_text())
    for name,digest in study["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,(name,"source drift")
    meta=json.loads(META.read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==meta["output_sha256"]
    assert meta["episode_count"]==12 and meta["sample_count_per_episode"]==251
    assert meta["all_source_episodes_verified_by_sha"]
    with np.load(DATA,allow_pickle=False) as z:
        assert z["qdot_ratio"].shape==(12,251,29)
        assert z["support"].shape==(12,251,2)
        assert z["v"].shape==(12,251,29)
        assert z["ball_v"].shape==(12,251,3)
        assert bool(z["exact_original_physx_5ms_samples"]) is True
        assert not bool(z["verified_real_g1_hardware"])
        assert len(set(z["source_case"]))==6
        np.testing.assert_allclose(np.diff(z["relative_contact_s"]),.005,atol=1e-7,rtol=0)

@pytest.mark.skipif(not RISK.exists(),reason="Original full 14s 12-rollout qdot safety audit only on Tang")
def test_original_delayed_overruns_cannot_be_hidden_by_preimpact_threshold():
    row=json.loads(RISK.read_text())
    assert row["all_run_source_hashes_verified"]
    assert row["controller_episode_count"]==12
    assert row["actual_entire_episode_overspeed_count"]==5
    assert row["pre_quiet_but_future_overrun_count"]==2
    assert row["contact_to_plus1s_window_missed_overruns"]==1
    assert not row["guarantee_of_future_safe_motion"]
    x=row["pre_impact_qdot_ratio_warning_confusion_vs_any_full14s_actual_limit_exceedance"]
    assert (x["tp"],x["fp"],x["fn"],x["tn"])==(3,1,2,6)
    assert row["risk_data_is_independent_test"] is False
    a=next(z for z in row["cases"] if z["id"]=="trial01" and z["controller"]=="style_v41")
    assert a["pre_0to_minus250ms_actual_max_qdot_ratio"]<.72
    assert a["full14s_speed_limit_violation"]
    assert .45<a["first_original_full_episode_qdot_overrun"]["relative_time_to_ball_contact_s"]<.47
    assert a["first_original_full_episode_qdot_overrun"]["joint"]=="left_ankle_roll_joint"
    b=next(z for z in row["cases"] if z["id"]=="trial06" and z["controller"]=="original_v32")
    assert b["contact_plus_1s_scope_misses_hazard"]
    assert b["first_original_full_episode_qdot_overrun"]["relative_time_to_ball_contact_s"]>1.
