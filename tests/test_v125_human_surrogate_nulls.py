"""v125 frozen, counterfactual real-human lag assessment; never causal force labels."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v125_human_surrogate_nulls as m

CFG=json.loads(m.CFG_PATH.read_text())

def synthetic_pair(n=600,lag=9):
    rng=np.random.default_rng(125)
    x=rng.standard_normal(n)
    y=np.r_[np.zeros(lag),x[:-lag]]
    return x,y

def test_shift_design_is_frozen_and_unique_and_avoids_alignment():
    s=m.shift_candidates(590,120,.5,127)
    assert len(s)==127 and len(set(s))==127
    assert min(s)>=60 and max(s)<=530
    assert s[0]==60 and s[-1]==530
    assert m.shift_candidates(150,120,.5,127)==list(range(60,91))

def test_no_available_null_is_rejected():
    with pytest.raises(ValueError,match="NO_SEPARATED"):
        m.shift_candidates(100,120,.5,127)

def test_null_only_shifts_second_signal_not_both():
    a,b=synthetic_pair()
    out=m.surrogate_audit(a,b,CFG)
    assert out["observed"]["best"]["lag_frames"]==9
    assert out["surrogates_attempted"]==127
    assert out["surrogates_qualified"]==127
    assert 0<=out["descriptive_rank_exceedance"]<=1
    assert out["temporal_split"]["stable_under_contiguous_split"]
    assert out["causal_lag_or_force_transfer_certified"] is False

def test_time_reversal_or_inverting_lag_is_not_causality():
    a,b=synthetic_pair(600,8)
    orig=m.surrogate_audit(a,b,CFG)
    reverse=m.surrogate_audit(a[::-1],b[::-1],CFG)
    assert orig["observed"]["best"]["lag_frames"]==8
    assert reverse["observed"]["best"]["lag_frames"]==-8
    assert orig["causal_lag_or_force_transfer_certified"] is False
    assert reverse["causal_lag_or_force_transfer_certified"] is False

def test_zero_variance_not_upgraded_to_professional_motion():
    out=m.surrogate_audit(np.ones(220),np.ones(220),CFG)
    assert out["qualified"] is False
    assert out["reason"]=="UNIDENTIFIABLE_OR_CONSTANT"

def test_nan_interval_mask_remains_unqualified_when_too_sparse():
    a,b=synthetic_pair(250)
    a[30:]=np.nan;b[30:]=np.nan
    out=m.surrogate_audit(a,b,CFG)
    assert out["qualified"] is False

def test_split_detects_phase_inconsistency():
    x1,y1=synthetic_pair(300,8)
    x2,y2=synthetic_pair(300,25)
    out=m.paired_split_lag(np.r_[x1,x2],np.r_[y1,y2],CFG)
    assert out["halves"][0]["lag_frames"]==8
    assert out["halves"][1]["lag_frames"]==25
    assert out["stable_under_contiguous_split"] is False

def test_half_window_with_sufficient_overlap_returns_two_results():
    a,b=synthetic_pair()
    v=m.paired_split_lag(a,b,CFG)
    assert len(v["halves"])==2
    assert v["same_sign"] is True

def test_freeze_sha_prevents_posthoc_config_change():
    assert m.frozen_config()["max_circular_shift_surrogates"]==127
    assert m.frozen_config()["descriptive_surrogate_exceedance_threshold"]==.05

def test_real_c3d_subjects_and_no_grf_or_policy_outputs():
    src=Path("/home/you/projects/kinetickick-wt-v116")
    if not (src/"reports/v116_human_qualification.audit.json").exists():
        pytest.skip("original C3D not mounted")
    out=m.run(CFG)
    assert out["raw_trials"]==6 and out["independent_subjects"]==2
    assert out["trial_pair_tests"]==12
    assert out["physical_motor_command_issued"] is False
    assert not out["subject_disjoint_train_val_test_possible"]
    assert out["true_impact_labels"]==0
    assert out["new_real_G1_physx_episodes"]==0
