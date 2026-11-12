"""v124 real-120Hz intrinsic human marker qualification and inference limits."""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v124_human_relative_coordination as m
CFG=json.loads(m.CONFIG.read_text())

def scene(n=160):
    t=np.arange(n)/120.
    a={
        "LBWT":(-.16,0.,1.),"RBWT":(.16,0.,1.),
        "LSHO":(-.22,0.,1.4),"RSHO":(.22,0.,1.4),
        "LWRA":(-.44,.12,1.22),"RWRA":(.44,.12,1.22),
        "LANK":(-.14,0.,.12),"RANK":(.14,0.,.12),
        "LTOE":(-.13,.23,.04),"RTOE":(.13,.23,.04),
        "LTHI":(-.14,0.,.8),"RTHI":(.14,0.,.8)
    }
    out={k:np.tile(np.array(v,dtype=float),(n,1)) for k,v in a.items()}
    out["LWRA"][:,1] += .06*np.sin(t*7.)
    out["RTOE"][:,1] += .09*np.sin(t*7.-.18)
    return out

def fake_c3d(n=160):
    data=scene(n)
    pts=np.zeros((4,len(m.NAMES),n))
    for i,name in enumerate(m.NAMES):
        pts[:3,i]=data[name].T*1000
    return {"parameters":{"POINT":{"LABELS":{"value":list(m.NAMES)},
                                    "UNITS":{"value":["mm"]}}},
            "header":{"points":{"frame_rate":120.}},
            "data":{"points":pts}}

def test_true_intrinsic_features_invariant_rigid_translation_and_rotation():
    a=scene()
    f1,r1=m.intrinsic_features(a,CFG)
    theta=.9
    rot=np.array([[np.cos(theta),-np.sin(theta),0],
                  [np.sin(theta),np.cos(theta),0],[0,0,1]])
    b={k:val@rot.T+np.array([4.,-3.,8.]) for k,val in a.items()}
    f2,r2=m.intrinsic_features(b,CFG)
    for key in f1:np.testing.assert_allclose(f1[key],f2[key],rtol=0,atol=3e-13)
    for key in r1:np.testing.assert_allclose(r1[key],r2[key],rtol=0,atol=5e-11)

def test_these_are_radial_change_rates_not_full_3d_marker_velocities():
    a=scene()
    rates=m.intrinsic_features(a,CFG)[1]
    assert "LWRA_LSHO" in rates
    assert len(rates["LWRA_LSHO"])==len(a["LWRA"])-1
    assert np.isfinite(rates["LWRA_LSHO"]).all()

def test_marker_missing_residual_masks_invalid_neighbors():
    a=fake_c3d()
    i=m.NAMES.index("LWRA")
    a["data"]["points"][3,i,32]=-1
    vectors,_,fraction=m._vectors_from_c3d(a,CFG)
    assert fraction==pytest.approx(159/160)
    rates=m.intrinsic_features(vectors,CFG)[1]["LWRA_LSHO"]
    assert np.isnan(rates[31:33]).all()

def test_clamped_cosine_and_zero_width_guard():
    a=scene()
    a["LSHO"][:]=a["RSHO"]
    f,_=m.intrinsic_features(a,CFG)
    assert np.isnan(f["shoulder_pelvis_transverse_cos"]).all()

def test_rate_units_sha_and_marker_count_are_not_silently_repaired():
    for field,value,error in [("rate",119.,"UNQUALIFIED_C3D_RATE"),
                              ("units","cm","UNQUALIFIED_C3D_UNITS")]:
        a=fake_c3d()
        if field=="rate":a["header"]["points"]["frame_rate"]=value
        else:a["parameters"]["POINT"]["UNITS"]["value"]=[value]
        with pytest.raises(ValueError,match=error):m._vectors_from_c3d(a,CFG)
    a=fake_c3d()
    a["parameters"]["POINT"]["LABELS"]["value"][0]="DUP"
    with pytest.raises(ValueError,match="MISSING_OR_AMBIGUOUS"):
        m._vectors_from_c3d(a,CFG)

def test_invalid_narrow_frames_or_low_marker_fraction_rejected():
    a=fake_c3d(119)
    with pytest.raises(ValueError,match="TOO_SHORT"):m._vectors_from_c3d(a,CFG)
    a=fake_c3d(160)
    a["data"]["points"][3,:,0:70]=-1
    with pytest.raises(ValueError,match="INSUFFICIENT_VALID_REAL_MARKERS"):
        m._vectors_from_c3d(a,CFG)

def test_adjacent_missing_samples_nan_and_zero_rate_observed():
    a=m.finite_adjacent([1.,2.,float("nan"),5.],120)
    np.testing.assert_allclose(a[0],120.)
    assert np.isnan(a[1]) and np.isnan(a[2])
    np.testing.assert_allclose(m.finite_adjacent([1,1,1]),[0,0])

def test_pair_lag_sign_and_correlated_source_not_causal_proof():
    rng=np.random.default_rng(981)
    a=rng.normal(size=260)
    b=np.concatenate([np.zeros(8),a[:-8]])
    row=m.lag_scan(a,b,120,.35,45)
    assert row["best"]["lag_frames"]==8
    assert row["best"]["correlation"]==pytest.approx(1.)
    null=m.lag_scan(a[::-1],b[::-1],120,.35,45)
    assert null["best"]["lag_frames"]==-8
    assert null["best"]["correlation"]==pytest.approx(1.)

@pytest.mark.parametrize("bad",[
    (np.ones(100),np.ones(100)),(np.zeros(100),np.arange(100)),
    (np.full(100,np.nan),np.arange(100))])
def test_lag_cannot_identify_zero_variance_or_missing(bad):
    assert not m.lag_scan(*bad,120,.35,45)["qualified"]

def test_cross_correlation_requires_overlap_and_equal_shape():
    with pytest.raises(ValueError,match="BAD_PAIR_SHAPE"):
        m.pair_corr(np.ones(20),np.ones(21),0,5)
    assert m.pair_corr(np.arange(20.),np.arange(20.),18,5) is None

def test_qualification_labels_never_impute_professional_force_or_ball_contact():
    row=m.analyze_trial(fake_c3d(),CFG)
    assert row["human_contact_time_label_available"] is False
    assert row["elite_skill_identity_verified"] is False
    assert row["grf_cop_available"] is False
    assert row["full_joint_angular_velocity_or_kinetic_chain_inferred"] is False
    assert len(row["exploratory_opposite_limb_radial_rate_lag"])==2

def test_frozen_configuration_and_no_policy_motor_activation():
    assert CFG["parent_sha"]=="b4da3f1a153a487962b68ddb5e8286a46e100538"
    assert CFG["no_train_or_motor_actuation"]
    assert not CFG["new_physics_permitted_while_GPU_busy"]
    assert len(CFG["human_trials"])==6

def test_original_trial_sha_and_source_provenance_real_if_local(tmp_path):
    source=Path(CFG["human_source_worktree"])
    if not (source/"reports/v116_human_qualification.audit.json").exists():
        pytest.skip("Original ignored C3D data absent in checkout")
    result=m.audit(CFG,source)
    assert result["qualified_real_human_c3d_trials"]==6
    assert result["independent_subjects"]==2
    assert result["cannot_claim_independent_train_validation_test"]
    assert result["marker_frames"]==3358
    assert result["physical_motor_command_issued"] is False
    assert {a["subject"] for a in result["per_trial"]}=={"10","11"}

def test_source_git_sha_drift_refused(tmp_path):
    with pytest.raises(Exception):
        m.audit(CFG,tmp_path)
