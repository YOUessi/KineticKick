"""v130 frozen original-G1 force/proxy/COM/29DoF workload safety tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import v130_real_com_support_work as m


@pytest.fixture
def cfg():
    return m.frozen()


def test_frozen_source_contract(cfg):
    assert cfg["version"]=="v0.1.130"
    assert cfg["native_physx_contact_pair_certified"] is False
    assert cfg["time_shift_negative_control_lags_frames"]==[0,40,80]
    assert cfg["no_motor_command_or_model_weight_changes"]


def test_reject_modified_pre_registration(tmp_path, monkeypatch):
    fake=tmp_path/"tampered.json"
    fake.write_text(m.CONFIG.read_text().replace('"dt_s": 0.005','"dt_s": 0.010'))
    monkeypatch.setattr(m, "CONFIG", fake)
    with pytest.raises(ValueError,match="PROTOCOL_DRIFT"):
        m.frozen()


@pytest.mark.parametrize("total,expected_valid", [(0,False),(49,False),(50,True),(600,True)])
def test_force_proxy_minimum(cfg,total,expected_valid):
    arr=np.array([[total,0]],float)
    _,valid,_,dom=m.load_shares(arr,cfg)
    assert bool(valid[0])==expected_valid
    assert int(dom[0])==(1 if expected_valid else 0)


def test_swap_foot_side_changes_dominance_not_total(cfg):
    arr=np.array([[200,2],[0,100],[40,40],[0,0]])
    total,valid,share,dom=m.load_shares(arr,cfg)
    total2,valid2,share2,dom2=m.load_shares(arr[:,::-1],cfg)
    assert np.array_equal(total,total2)
    assert np.array_equal(valid,valid2)
    assert np.array_equal(dom2,-dom)
    assert np.isnan(share[-1]) and np.isnan(share2[-1])


@pytest.mark.parametrize("bad", [np.array([[np.nan,4.]]),np.array([[-1.,2.]]),
                                   np.array([[float("inf"),0.]]),np.array([1,2])])
def test_force_rejects_invalid(cfg,bad):
    with pytest.raises(ValueError,match="FOOT_FORCE"):
        m.load_shares(bad,cfg)


def test_threshold_is_strict_80_percent(cfg):
    force=np.array([[80,20],[80.001,19.999],[19.999,80.001]],float)
    _,_,_,dominant=m.load_shares(force,cfg)
    assert dominant.tolist()==[0,1,-1]


@pytest.mark.parametrize("num", [280,350,500])
def test_disallow_missing_200hz_impact_windows(cfg,num):
    t=np.arange(1,num+1)*.005
    with pytest.raises(ValueError,match="MISSING_PRESCRIBED_IMPACT_WINDOWS"):
        m.windows(t,1.57,cfg)


def test_frozen_80_and_250_samples(cfg):
    t=np.arange(1,2801)*.005
    w=m.windows(t,1.57,cfg)
    assert [int(w[k].sum()) for k in ["pre_400ms","post_1250ms"]]==[80,250]
    assert all(not np.any(w["pre_400ms"]&w[k]) for k in ["post_1250ms","observed_late_recovery"])


def test_200hz_time_discontinuity_fails(cfg):
    t=np.arange(1,2801)*.005
    t[245]=t[244]
    with pytest.raises(ValueError,match="UNCERTIFIED_OR_NONCONTIGUOUS"):
        m.windows(t,1.57,cfg)


@pytest.mark.parametrize("x,y,expected", [
    ([1,2,3,4],[3,6,9,12],1.0),
    ([1,2,3,4],[12,9,6,3],-1.0),
])
def test_pearson_signed(x,y,expected):
    assert m.pearson(x,y)==pytest.approx(expected)


def test_pearson_undefined_does_not_default_to_zero():
    assert m.pearson(np.ones(8),np.arange(8)) is None


def test_pearson_invalid_shapes():
    with pytest.raises(ValueError,match="CORRELATION"):
        m.pearson(np.arange(4),np.arange(5))


def make_synthetic_arrays(n=250):
    t=np.arange(1,n+1)*.005
    vel=np.tile(np.arange(29)/100.0,(n,1))
    tau=np.tile((np.arange(29)-10)/10.0,(n,1))
    com=np.stack([t,t*0,t*.1],axis=1)
    com_vel=np.stack([np.ones(n),t*0,0.1+0.01*np.sin(t*3)],axis=1)
    foot=np.stack([100.+40.*np.sin(4*t), 100.-40.*np.sin(4*t)],axis=1)
    nonfoot=np.zeros(n)
    return t,vel,tau,com,com_vel,foot,nonfoot


def test_signed_work_and_negative_energy_preserved(cfg):
    t,vel,tau,com,com_vel,force,nonfoot=make_synthetic_arrays()
    stats=m.interval_stats(t,vel,tau,com,com_vel,force,nonfoot,
                            np.ones(len(t),bool),cfg)
    expected=float((vel*tau).sum()*.005)
    assert stats["total_signed_joint_motor_work_proxy_J"]==pytest.approx(expected)
    assert stats["total_positive_joint_motor_work_proxy_J"]>0
    assert stats["total_negative_joint_motor_work_proxy_J"]<0
    assert stats["native_physx_foot_pair_or_GRF_certified"] is False
    assert stats["support_polygon_or_CoP_available"] is False
    assert set(stats["com_vertical_acceleration_vs_total_support_proxy_shifted_correlation"])=={"0","40","80"}


def test_joint_swap_reallocates_work_but_keeps_total(cfg):
    t,v,tau,com,com_v,f,nf=make_synthetic_arrays()
    a=m.interval_stats(t,v,tau,com,com_v,f,nf,np.ones(len(t),bool),cfg)
    b=m.interval_stats(t,v[:,::-1],tau[:,::-1],com,com_v,f,nf,
                       np.ones(len(t),bool),cfg)
    assert b["total_signed_joint_motor_work_proxy_J"]==pytest.approx(
        a["total_signed_joint_motor_work_proxy_J"])
    assert b["max_joint_power_joint"] != a["max_joint_power_joint"]


def test_real_v54_sha_and_censoring_on_3_physical_traces():
    result=m.audit()
    assert result["verified_original_physx_episodes"]==3
    assert result["physical_200hz_frames"]==6628
    assert result["complete_14s_episodes"]==2
    assert result["censored_failures"]==1
    fail=next(x for x in result["evidence"] if x["case"]=="cmu_10_05")
    assert fail["original_physics_frames"]==1028
    assert fail["nonphysics_sentinel_excluded"]==1
    assert fail["original_200hz_end_s"]==pytest.approx(5.14)
    assert fail["phases"]["pre_400ms"]["physical_200hz_frames"]==80
    assert fail["phases"]["post_1250ms"]["physical_200hz_frames"]==250
    assert not fail["complete_14_second_recovery_observed"]
    assert fail["native_colliders_and_human_contact_truth"]=="UNKNOWN"
    assert result["physical_actuation_or_training_performed"] is False


def test_pre_400ms_80_step_shift_is_not_independent_null(cfg):
    t,v,tau,com,com_v,f,nf=make_synthetic_arrays(80)
    stats=m.interval_stats(t,v,tau,com,com_v,f,nf,np.ones(80,bool),cfg)
    assert stats["invalid_circular_null_lags_aliasing_zero"]==[80]
    assert stats["com_vertical_acceleration_vs_total_support_proxy_shifted_correlation"]["80"] is None


def test_unknown_foot_force_gap_is_not_direct_support_transfer(cfg):
    t,v,tau,com,com_v,f,nf=make_synthetic_arrays(30)
    f[:10]=[120,0]
    f[10:20]=[0,0]
    f[20:]=[0,120]
    s=m.interval_stats(t,v,tau,com,com_v,f,nf,np.ones(30,bool),cfg)
    assert s["direct_consecutive_side_proxy_switches"]==0
    assert s["side_proxy_switches_across_unknown_gap_not_certified"]==1
