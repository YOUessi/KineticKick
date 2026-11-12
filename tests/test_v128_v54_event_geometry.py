#!/usr/bin/env python3
"""v128 contract, original-G1 logging, sensor ambiguity, and adversarial CPU tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))

import v128_v54_event_geometry as a


@pytest.fixture
def cfg():
    return a.frozen_protocol()


def fake_episode(n=8):
    rows=[]
    for i in range(n):
        rows.append({
            "time_s": .005*(i+1),
            "joint_pos":[0.]*29,
            "joint_vel":[0.]*29,
            "whole_body_com_xyz":[0.,0.,.7],
            "support_foot_force_N":{"left":100., "right":50.},
            "nonfoot_contact_N":0.,
            "terminated":False,
        })
    return dict(joint_names=list(a.G1_JOINT_ORDER),
                root_fixed=False,seed=1201,post_initialization_state_writes=False,
                traces=[rows],joint_velocity_limits=[[32.]*29],
                records=[{"contact_time_s":1.57}])


def test_frozen_pre_registered_sha_and_safety(cfg):
    assert cfg["parent_commit"].startswith("e435616")
    assert cfg["native_contact_pair_certified"] is False
    assert cfg["no_unobserved_event_extrapolation"] is True
    assert cfg["capsule_clearance_thresholds_m"]==[0.0,.02]
    assert len(cfg["input_sha256"])==5


def test_frozen_config_changed_refused(monkeypatch,cfg,tmp_path):
    altered=tmp_path/"cfg.json"
    altered.write_text(json.dumps(cfg))
    monkeypatch.setattr(a,"PROTOCOL",altered)
    with pytest.raises(ValueError,match="FROZEN_PROTOCOL_DRIFT"):
        a.frozen_protocol()


def test_sha_fails_closed(tmp_path,cfg):
    p=tmp_path/"bad.json";p.write_text("{}")
    with pytest.raises(ValueError,match="SOURCE_SHA256_DRIFT"):
        a.require_sha(p,cfg["input_sha256"]["v054_original_named_nonfoot_audit.json"])


def test_symlink_source_refused(tmp_path):
    p=tmp_path/"true";p.write_text("sample")
    other=tmp_path/"link";other.symlink_to(p)
    with pytest.raises(ValueError,match="SYMLINK"):
        a.require_sha(other,a.sha256(p))


def test_good_original_29dof_schema():
    d=a.validate_rows(fake_episode(),8,.005)
    assert d["q"].shape==(8,29)
    assert d["terminal_sentinel"]==0
    assert d["original_logged_rows"]==8


@pytest.mark.parametrize("mutate,reason", [
    (lambda d:d.update(root_fixed=True),"ORIGINAL_SEED_OR_ROOT"),
    (lambda d:d.update(seed=3),"ORIGINAL_SEED_OR_ROOT"),
    (lambda d:d.update(joint_names=list(d["joint_names"])[:-1]),"ORIGINAL_G1_29DOF"),
    (lambda d:d.update(post_initialization_state_writes=True),"STATE_TAMPERING"),
    (lambda d:d["traces"][0][2].update(time_s=.010),"NOT_ORIGINAL_200HZ"),
    (lambda d:d["traces"][0][2].update(joint_pos=[0.]*28),"INVALID_REAL"),
    (lambda d:d["traces"][0][2].update(joint_vel=[float("nan")]*29),"INVALID_REAL"),
    (lambda d:d["traces"][0][2].update(nonfoot_contact_N=float("inf")),"INVALID_REAL"),
    (lambda d:d.update(joint_velocity_limits=[[0.]*29]),"INVALID_REAL"),
    (lambda d:d["records"][0].update(contact_time_s=99.0),"CONTACT_TIME_DRIFT"),
])
def test_original_physics_corruption_fails(mutate,reason):
    d=fake_episode()
    mutate(d)
    with pytest.raises((ValueError,TypeError),match=reason):
        a.validate_rows(d,8,.005)


def test_premature_termination_sentinel_distinguished_from_physics():
    d=fake_episode(1028)
    sentinel=copy.deepcopy(d["traces"][0][-1])
    sentinel.update(time_s=5.140001,terminated=True)
    d["traces"][0].append(sentinel)
    x=a.validate_rows(d,1029,.005)
    assert x["t"][-1]==5.14
    assert len(x["q"])==1028
    assert x["terminal_sentinel"]==1
    assert x["original_logged_rows"]==1029


@pytest.mark.parametrize("field,value",[
    ("joint_pos",[.1]*29), ("time_s",5.145), ("terminated",False),
    ("nonfoot_contact_N",25.), ("whole_body_com_xyz",[0,0,0]),
])
def test_forged_or_misaligned_terminal_sentinel_refused(field,value):
    d=fake_episode(1028)
    sentinel=copy.deepcopy(d["traces"][0][-1])
    sentinel.update(time_s=5.140001,terminated=True)
    sentinel[field]=value
    d["traces"][0].append(sentinel)
    with pytest.raises(ValueError,match="TERMINATION_SENTINEL"):
        a.validate_rows(d,1029,.005)


def test_observed_last_state_but_missing_event_state():
    t=np.arange(1,1029)*.005
    r=a.event_observability(t,5.144999885,.005)
    assert not r["event_has_directly_sampled_joint_state"]
    assert r["last_pre_event_time_s"]==pytest.approx(5.14)
    assert r["about_one_200hz_step_coverage_gap"]
    assert not r["collision_at_unsampled_event_certified"]


def test_observed_event_state_still_not_contact_truth():
    t=np.arange(1,1030)*.005
    r=a.event_observability(t,5.145,.005)
    assert r["event_has_directly_sampled_joint_state"]
    assert not r["collision_at_unsampled_event_certified"]


@pytest.mark.parametrize("times", [
    [0.005,0.005], [0.01,0.005], [float("nan"),0.01], [],
])
def test_unqualified_time_grid_rejected(times):
    with pytest.raises(ValueError,match="INVALID_MONOTONIC_TIME"):
        a.event_observability(np.asarray(times),.02,.005)


def test_no_prior_state_refused():
    with pytest.raises(ValueError,match="NO_PRIOR"):
        a.event_observability(np.array([.02,.025]),.01,.005)


def test_independent_phase_metric_preserves_qdot_and_support():
    t=np.array([.005,.01,.015])
    gaps=np.array([.04,.019,.001])
    dq=np.zeros((3,29));dq[2,4]=33.
    v=a.slice_stat(t,np.array([True,True,True]),gaps,dq,
                   np.ones(29)*32,np.array([0.,0.,25.]),np.zeros((3,3)),
                   np.array([[10,20]]*3,dtype=float))
    assert v["nonpositive_gap_frames"]==0
    assert v["inside_20mm_alert_frames"]==2
    assert v["minimum_capsule_gap_m"]==pytest.approx(.001)
    assert v["original29_qdot_exceedance_frames"]==1
    assert v["nonfoot_net_force_peak_N"]==25.
    assert v["left_foot_ground_force_proxy_integral_Ns"]==pytest.approx(.15)
    assert not v["native_actor_collider_manifold_certified"]


def test_positive_capsule_gap_only_excludes_model_overlap(cfg):
    assert cfg["force_pair_not_manifold_proof"] is True
    assert cfg["no_runtime_USD_collider_inference"] is True


def test_original_v54_real_data_three_cases_cpu_integration(cfg):
    # Requires original source JSON+URDF; freeze SHA ensures exact data provenance.
    out=a.run(cfg)
    assert out["original_total_logged_rows_including_terminal_marker"]==6629
    assert out["original_total_observed_physics_steps"]==6628
    assert out["new_G1_PhysX_episodes"]==0
    assert out["v128_independently_rechecked_v52_vs_v54_all_physical_fields"]
    assert len(out["v128_exact_common_physical_field_names"])==9
    assert out["original_reported_success_in_these_cases"]==2
    assert out["independent_all_original_hard_gates_reaudited"] is False
    fail=out["per_case"]["cmu_10_05"]
    assert fail["nonphysics_termination_marker_rows"]==1
    assert fail["observed_physical_frames"]==1028
    assert fail["source_nonfoot_net_force_peak_time_s"]==pytest.approx(5.135)
    assert fail["minimum_URDF_gap_time_s"]==pytest.approx(5.135)
    assert fail["geometry_minimum_and_net_force_trace_peak_coincide_at_same_step"]
    assert fail["URDF_gap_at_logged_trace_nonfoot_peak_m"] < .001
    assert fail["named_sensor_event_minus_trace_force_peak_s"]==pytest.approx(.01,abs=2e-5)
    assert not fail["directly_sampled_at_nonfoot_sensor_event"]
    assert not out["physical_motor_command_issued"]
    assert out["no_contact_pair_manifold_collider_paths_truth"]
    for base in ("frozen_v32","cmu_10_02"):
        assert out["per_case"][base]["observed_physical_frames"]==2800
        assert out["original_failed_case_5p14s_vs_matched_control"][base]["baseline_continues_to_14s"]
