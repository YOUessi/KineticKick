"""v72 fail-closed smoke shutdown, 200Hz post-event and human optical tests.

No Isaac imports or GPU. Real raw and six optical source tests are read-only.
"""
import json, hashlib
from pathlib import Path
import numpy as np
import pytest

from v072_isolated_runtime_probe import classify, LOG, RESULT
from v072_measured_support_coupling import safe_rms,lagged_signed_cov,window_stats,ROOT as ROOT72
from v072_stage_qualification import stage,validate_smoke,OUT as LIFECYCLE
from v072_cmu_geometry_qualification import geom_candidates, OUT as C3D_REPORT

SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

@pytest.mark.parametrize("s,rc,timeout,expected",[
    ("KINETICKICK_ISAAC_SMOKE_OK",0,False,"ISAAC_CONTEXT_EIGHT_STEPS_VALID"),
    ("KINETICKICK_ISAAC_SMOKE_OK",124,True,"ISAAC_EIGHT_STEPS_DONE_UNCLEAN_SHUTDOWN_OR_FINALIZATION_NOT_G1"),
    ("fatal init",124,True,"ISAAC_BOOT_OR_SIM_STEP_TIMEOUT_NOT_G1"),
    ("fatal init",2,False,"PYTHON_OR_APP_LAUNCHER_BEFORE_KIT"),
    ("[INFO][AppLauncher] error",1,False,"ISAAC_CONTEXT_EXIT_NONZERO_NOT_G1"),
])
def test_probe_exit_code_not_masquerade_as_soccer(s,rc,timeout,expected):
    assert classify(s,rc,timeout)==expected

def test_shutdown_timeout_cannot_be_labeled_normal_exit():
    meta=json.loads(RESULT.read_text());s=LOG.read_text()
    assert SHA(LOG)==meta["raw_log_sha256"]
    state=validate_smoke(meta,s)
    assert state["smoke_finished_eight_steps"]
    assert not state["subprocess_exit_ok"]
    assert state["g1_episodes"]==0
    assert "SHUTDOWN_TIMEOUT" in state["diagnosis"] or "APP_CLOSE_TIMEOUT" in state["diagnosis"]

@pytest.mark.parametrize("raw,expected",[
    ("GLFW initialization failed\nV058_NATIVE_RETURN cuda:0\n\tEnvironment device    : cuda:0",True),
    ("GLFW initialization failed\nV058_NATIVE_START cuda:0",False),
    ("KINETICKICK_ISAAC_SMOKE_OK",False),
])
def test_app_warning_never_proves_G1(raw,expected):
    assert stage(raw,0)["g1_native_return"]==expected

def test_stage_warnings_do_not_prove_failure():
    x=stage("GLFW initialization failed\nDriverShaderCacheManager",124)
    assert x["glfw_warn"] and x["shader_warn"]
    assert not x["g1_native_return"]

def test_historical_staging_integration():
    c=json.loads(LIFECYCLE.read_text())
    assert c["v70_earlier_jobs"]==12
    assert c["v70_timeout_without_original_G1"]==12
    assert c["v70_native_G1_valid"]==0
    assert c["historical_v67_reference_log_stage"]["glfw_warn"]
    assert c["historical_v67_reference_log_stage"]["g1_native_return"]

@pytest.mark.parametrize("data",[[],[float("nan")],[float("inf")]])
def test_no_unmeasured_rms(data):
    with pytest.raises(ValueError):
        safe_rms(data)

def test_support_stats_minimal_29dof():
    def frame(t):
        return dict(time_s=t, joint_vel=list(range(29)),
                    support_foot_force_N={"left":90.,"right":10.},
                    whole_body_com_xyz=[0.1,0.,.65],nonfoot_contact_N=0.)
    rows=[frame(t*.005) for t in range(30)]
    x=window_stats(rows,.03,.08)
    assert x["actual_rows"]==11
    assert x["left_support_fraction_mean"]==pytest.approx(.9)
    assert x["measured_rms_qdot_rad_s"]["pelvis_waist"]>0
    assert x["minimum_wrist_hip_origin_gap_m"] is None

def test_support_stats_missing_com_fails():
    z=dict(time_s=0.005,joint_vel=[0.]*29,
           support_foot_force_N={"left":100.,"right":0.},
           whole_body_com_xyz=[float('nan'),0,0],nonfoot_contact_N=0.)
    with pytest.raises(ValueError):
        window_stats([z],0,.02)

def test_kinematic_lag_requires_enough_frames():
    assert lagged_signed_cov([1.,2.,3.],[3.,2.,1.]) is None

def test_kinematic_lag_no_causal_claim():
    x=lagged_signed_cov(np.sin(np.linspace(0,30,150)),np.sin(np.linspace(0,30,150)))
    assert len(x)==5 and x[2]["corr"]==pytest.approx(1.,abs=1e-9)

def test_optical_rejects_missing_ball_and_foot():
    with pytest.raises(ValueError):
        geom_candidates(np.zeros((4,7,50)),["foo"]*7,120)

def test_optical_valid_two_ball_markers_not_physical_contact():
    names=["soccerball:a","soccerball:b","soccerball:c",
           "RTOE","RMT5","RHEE","LTOE","LMT5","LHEE"]
    pts=np.ones((4,9,50),dtype=float)
    pts[:3,:,:]=100
    pts[3,:,:]=1
    d=geom_candidates(pts,names,120)
    assert d["coverage_2_ball_marker_fraction"]==1
    assert d["top_optical_candidates"]
    assert not d["top_optical_candidates"][0]["verified_impact"]

@pytest.mark.parametrize("bad",["nan","inf"])
def test_optical_nan_mask_and_units(bad):
    names=["soccerball:a","soccerball:b","soccerball:c",
           "RTOE","RMT5","RHEE","LTOE","LMT5","LHEE"]
    pts=np.ones((4,9,50),dtype=float)*100
    pts[3]=1.
    pts[:3,0,:]=float(bad)
    d=geom_candidates(pts,names,120)
    assert d["coverage_2_ball_marker_fraction"]==1

def test_original_nine_g1_ledger_is_postregistered_only():
    p=ROOT72/"reports/v072_measured_200hz_wholebody_coupling.json"
    d=json.loads(p.read_text())
    assert d["episodes_original"]==9
    assert d["episodes_new"]==0
    assert len(d["cases"])==9
    assert d["strict_partial_gate_count_not_full_qualification"]==4
    assert not any(c["original_contact_pair_directly_certified"] for c in d["cases"])
    assert sum(c["original_physics_rows"] for c in d["cases"])==15728

def test_early_term_sentinel_not_full_recovery():
    d=json.loads((ROOT72/"reports/v072_measured_200hz_wholebody_coupling.json").read_text())
    x=[v for v in d["cases"] if v["filename"].endswith("c_v32.json")][0]
    assert x["terminal_extra_sample_1us"] is True
    assert x["unique_physical_200hz_steps"]==328
    assert x["actual_end_s"]<2
    assert x["pre_impact"] is None and x["missing_contact_not_imputed"]

def test_b_instep_failure_is_not_hard_gate_pass():
    d=json.loads((ROOT72/"reports/v072_measured_200hz_wholebody_coupling.json").read_text())
    x=[v for v in d["cases"] if v["filename"].endswith("b_synth.json")][0]
    assert x["original_record_goal"] is True
    assert x["original_record_inst_fraction"]<.95
    assert x["observed_14s"] is True

def test_six_c3d_only_geometry_and_subject_split():
    x=json.loads(C3D_REPORT.read_text())
    assert x["six_sha_locked_original_optical_clips"]==6
    assert x["distinct_subject_groups"]==["10","11"]
    assert x["observed_force_calibrated_clips"]==0
    assert all(not c["human_surface_contact_force_groundtruth"] for c in x["clips"])
    assert all(not r["verified_impact"] for c in x["clips"] for r in c["top_optical_candidates"])
    assert any(c["top_optical_candidates"][0]["minimum_point_distance_m"]<.01 for c in x["clips"])

def test_immutable_source_hashes_match_6_c3d():
    x=json.loads(C3D_REPORT.read_text())
    root=Path("/home/you/projects/kinetickick/data/research_human_kicks/cmu_soccer_120hz_v050/raw")
    for d in x["clips"]:
        path=root/d["subject_group"]/d["name"]
        assert SHA(path)==d["sha256"]
