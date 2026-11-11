from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

m = module("tools/v077_kinematic_coupling_audit.py","v077c")
g = module("tools/v077_g1_liveness_batch.py","v077g")

def test_signed_lag_perfect_identity():
    t=np.linspace(0,5,120)
    x=np.sin(t)
    assert m.corr_lags(x,x)["0"] == 1.0

def test_lag_zero_variance_unknown():
    x=np.ones(100)
    assert all(v is None for v in m.corr_lags(x,x).values())

def test_lags_short_window_unknown():
    x=np.ones(20)
    assert all(v is None for v in m.corr_lags(x,x).values())

def test_classify_no_output_not_a_shot_failure():
    assert g.classify("Actor RNN: ready",False,True)["outcome"]=="NO_CERTIFIED_EPISODE"

def test_isaac_environment_marker_without_physics_not_certified():
    c=g.classify("Completed setting up the environment\nActor RNN:",False,True)
    assert c["markers"]["isaac_environment_constructed"]
    assert not c["markers"]["original_g1_report_marker"]
    assert c["outcome"]=="NO_CERTIFIED_EPISODE"

def test_real_g1_requires_29_joint_names(tmp_path):
    path=tmp_path/"a.json"
    path.write_text('{"traces":[[{"time_s":0.0}]],"joint_names":[1,2],"records":[{}]}')
    passed,why=g.is_valid_g1_report(path)
    assert not passed
    assert why["joint_count"]==2

def test_missing_g1_result_is_unknown(tmp_path):
    passed,why=g.is_valid_g1_report(tmp_path/"nonexistent")
    assert not passed
    assert why["reason"]=="NO_PHYSICS_REPORT"

def test_bad_200hz_timestamp_rejected():
    with pytest.raises(ValueError,match="29DOF_REQUIRED"):
        m.audit_trace({"joint_names":[]},{"hard_gate_pass":False})

def test_strict_source_consistency():
    r=m.run()
    assert r["original_gpu_g1_attempts"]==9
    assert r["historical_strict_pass"]==4
    assert r["historical_complete_14s"]==5
    assert r["contact_window_0p4_pre_1p25_post_valid"]==6
    assert len({v["original_sha256"] for v in r["cases"]})==9
    assert all(v["original_29_qdot_not_exceeding_limit"] for v in r["cases"])
    b=next(v for v in r["cases"] if v["case"]=="new_stance_b_synth")
    assert b["original_goal"]
    assert not b["instep_95pct_gate"]
    assert not b["authoritative_independent_all_hard_gate_pass"]

def test_all_censored_cases_do_not_mint_windows():
    r=m.run()
    censored=[v for v in r["cases"] if not v["contact_plus_1p25_observed"]]
    assert len(censored)==3
    assert all(v["windows"] == {} for v in censored)
