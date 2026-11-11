"""v79 safety: freeze provenance, fail closed, native callback cannot invent proof."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from tools import v079_native_pair_batch as m

def test_frozen_config_precedes_gpu():
    c=m.read_config()
    assert m.sha(m.CONF)==m.CONF_SHA
    assert c["version"]=="v0.1.79"
    assert len(c["trial_order"])==3
    assert [r["case"] for r in c["trial_order"]]==["original_v32","cmu_10_02","cmu_10_05"]
    assert c["observer_modes_order"]==["off","on"]
    assert c["minimum_free_gib"]>=20

def test_controller_source_and_original_physics_pinned():
    c=m.read_config()
    assert all(m.sha(m.ROOT/p)==h for p,h in c["source_sha256"].items())
    assert all(m.sha(m.ORIG/p)==h for p,h in c["original_v054_sha256"].items())

def test_hard_gates_cannot_be_lowered():
    h=m.read_config()["hard_gates"]
    assert h["instep_impulse_fraction_min"]>=.95
    assert h["ball_speed_mps_proxy_min"]>=3.5
    assert h["nonfoot_force_max_N"]<=20
    assert h["qdot_ratio_max"]<=1.000001
    assert h["full_duration_min_s"]>=13.995

def test_native_no_subscription_not_negative_contact():
    out=m.native_summary(None)
    assert out["wrist_hip_direct_contact_certified"] is False
    assert out["status"]=="NOT_PRESENT"

def test_net_sensor_forces_not_valid_native_manifold():
    report=dict(native_headers=2,callback_calls=2,
        report_api_present_on_original_bodies={},saved_interesting_events=[],
        wrist_hip_direct_path_events=0,report_capture_complete=False,
        dropped_events=0,errors=[])
    value=m.native_summary(report)
    assert not value["wrist_hip_manifold_candidate"]
    assert not value["original_prim_report_api_present"]
    assert value["status"]=="CALLBACK_UNVERIFIED_UNTIL_PARITY"

def test_on_off_exact_mismatch_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"ROOT",tmp_path)
    c=dict(parity_keys=["traces","records"],run_namespace="v079_test")
    off=dict(raw_sha256="off",independent_hard_gates={"hard_gate_pass":False},
             original_g1_full_physics_certified=True)
    on=dict(raw_sha256="on",independent_hard_gates={"hard_gate_pass":False},
            original_g1_full_physics_certified=True,
            native_callback={"wrist_hip_manifold_candidate":True,
                "original_prim_report_api_present":True,"native_complete":True,
                "errors":[]})
    (tmp_path/"reports").mkdir()
    p=m.compare(c,{"case":"mismatch","seed":1201},off,on,
                {"traces":[1],"records":[1]},{"traces":[2],"records":[1]})
    assert p["exact_mismatched_keys"]==["traces"]
    assert not p["shared_physical_fields_exactly_equal"]
    assert not p["original_readonly_native_wrist_hip_collider_path_certified"]

def test_pair_positive_requires_complete_direct_callback_and_exact_parity(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"ROOT",tmp_path)
    (tmp_path/"reports").mkdir()
    gates={"hard_gate_pass":False}
    off={"raw_sha256":"off","independent_hard_gates":gates,
         "original_g1_full_physics_certified":True}
    on={"raw_sha256":"on","independent_hard_gates":gates,
        "original_g1_full_physics_certified":True,
        "native_callback":{"wrist_hip_manifold_candidate":True,
        "original_prim_report_api_present":True,"native_complete":True,"errors":[]}}
    result=m.compare({"parity_keys":["traces","records"],"run_namespace":"v079_test"},
        {"case":"pair","seed":1201},off,on,
        {"traces":[1,2],"records":[{"x":1}]},{"traces":[1,2],"records":[{"x":1}]})
    assert result["shared_physical_fields_exactly_equal"]
    assert result["original_readonly_native_wrist_hip_collider_path_certified"]

def test_missing_direct_native_api_remains_unproved(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"ROOT",tmp_path)
    (tmp_path/"reports").mkdir()
    gates={"hard_gate_pass":True}
    off={"raw_sha256":"off","independent_hard_gates":gates,
         "original_g1_full_physics_certified":True}
    on={"raw_sha256":"on","independent_hard_gates":gates,
        "original_g1_full_physics_certified":True,
        "native_callback":{"wrist_hip_manifold_candidate":True,
        "original_prim_report_api_present":False,"native_complete":True,"errors":[]}}
    result=m.compare({"parity_keys":["traces"],"run_namespace":"v079_test"},{"case":"noapi","seed":1201},
                     off,on,{"traces":[1]},{"traces":[1]})
    assert not result["original_readonly_native_wrist_hip_collider_path_certified"]
    assert result["native_negative_interpretation"]=="UNKNOWN"

def test_write_once_guarded_by_existing_path(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"ROOT",tmp_path)
    (tmp_path/"reports").mkdir()
    (tmp_path/"reports"/"v079_original_g1_native_pair_trial_off.json").write_text("evidence")
    with pytest.raises(FileExistsError,match="WRITE_ONCE"):
        m.one({"run_namespace":"v079_original_g1_native_pair"},
              {"case":"trial","seed":1201,"source_config":"frozen"},"off")

