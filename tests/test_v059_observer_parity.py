"""v059 CPU native-subscriber physical-parity and no-sample-fabrication tests."""
import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from audit_v059_observer_parity import compare_payload,compare_raw,DYNAMIC_FIELDS
from run_backend_batch_v059 import command

def trace():
    return {k:None for k in DYNAMIC_FIELDS} | {
        "time_s":0.005,"joint_pos":[0.]*29,"joint_vel":[0.]*29,
        "root_xyz":[0.,0.,.7],"root_tilt_deg":0.,"support_foot_force_N":{"left":25.,"right":80.},
        "ball_xyz":[0.,-4.,.1],"terminated":False}

def frame():
    return {"traces":[[trace(),trace()]]}

def test_same_backend_identical_all_25_fields():
    a=frame()
    b=copy.deepcopy(a)
    r=compare_raw(a,b)
    assert r["matches_at_abs_tol_1e_6"] and r["common_field_count"]==len(DYNAMIC_FIELDS)

def test_reporter_metadata_is_not_physics():
    a=frame();b=copy.deepcopy(a)
    b["physx_native_report_v056"]={"saved_interesting_events":[{"sample":1}]}
    assert compare_raw(a,b)["matches_at_abs_tol_1e_6"]

def test_changed_original_joint_physics_reject_parity():
    a=frame();b=copy.deepcopy(a)
    b["traces"][0][0]["joint_vel"][4]=.0005
    r=compare_raw(a,b)
    assert not r["matches_at_abs_tol_1e_6"]
    assert r["max_abs_numeric_difference"]==pytest.approx(.0005)

def test_changed_support_and_contacts_reject_parity():
    a=frame();b=copy.deepcopy(a)
    b["traces"][0][0]["support_foot_force_N"]["left"]=30.
    b["traces"][0][1]["contacts"]=[{"different":"body"}]
    r=compare_raw(a,b)
    assert r["mismatch_count"]>=2

def test_different_physics_length_reject_parity():
    a=frame();b=copy.deepcopy(a)
    b["traces"][0].append(trace())
    assert not compare_raw(a,b)["matches_at_abs_tol_1e_6"]

def test_numeric_tolerance_applies_only_to_tiny_roundoff():
    a=frame();b=copy.deepcopy(a)
    b["traces"][0][0]["joint_vel"][4]=.0000005
    assert compare_raw(a,b)["matches_at_abs_tol_1e_6"]

def test_different_boolean_terminal_reject_parity():
    a=frame();b=copy.deepcopy(a)
    b["traces"][0][1]["terminated"]=True
    assert not compare_raw(a,b)["matches_at_abs_tol_1e_6"]

def test_same_length_different_raw_capture_timestamps_reject_parity():
    a=frame();b=copy.deepcopy(a)
    b["traces"][0][1]["time_s"]=.01
    assert not compare_raw(a,b)["matches_at_abs_tol_1e_6"]

def test_v059_no_observer_does_not_request_native_pair():
    from run_backend_batch_v059 import ROOT
    c={"policies":{"baseline":"configs/does_not_need_to_exist.json"}}
    k={"policy":"baseline","observer":"none","backend":"cpu","seed":1201,
       "perturb_reset":False,"output":"reports/test_output_unique.json"}
    s=command(c,k)
    assert s.count("--sim-device")==1 and s[s.index("--sim-device")+1]=="cpu"
    assert "--native-readonly-pair-v056" not in s
    assert "--pair-dynamics-v055" not in s

def test_v059_full_observer_and_perturbation_forwarded():
    c={"policies":{"baseline":"configs/does_not_need_to_exist.json"}}
    k={"policy":"baseline","observer":"full","backend":"cuda:0","seed":2333,
       "perturb_reset":True,"output":"reports/test_output_unique.json"}
    s=command(c,k)
    assert "--native-readonly-pair-v056" in s
    assert "--pair-dynamics-v055" in s
    assert "--perturb-reset-v056" in s
    assert s[s.index("--sim-device")+1]=="cuda:0"

def test_key_set_changes_are_non_parity():
    r=compare_payload({"root":0., "extra":0.},{"root":0.})
    assert not r["matches_at_abs_tol_1e_6"]
