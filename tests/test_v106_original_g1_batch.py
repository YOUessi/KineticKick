"""v106 frozen original-G1 physics batch: fail-closed contracts and strict parity."""
import json
import sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/"tools"))
import v106_original_g1_batch as batch

def _row(observer=False):
    return {"name":"L_v32_"+("on" if observer else "off"),"scene":"L",
       "control":"v32","observer":observer,"seed":1601,
       "ball_offset_xy":[-.095,-.035],"velocity_delta":0,
       "certified_original_G1_cuda_200Hz":True}

def test_version_protocol_matrix_is_complete_and_ordered():
    c=json.loads((HERE/"configs/v106_frozen_matrix.json").read_text())
    assert c["version"]=="v0.1.106"
    assert len(c["matrix"])==8
    assert [x["name"] for x in c["matrix"]] == [
        "L_v32_off","L_v32_on","L_cmu02_off","L_cmu02_on",
        "L_cmu05_off","L_cmu05_on","M_v32_off","M_v32_on"]
    assert c["learned_policy_actuation"] is False
    assert c["observer_not_native_manifold"] is True
    assert c["stop_after_first_unverified_or_parity_fail"] is True
    assert c["per_child_watchdog_s"]==140
    assert all(x["seed"] in (1601,1607) for x in c["matrix"])

@pytest.mark.parametrize("a,b,want",[
    ({"a":[1.,{"b":2.}]},{"a":[1.,{"b":2.}]},0.),
    ({"a":[1.,{"b":2.}]},{"a":[1.,{"b":3.}]},1.),
    ({"a":1},{"a":1,"b":2},float("inf")),
    ([1,2],[1],float("inf")),
    ({"a":True},{"a":1},float("inf")),
    ({"a":"one"},{"a":"two"},float("inf")),
])
def test_recursive_zero_tolerance(a,b,want):
    assert b_module_max(a,b)==want

def b_module_max(a,b):
    return batch._max_abs(a,b)

def _populate(tmp_path, left, right):
    reports=tmp_path/"reports";reports.mkdir()
    for name,p in [("L_v32_off",left),("L_v32_on",right)]:
        (reports/("v106_"+name+".json")).write_text(json.dumps(p))
    return reports

def test_raw_200hz_full_row_and_original_record_exact(monkeypatch,tmp_path):
    src={"traces":[[{"time_s":.005,"joint_vel":[0.]*29,
         "support_foot_force_N":{"left":10,"right":10}}]],
         "records":[{"goal":True}],"contacts":[]}
    _populate(tmp_path,src,src)
    monkeypatch.setattr(batch,"ROOT",tmp_path)
    p=batch.parity(_row(),_row(True))
    assert p["passed"] is True and p["true_recorded_trace_rows"]==1

@pytest.mark.parametrize("field,value",[
    ("time_s",.010),("joint_vel",[.00001]*29),
    ("support_foot_force_N",{"left":11,"right":10})])
def test_any_measured_change_fails(monkeypatch,tmp_path,field,value):
    from copy import deepcopy
    src={"traces":[[{"time_s":.005,"joint_vel":[0.]*29,
         "support_foot_force_N":{"left":10,"right":10}}]],
         "records":[{"goal":True}],"contacts":[]}
    other=deepcopy(src);other["traces"][0][0][field]=value
    _populate(tmp_path,src,other)
    monkeypatch.setattr(batch,"ROOT",tmp_path)
    assert batch.parity(_row(),_row(True))["passed"] is False

def test_original_shot_record_change_fails(monkeypatch,tmp_path):
    src={"traces":[[{"time_s":.005}]],"records":[{"goal":True}]}
    other={"traces":[[{"time_s":.005}]],"records":[{"goal":False}]}
    _populate(tmp_path,src,other)
    monkeypatch.setattr(batch,"ROOT",tmp_path)
    assert not batch.parity(_row(),_row(True))["passed"]

def test_unqualified_pair_is_never_positive():
    p=_row();q=_row(True);q["certified_original_G1_cuda_200Hz"]=False
    assert not batch.parity(p,q)["passed"]

def test_preflight_rejects_write_once_existing_master(monkeypatch,tmp_path):
    p=tmp_path/"already.json";p.write_text("{}")
    monkeypatch.setattr(batch,"MASTER",p)
    # This inherited version-specific test must not depend on the current later version branch.
    monkeypatch.setattr(batch,"git",lambda *args: "version/v0.1.106" if args==("branch","--show-current") else "")
    with pytest.raises(FileExistsError):
        batch.preflight()
