import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from tools.v111_original_g1_batch import parity, _max_abs
from tools.v111_200hz_coupled_audit import group_indices, strip_terminal_event_only
ROOT=Path(__file__).resolve().parents[1]
def test_frozen_matrix_independent_units_and_repeats():
    c=json.loads((ROOT/"configs/v111_frozen_matrix.json").read_text())
    assert len(c["matrix"])==10
    assert len({(x["scene"],x["control"],x["seed"]) for x in c["matrix"]})==6
    assert sum(x["observer"] for x in c["matrix"])==4
    assert len({x["seed"] for x in c["matrix"]})==3
    assert c["learned_policy_actuation"] is False
def test_no_duplicate_scene_control_observer():
    c=json.loads((ROOT/"configs/v111_frozen_matrix.json").read_text())
    assert len({(x["scene"],x["control"],x["observer"]) for x in c["matrix"]})==len(c["matrix"])
def test_exact_pair_detector_does_not_silently_accept_missing():
    assert _max_abs([1.0,{"force":4}],[1.0,{"force":4}])==0
    assert _max_abs([1.0,{"force":4}],[1.0,{"force":5}])==1
    assert _max_abs({"x":[1,2]},{"x":[1]})==float("inf")
def test_terminal_marker_distinguished_from_real_step():
    p={"time_s":5.0,"joint_pos":[1,2],"terminated":False}
    q=dict(p,time_s=5.0,terminated=True)
    rows,count=strip_terminal_event_only([p,q])
    assert len(rows)==1 and count==1
def test_not_claim_human_truth_or_native_manifold():
    text=(ROOT/"docs/V111_FROZEN_PROTOCOL.md").read_text()
    assert "actuation remains **disabled**" in text
    assert "not native proof" in text
