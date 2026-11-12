import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v120_original_g1_batch import CFG, parity

def test_g1_matrix_seven_and_unique_seeds():
    c=json.loads(CFG.read_text())
    assert c["version"]=="v0.1.120" and c["min_free_gib"]==20
    assert c["per_child_watchdog_s"]==140 and c["learned_policy_actuation"] is False
    assert c["stop_after_first_unverified_or_parity_fail"] is True
    assert len(c["matrix"])==7
    assert [(r["scene"],r["seed"]) for r in c["matrix"]]==[("AO",3119)]*4+[("AP",3167)]*3
    assert sum(bool(x["observer"]) for x in c["matrix"])==1
    assert c["matrix"][0]["control"]==c["matrix"][1]["control"]=="v32"
    assert c["matrix"][0]["ball_offset_xy"]==c["matrix"][1]["ball_offset_xy"]

def test_no_new_actuator_and_all_original_controllers():
    c=json.loads(CFG.read_text())
    assert list(c["control_configs"])==["v32","cmu02","cmu05"]
    assert c["off_original_runner"]=="tools/v080_run_poll_strike.py"
    assert c["on_original_runner"]=="tools/v098_run_pair_strike.py"
    assert c["on_only_flag"]=="--tensor-pair-v098"

def test_censored_or_missing_physics_fails_closed():
    a={"certified_original_G1_cuda_200Hz":False}
    assert not parity(a,dict(a))["passed"]
