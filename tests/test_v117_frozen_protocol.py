from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v117_original_g1_batch import _max_abs
R=Path(__file__).resolve().parents[1]
def test_v117_matrix_frozen_and_distinct():
    c=json.loads((R/"configs/v117_frozen_matrix.json").read_text())
    assert len(c["matrix"])==11 and len({x["name"] for x in c["matrix"]})==11
    assert sum(x["observer"] for x in c["matrix"])==2
    assert len({(x["scene"],x["control"]) for x in c["matrix"]})==9
    assert len({x["seed"] for x in c["matrix"]})==3
    assert not c["learned_policy_actuation"] and c["min_free_gib"]>=20
    assert c["per_child_watchdog_s"]==140
    assert c["stop_after_first_unverified_or_parity_fail"]
def test_v117_all_fields_parity_adversarial():
    assert _max_abs({"v":[1.,2.],"x":{"abc":3}},{"v":[1.,2.],"x":{"abc":3}})==0
    assert _max_abs({"v":[1.,2.]},{"v":[1.,2.,3.]})==float("inf")
    assert _max_abs({"v":[1.,2.]},{"v":[1.,2.1]})>0
    assert _max_abs({"v":False},{"v":0})==float("inf")
def test_v117_original_controllers_unchanged():
    c=json.loads((R/"configs/v117_frozen_matrix.json").read_text())
    assert c["control_configs"]=={
        "v32":"configs/cmu_real_120hz_10_02_physx_zero_v052.json",
        "cmu02":"configs/cmu_real_120hz_10_02_physx_v052.json",
        "cmu05":"configs/cmu_real_120hz_10_05_physx_v052.json"}
