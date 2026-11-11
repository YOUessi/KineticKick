"""v060 frozen factor matrix, original G1 integrity and fail-closed observer tests."""
from __future__ import annotations
import copy
import hashlib
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_v059_observer_parity import DYNAMIC_FIELDS
from audit_v060_observer_factorial import audit,assert_complete_physics_fields
from run_backend_batch_v060 import command,freeze,validate

def config():
    return json.loads((ROOT/"configs/g1_observer_factorial_v060.json").read_text())

def test_frozen_v060_protocol_all_source_hashes_match():
    assert len(freeze(config()))==64

@pytest.mark.parametrize("mode,native,dynamics",[
    ("none",False,False),("native_only",True,False),
    ("dynamics_only",False,True),("full",True,True)])
def test_factorial_flags_are_independent(mode,native,dynamics):
    c=config()
    k=next(x for x in c["cases"] if x["observer"]==mode)
    args=command(c,k)
    assert ("--native-readonly-pair-v056" in args)==native
    assert ("--pair-dynamics-v055" in args)==dynamics
    assert args[args.index("--sim-device")+1]==k["backend"]

def test_v060_has_frozen_10_unique_factor_cases():
    c=config()
    assert len(c["cases"])==10
    assert len({x["id"] for x in c["cases"]})==10
    assert len({x["output"] for x in c["cases"]})==10
    assert all(x["phase"]=="diagnostic_only_not_training_not_generalization" for x in c["cases"])
    assert all(x["perturb_reset"] is False for x in c["cases"])
    assert {x["observer"] for x in c["cases"]}=={"none","native_only","dynamics_only","full"}

def mock_frame():
    frame={f:0 for f in DYNAMIC_FIELDS}
    frame.update(joint_pos=[0.0]*29,joint_vel=[0.0]*29,
                 time_s=.005,ball_xyz=[0.0,0.0,0.0],
                 support_foot_force_N={"left":0.0,"right":0.0})
    return frame

def test_parity_cannot_be_accepted_when_both_missing_same_field():
    f=mock_frame()
    del f["support_foot_force_N"]
    with pytest.raises(ValueError,match="Missing recorded 200Hz physical fields"):
        assert_complete_physics_fields([f,f])

def test_full_original_29dof_field_contract_accepts_complete():
    assert_complete_physics_fields([mock_frame()])

def test_incomplete_original_29dof_is_rejected():
    f=mock_frame()
    f["joint_pos"]=[0.0]*28
    with pytest.raises(ValueError,match="Incomplete original 29DoF"):
        assert_complete_physics_fields([f])

def test_validate_native_only_accepts_real_complete_raw_shape():
    c=config()
    k=next(x for x in c["cases"] if x["id"]=="cuda_v32_native_only")
    frames=[]
    for i in range(36):
        frames.append({"time_s":i*.005,"joint_vel":[0.]*29})
    raw={"v057_requested_physics_backend":"cuda:0","root_fixed":False,"seed":1201,
         "policy_sha256":c["policy_hashes"][c["policies"][k["policy"]]],
         "joint_names":[f"j{i}" for i in range(29)],
         "traces":[frames],"records":[{"contact_time_s":.1}],
         "stride_shot_success":False,
         "physx_native_report_v056":{"report_capture_complete":True,"errors":[],
                                      "dropped_events":0,"native_headers":0}}
    result=validate(c,k,raw,"V058_NATIVE_RETURN V058_BACKEND_ARG_FORWARDED","cuda:0")
    assert result["strict_success"] is False
    assert result["trace_rows_including_terminal"]==36
    tampered=copy.deepcopy(raw)
    tampered["physx_native_report_v056"]["dropped_events"]=1
    with pytest.raises(ValueError,match="Incomplete native"):
        validate(c,k,tampered,"V058_NATIVE_RETURN V058_BACKEND_ARG_FORWARDED","cuda:0")

def test_missing_raw_even_if_exit_zero_remains_unscorable(tmp_path):
    (tmp_path/"configs").mkdir()
    (tmp_path/"reports").mkdir()
    c=copy.deepcopy(config())
    c["source_hashes"]={}
    cfg=tmp_path/"configs/g1_observer_factorial_v060.json"
    cfg.write_text(json.dumps(c))
    digest=hashlib.sha256(cfg.read_bytes()).hexdigest()
    (tmp_path/"reports/v060_protocol_lock.json").write_text(json.dumps(
        {"protocol_sha256":digest}))
    entries=[]
    for case in c["cases"]:
        log=tmp_path/case["output"].replace(".json",".log")
        log.write_text(f"Environment device    : {case['backend']}\n")
        entries.append({"id":case["id"],"observer":case["observer"],
                        "backend_requested":case["backend"],
                        "backend_observed":case["backend"],"exit_code":0,
                        "status":"NO_G1_OUTPUT",
                        "log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()})
    progress=tmp_path/"reports/v060_batch_progress.json"
    progress.write_text(json.dumps({"protocol_sha256":digest,"runs":entries}))
    result=audit(tmp_path)
    assert result["physics_attempts"]==10
    assert result["validated_original_G1"]==0
    assert result["strict_successes"]==0
    assert all(not x["valid"] and x["strict_success"] is None for x in result["runs"])
    assert result["four_factor_observer_comparisons"]["cuda_v32_full_vs_none"]["available"] is False
    # An altered recorded log must never be treated as previously authenticated.
    firstlog=tmp_path/c["cases"][0]["output"].replace(".json",".log")
    firstlog.write_text(firstlog.read_text()+"tampered")
    with pytest.raises(ValueError,match="Log missing or changed"):
        audit(tmp_path)

def test_v060_gpu_failures_not_conflated_with_cpu_backend():
    c=config()
    assert all(x["backend"]=="cuda:0" for x in c["cases"] if x["id"].startswith("cuda_"))
    assert all(x["backend"]=="cpu" for x in c["cases"] if x["id"].startswith("cpu_"))
