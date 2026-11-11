"""v057 observational truth labels must fail closed on missing physics."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_v057_backend_qualification import audit

def sample(tmp_path):
    cases=[]
    runs=[]
    for backend in ("cpu","cuda:0"):
        for name in ("frozen_v32","cmu_10_02","cmu_10_05"):
            id=f"center_{backend.replace(':','_')}_{name}"
            target=f"reports/{id}.json"
            p=tmp_path/"reports"/f"{id}.log";p.parent.mkdir(exist_ok=True)
            p.write_text("Environment device    : cuda:0\n")
            from audit_v057_backend_qualification import sha
            cases.append(dict(id=id,backend=backend,output=target))
            runs.append(dict(id=id,exit_code=0,status="NO_COMPLETE_PHYSICS",log_sha256=sha(p)))
    for backend in ("cpu","cuda:0"):
        for label in ("collide","separated"):
            id=f"toy_{backend.replace(':','_')}_{label}"
            target=f"reports/{id}.json"
            lp=tmp_path/"reports"/f"{id}.log"
            lp.write_text("some runtime messages")
            out=tmp_path/target
            out.write_text('{"case":"%s"}'%label)
            from audit_v057_backend_qualification import sha
            cases.append(dict(id=id,backend=backend,output=target))
            runs.append(dict(id=id,exit_code=124,status="WRITTEN_BUT_CHILD_EXIT_ABNORMAL",
                             report_sha256=sha(out),log_sha256=sha(lp),contact_events=239 if id=="toy_cpu_collide" else 0))
    return dict(cases=cases[:6],toy_cases=cases[6:]),dict(runs=runs,protocol_sha256="a")

def test_all_missing_g1_never_count_as_fail_or_success(tmp_path):
    c,p=sample(tmp_path)
    a=audit(c,p,tmp_path)
    assert a["validated_G1"]==0
    assert a["success_rate"]=="NOT_ESTIMABLE_WITH_ZERO_VALID_G1"
    assert a["cpu_G1_backend_mismatch"]==3
    assert a["toy_cpu_contact_certified"] is True
    assert a["g1_wrist_hip_pair_certified"] is False

def test_missing_toy_is_not_acceptable(tmp_path):
    c,p=sample(tmp_path)
    (tmp_path/c["toy_cases"][0]["output"]).unlink()
    with pytest.raises(ValueError,match="Report/progress"):
        audit(c,p,tmp_path)

def test_reordered_case_rejected(tmp_path):
    c,p=sample(tmp_path)
    p["runs"][0],p["runs"][1]=p["runs"][1],p["runs"][0]
    with pytest.raises(ValueError,match="Incomplete/unordered"):
        audit(c,p,tmp_path)

def test_missing_log_rejected(tmp_path):
    c,p=sample(tmp_path)
    (tmp_path/"reports"/"center_cpu_frozen_v32.log").unlink()
    with pytest.raises(ValueError,match="Missing or drifted"):
        audit(c,p,tmp_path)

def test_report_without_log_proof_rejected(tmp_path):
    c,p=sample(tmp_path)
    p["runs"][6]["log_sha256"]="bad"
    with pytest.raises(ValueError,match="Missing or drifted"):
        audit(c,p,tmp_path)
