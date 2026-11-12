"""Independent phase motor audit must reject source drift and unscored physics."""
import json
from pathlib import Path
import pytest
from tools.v116_phase_motor_independent import audit

def source(tmp_path,version="v0.1.116",runs=()):
    p=tmp_path/"reports";p.mkdir()
    (p/"v116_original_g1_batch.audit.json").write_text(
        json.dumps({"version":version,"runs":list(runs)}))
    return p

def test_wrong_source_version_rejected(tmp_path):
    source(tmp_path,"v0.1.113")
    with pytest.raises(ValueError,match="WRONG_SOURCE_BATCH"):
        audit(tmp_path)

def test_unscored_launch_not_a_physical_failure(tmp_path):
    source(tmp_path,runs=[{"name":"AG_v32_off",
        "certified_original_G1_cuda_200Hz":False}])
    r=audit(tmp_path)
    assert r["new_scored_original_29dof_g1_runs"]==0
    assert r["independent_conditions"]==0

def test_raw_source_sha_mismatch_cannot_be_silently_accepted(tmp_path):
    p=source(tmp_path,runs=[{"name":"AG_v32_off","scene":"AG","control":"v32",
        "observer":False,"certified_original_G1_cuda_200Hz":True,
        "raw_sha256":"0"*64,"hard_gates":{"hard_gate_pass":False,"failed_gates":["qdot"]}}])
    (p/"v116_AG_v32_off.json").write_text("{}")
    with pytest.raises(ValueError,match="PHYSICS_SHA_DRIFT"):
        audit(tmp_path)
