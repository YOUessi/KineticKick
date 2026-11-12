import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v119_independent_backend_audit import score

def test_six_sha_qualified_cases_and_readonly_parity():
    r=Path(__file__).resolve().parents[1]
    m=json.loads((r/"reports/v119_native_backend_master.audit.json").read_text())
    out=score(r,m)
    assert out["actual_physx_runs"]==6
    assert out["actual_200hz_steps"]==1080
    assert out["cpu_and_cuda_recorded_off_on_parity_exact"]=={"cpu":True,"cuda":True}
    assert out["cannot_infer_wrist_hip_noncollision"] is True

def test_raw_sha_tampering_rejected(tmp_path):
    r=Path(__file__).resolve().parents[1]
    m=json.loads((r/"reports/v119_native_backend_master.audit.json").read_text())
    (tmp_path/"reports").mkdir()
    with pytest.raises(ValueError,match="RAW_SHA_MISMATCH"):
        score(tmp_path,m)

def test_tensor_witness_is_not_native_collider_truth():
    r=Path(__file__).resolve().parents[1]
    m=json.loads((r/"reports/v119_native_backend_master.audit.json").read_text())
    result=score(r,m)
    assert result["cuda_tensor_positive_witness"]
    assert not result["native_actor_collider_manifold_certified"]["cuda"]
    assert not result["native_actor_collider_manifold_certified"]["cpu"]
