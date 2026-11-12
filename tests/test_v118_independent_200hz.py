import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v118_independent_200hz_audit import validate_record

def test_reject_missing_or_wrong_source_sha(tmp_path):
    (tmp_path/"reports").mkdir()
    path=tmp_path/"reports/v118_A.json"
    path.write_text("{}")
    with pytest.raises(ValueError,match="RAW_SHA_MISMATCH"):
        validate_record(tmp_path,dict(name="A",raw_sha256="0"*64))

def test_actual_repo_8_run_physics_exists_only_as_ignored_evidence_not_test_fixture():
    root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/"configs/v118_frozen_matrix.json").read_text())["matrix"]
    assert len(expected)==8
    assert all(r["scene"] in ("AM","AN") for r in expected)
    assert len({(x["scene"],x["control"]) for x in expected})==6
