import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v101_batch import parity,sha,preflight
from v101_trace_launcher import main as trace_main

def test_parity_exact():
    a={"traces":[[{"q":[1.0,2.0],"dq":[3.0]}]],"records":[{"score":0}]}
    assert parity(a,a)["certified"]
    b={"traces":[[{"q":[1.0,2.01],"dq":[3.0]}]],"records":[{"score":0}]}
    p=parity(a,b)
    assert not p["certified"] and abs(p["max_trace_abs_delta"]-.01)<1e-7
    assert not parity(a,{"records":[]})["certified"]

def test_frozen_preflight_requires_exact_matrix():
    c=json.loads((ROOT/"configs/v101_frozen_diagnostics.json").read_text())
    assert c["parent_sha"]=="b9a1ab1731bdf3f6f2beb1db638d4cf2d0f66a0b"
    assert len(c["g1_cases"])==6
    assert [(x["control"],x["observer"]) for x in c["g1_cases"]]==[
      ("v32",False),("v32",True),("cmu02",False),("cmu02",True),
      ("cmu05",False),("cmu05",True)]
    assert c["min_free_gib"]>=20
    assert c["max_child_s"]<=180

def test_source_trace_rejects_unknown_target(monkeypatch):
    monkeypatch.setattr(sys,"argv",["trace","/tmp/evil.py"])
    with pytest.raises(ValueError):trace_main()

def test_source_hash_matches_known_protocol():
    c=json.loads((ROOT/"configs/v101_frozen_diagnostics.json").read_text())
    assert sha(ROOT/"configs/v100_frozen_matrix.json")==c["frozen_legacy_matrix_sha256"]
