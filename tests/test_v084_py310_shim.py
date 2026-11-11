"""Postregistered independent digest adapter tests; no change to v84 frozen code."""
import hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v084_evidence_py310 import sha_compatible
def test_py310_streaming_sha_is_exact(tmp_path):
    p=tmp_path/"payload.bin";data=bytes(range(256))*501
    p.write_bytes(data)
    assert sha_compatible(p)==hashlib.sha256(data).hexdigest()
def test_pre_frozen_protocol_digest_and_main_script_are_unchanged():
    import subprocess
    for p in ("configs/v084_frozen_generalization.json","tools/v084_batch.py",
              "tools/v084_evidence.py","tests/test_v084_independent_evidence.py"):
        frozen=subprocess.check_output(["git","show","0c85f4b1384e68d123122df2ca8b1da30ef8f95a:"+p],cwd=ROOT)
        assert frozen==(ROOT/p).read_bytes()
