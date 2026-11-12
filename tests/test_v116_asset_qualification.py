"""Contract tests for missing/mismatched original G1 models and human markers."""
import hashlib
import json
from pathlib import Path
import pytest
from tools.v116_asset_qualification import validate_asset, validate_controller_assets, assert_all_assets

ROOT=Path(__file__).resolve().parents[1]

def test_missing_checkpoint_fails_before_gpu(tmp_path):
    with pytest.raises(FileNotFoundError,match="MISSING_G1_ASSET"):
        validate_asset(tmp_path,"experiments/recovery.pt","0"*64,"recovery_checkpoint")

def test_checksum_mismatch_fails_before_gpu(tmp_path):
    p=tmp_path/"checkpoint.pt";p.write_bytes(b"wrong-weight")
    with pytest.raises(ValueError,match="G1_ASSET_SHA_MISMATCH"):
        validate_asset(tmp_path,"checkpoint.pt","0"*64,"recovery_checkpoint")

def test_dotdot_escape_fails_closed(tmp_path):
    with pytest.raises(ValueError,match="UNSAFE_ASSET_PATH"):
        validate_asset(tmp_path,"../outside.pt","0"*64,"recovery_checkpoint")

def test_valid_asset_read_only(tmp_path):
    p=tmp_path/"file.pt";p.write_bytes(b"evidence")
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    a=validate_asset(tmp_path,"file.pt",h,"recovery_checkpoint")
    assert a["sha256"]==h and p.read_bytes()==b"evidence"

def test_original_three_controllers_checkpoint_qualified():
    c=json.loads((ROOT/"configs/v116_frozen_matrix.json").read_text())
    d=assert_all_assets(ROOT,c)
    assert set(d)=={"v32","cmu02","cmu05"}
    for a in d.values():
        assert a[0]["relative_path"]=="experiments/post_shot_v032/ppo_s3201/last_model.pt"
        assert a[0]["sha256"]=="ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326"
        assert all(x["bytes"]>0 for x in a)
