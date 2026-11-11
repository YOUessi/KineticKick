"""CPU-only proof tests for v074 watchdog, classifier and source gates."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pytest
from v074_gpu_lifecycle_batch import classify, sha

ROOT=Path(__file__).resolve().parents[1]
def test_exact_protocol_branch_and_sha():
    c=json.loads((ROOT/"configs/v074_frozen_native_lifecycle_research.json").read_text())
    assert c["version"]=="v0.1.74"
    assert c["parent_sha"]=="f206d1a8a65c641acb5f7ef8b65cfa32bcd16e55"
    assert c["free_disk_gib_minimum"]>=20
    assert len(c["stages"])==3
    assert all((ROOT/p).exists() for p in c["source_sha256"])
    assert all(sha(ROOT/p)==h for p,h in c["source_sha256"].items())
    root=Path("/home/you/projects/kinetickick")
    assert all(sha(root/p)==h for p,h in c["original_physx_sha256"].items())

@pytest.mark.parametrize(("text","code","timeout","expected"),[
    ("EIGHT_REAL_SIM_STEPS_DONE_NOT_G1 APP_CLOSE_RETURN",0,False,"NORMAL_EIGHT_STEP_SHUTDOWN"),
    ("APP_CLOSE_ENTER", -15,True,"APP_CLOSE_ENTERED_NEVER_RETURNED_WITHIN_WATCHDOG"),
    ("APP_CLOSE_ENTER APP_CLOSE_RETURN",-15,True,"APP_CLOSE_RETURNED_INTERPRETER_EXIT_HUNG"),
    ("SIM_STEP_8_RETURN",-15,True,"EIGHT_STEPS_DONE_PRE_CLOSE_OR_FINALIZATION_TIMEOUT"),
    ("APP_LAUNCH_ENTER",-15,True,"ISAAC_INIT_OR_SIM_TIMEOUT"),
    ("APP_CLOSE_RETURN",0,False,"NORMAL_WITHOUT_ALL_STEPS"),
    ("",1,False,"ISAAC_PROBE_NONZERO"),
])
def test_classifier(text,code,timeout,expected):
    assert classify(text,code,timeout)==expected

def test_original_physx_source_file_read_only():
    protocol=json.loads((ROOT/"configs/v074_frozen_native_lifecycle_research.json").read_text())
    assert protocol["immutable_write_once"]
    assert protocol["no_delete"] and protocol["no_force_push"]
    assert protocol["max_new_gpu_smoke"]==2
    assert all(s not in ("G1", "original") for s in ("baseline", "cleanup"))
    assert protocol["stages"][2]["requires"].startswith("smoke_shutdown_exit_code_zero")
