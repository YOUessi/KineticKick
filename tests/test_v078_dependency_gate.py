"""Pinned pre-boot data / controller checks, with no GPU or Isaac imports."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("v078", ROOT/"tools/v078_g1_liveness_batch.py")
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def test_frozen_schema_version_and_budget():
    c=json.loads(v.CONFIG.read_text())
    assert c["version"]=="v0.1.78"
    assert c["child_timeout_s"]==150
    assert c["stack_signal_s"]==[25,65,110,140]
    assert c["preregistered_stop_after_first_uncertified"] is True

def test_current_frozen_config_matches_executable_digest():
    assert v.sha(v.CONFIG)==v.FROZEN_SHA

def test_two_independent_upstream_dependency_directories_exist():
    c=json.loads(v.CONFIG.read_text())
    assert all((ROOT/p).is_dir() for p in c["dependency_directories_required"])

def test_upstream_full_controller_source_provenance_pinned():
    from targeted_shooting import controller_provenance
    c=json.loads(v.CONFIG.read_text())
    assert controller_provenance(ROOT)==c["full_controller_provenance"]

def test_previous_failed_run_preserved_without_physx():
    c=json.loads(v.CONFIG.read_text())
    audit=Path("/home/you/projects/kinetickick-wt-v077/reports/v077_g1_original_v32_liveness_1.audit.json")
    assert v.sha(audit)==c["prior_v077_diagnostic_audit_sha256"]
    a=json.loads(audit.read_text())
    assert a["physics_episode_certified"] is False
    assert a["strict_shot_success_rate"] is None
    assert a["liveness"]["outcome"]=="NO_CERTIFIED_EPISODE"

def test_original_frozen_v54_three_physx_unchanged():
    c=json.loads(v.CONFIG.read_text())
    for rel,expected in c["original_physics_sha256"].items():
        assert v.sha(v.ORIGINAL/rel)==expected
