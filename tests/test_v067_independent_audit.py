"""Fail closed on absent physical evidence; no positive fixtures counted."""
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_auditor_will_refuse_incomplete_nine_case_batch(monkeypatch):
    import sys
    sys.path.insert(0,str(ROOT/'tools'))
    from audit_v067_directionality import audit, PROGRESS
    if not PROGRESS.exists():
        try:audit()
        except FileNotFoundError:pass
        else:raise AssertionError('missing original physics accepted')

def test_no_human_truth_or_manifold_claim_in_contract():
    d=json.loads((ROOT/'configs/g1_phase_directionality_batch_v067.json').read_text())
    assert d['professional_instep_contact_truth'] is False
    assert d['human_training_clip_count']==0
    assert d['all_G1_29DoF_joint_qdot_and_95pct_instep_and_14s_recovery_gate_unchanged']
