"""v42 novel exact ball/velocity combinations: frozen study and real PhysX audit."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from strike_cases import load_cases
from run_novel_ball_v042 import verify
from audit_novel_ball_v042 import interval

PROTOCOL=ROOT/"configs/novel_ball_generalization_v042.json"
RESULT=ROOT/"reports/v042_paired_novel_ball_audit.json"

def test_stratified_novel_parameters_predeclared_and_not_named_cases():
    cfg=json.loads(PROTOCOL.read_text())
    verify(cfg)
    assert cfg["no_adaptive_parameter_search"] and cfg["no_retraining"]
    assert cfg["number_of_new_parameter_combinations"]==6
    assert cfg["number_of_prespecified_original_physical_runs"]==12
    seen={
        tuple(round(float(v),3) for v in [*c["ball_offset_xy_m"],c["root_velocity_y_delta_mps"]])
        for c in load_cases(ROOT).values()}
    for case in cfg["trials"]:
        triple=tuple([*case["ball_offset_xy_m"],case["velocity_y_delta_mps"]])
        assert triple not in seen
        assert all(abs(a)<.151 for a in case["ball_offset_xy_m"])
        assert abs(case["velocity_y_delta_mps"])<.301

def test_wilson_interval_is_bounded_and_monotonic():
    for n in (6,20):
        intervals=[interval(k,n) for k in range(n+1)]
        assert abs(intervals[0][0])<1e-12
        assert abs(intervals[-1][1]-1.)<1e-12
        for k,(lo,hi) in enumerate(intervals):
            assert -1e-12<=lo<=k/n+1e-12 and k/n-1e-12<=hi<=1+1e-12
            if k<n:
                assert intervals[k+1][0]>=lo
                assert intervals[k+1][1]>=hi

def test_shas_preserved_no_hidden_retraining():
    cfg=json.loads(PROTOCOL.read_text())
    for name,h in cfg["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,(name,"source drift")
    assert cfg["original_v32_checkpoint_sha256"]=="ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326"
    assert cfg["reference_generated_q_sha256"]=="e74438befc96d523dc95032f93dee67b2911988ca920305e362fa951ec3a9881"

@pytest.mark.skipif(not RESULT.exists(),reason="Novel-ball paired G1 PhysX study still executing")
def test_all_frozen_paired_physx_samples_have_independent_task_gate():
    r=json.loads(RESULT.read_text())
    assert r["n_cases"]==6
    assert r["protocol_sha256"]==hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert not r["population_robustness_established"]
    assert not r["professional_instep_biomechanics_established"]
    assert r["no_hyperparameter_tuning"] and r["no_trained_new_controller"]
    assert len(r["trials"])==6
    assert 0<=r["old_success_count"]<=6
    assert 0<=r["new_success_count"]<=6
    for trial in r["trials"]:
        for key in ("original_v32","style_v41"):
            side=trial[key]
            assert side["qdot_speed_limit_violation_samples"]>=0
            assert side["physics_steps"]>0
            assert side["sha256"]==hashlib.sha256((ROOT/side["path"]).read_bytes()).hexdigest()
            assert side["strict_functional_pass"]==all(side["original_physx_gates"].values())
        assert trial["style_v41"]["style_applied_decisions"]>=0

@pytest.mark.skipif(not RESULT.exists(),reason="Novel-ball paired G1 PhysX study still executing")
def test_independent_exact_new_parameter_trials_are_not_random_seed_statistics():
    r=json.loads(RESULT.read_text())
    assert r["source_parameter_triples_unseen_in_named_fixed_development_cases"]
    assert r["population_robustness_established"] is False
    assert r["no_promotion"] is True
    assert 0 <= r["descriptive_new_wilson95"][0] <= r["descriptive_new_wilson95"][1] <= 1
