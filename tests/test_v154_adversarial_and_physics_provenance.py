"""v154 negative controls, historical real 200Hz provenance and solver gates."""
import itertools
import pytest
from v154_cpu_feasibility_campaign import (
    synthetic_problem,mutate,NEGATIVES,SUPPORT,SEEDS,
    qualified_history,run_campaign)
from v154_floating_base_qp import solve
from v154_independent_wbc_audit import independent_check

@pytest.mark.parametrize("fault,stance,seed",list(itertools.product(NEGATIVES,SUPPORT,SEEDS)))
def test_adversarial_negative_rejected(fault,stance,seed):
    p=mutate(synthetic_problem("swing",stance,seed),fault)
    try:
        r=solve(p)
        assert not independent_check(p,r)["pass"], (fault,stance,seed)
    except (ValueError,KeyError):
        pass

def test_historical_original_g1_real_sources_sha_pinned():
    data=qualified_history()
    assert {k:v["true_historical_200hz_rows"] for k,v in data.items()}=={
        "frozen_v32":2800,"cmu_10_02":2800,"cmu_10_05":1028}
    assert sum(v["true_historical_200hz_rows"] for v in data.values())==6628
    assert data["cmu_10_05"]["extra_terminal_note_not_a_physics_step"]==1
    assert all(v["native_contact_manifold_truth"]=="UNKNOWN" for v in data.values())
    assert all(not v["ground_only_signed_force_or_cop_recorded"] for v in data.values())

def test_full_frozen_campaign_outcomes_not_spoofing_real_physics():
    r=run_campaign()
    assert r["nominal_cpu_synthetic_count"]==48
    assert r["negative_cpu_synthetic_count"]==len(NEGATIVES)*2*4
    assert r["negative_rejected"]==r["negative_cpu_synthetic_count"]
    assert r["nominal_solved"]<=48
    assert r["new_real_physx_runs"]==0
    assert r["new_real_instep_success_rate"] is None
    assert r["actual_native_collision_manifold"]=="UNKNOWN"
    assert not r["actually_learned_professional_human_kinetics"]
