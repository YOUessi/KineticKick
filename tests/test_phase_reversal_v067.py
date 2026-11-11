"""No PhysX fixture counts as proof: shape/temporal invariants and immutable contract."""
import json
import numpy as np
from pathlib import Path
from tools.phase_coordinated_v063 import phase_trajectory, WINDOW
ROOT=Path(__file__).resolve().parents[1]

def test_direction_reversal_has_exact_phase_multiset_and_zero_outside():
    p=phase_trajectory()
    a,b=43,78
    q=p.copy();q[a:b+1]=p[a:b+1][::-1]
    assert p.shape==q.shape==(700,29)
    assert not np.array_equal(p[a:b+1],q[a:b+1])
    assert np.array_equal(np.sort(np.abs(p[a:b+1]),axis=0),np.sort(np.abs(q[a:b+1]),axis=0))
    assert np.array_equal(p[:a],q[:a]) and np.array_equal(p[b+1:],q[b+1:])
    assert max(np.max(np.abs(x)) for x in (p,q))<=.075001
    assert WINDOW==(.86,1.56)

def test_preregistered_nine_cases_and_three_distinct_seeds():
    p=json.loads((ROOT/'configs/g1_phase_directionality_batch_v067.json').read_text())
    assert len(p['cases'])==9
    assert set(x['seed'] for x in p['cases'])=={6707,6719,6731}
    assert all(x['perturb_reset'] for x in p['cases'])
    assert set(p['policies'])=={'v32','synth','reverse'}
    assert p['professional_instep_contact_truth'] is False
    assert len({x['output'] for x in p['cases']})==9
    assert all(x['output'].startswith('reports/v067_') for x in p['cases'])

def test_reversal_controller_must_be_synthetic_not_human_truth():
    cfg=json.loads((ROOT/'configs/g1_phase_reversed_negative_v067.json').read_text())
    assert cfg['phase_reversal_v067']['human_contact_groundtruth'] is False
    assert cfg['phase_coordinated_v063']['source']=='synthetic_temporal_ablation_not_human_learned'
    assert cfg['phase_reversal_v067']['unchanged_original_g1_physics']
