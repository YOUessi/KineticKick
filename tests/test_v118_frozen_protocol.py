import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from v118_original_g1_batch import _max_abs,parity

def test_frozen_protocol_independence_and_run_count():
    p=json.loads((ROOT/'configs/v118_frozen_matrix.json').read_text())
    assert p['version']=='v0.1.118'
    assert p['parent_commit']=='24857ba8e5544dcf72ce904d38b2d5699e6426fb'
    assert len(p['matrix'])==8
    assert len({(r['scene'],r['control']) for r in p['matrix']})==6
    assert [r['scene'] for r in p['matrix']]==['AM']*4+['AN']*4
    assert [r['observer'] for r in p['matrix']]==[False,True,False,False]*2
    assert {r['seed'] for r in p['matrix']}=={3001,3049}
    assert p['per_child_watchdog_s']==140 and p['min_free_gib']==20
    assert p['learned_policy_actuation'] is False
    assert p['stop_after_first_unverified_or_parity_fail'] is True

def test_parity_recurses_into_nested_trace_and_boolean():
    assert _max_abs({'a':[1.,{'x':True}]},{'a':[1.,{'x':True}]})==0.
    assert _max_abs({'a':[1.,{'x':True}]},{'a':[1.1,{'x':True}]})>0
    assert _max_abs({'x':False},{'x':0})==float('inf')
    assert _max_abs([1,2],[1,2,3])==float('inf')
