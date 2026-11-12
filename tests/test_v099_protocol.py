import json
from pathlib import Path
from tools.v099_gpu_batch import _max_abs
ROOT=Path(__file__).resolve().parents[1]
def test_frozen_matrix():
 c=json.loads((ROOT/'configs/v099_frozen_matrix.json').read_text())
 rows=c['cases']
 assert len(rows)==12 and len({r['name'] for r in rows})==12
 expected=[(s,a,m) for s in 'LMN' for a in ('v32','cmu02','cmu05') for m in ((False,True) if s=='L' else (False,))]
 assert [(r['scene'],r['control'],r['observer']) for r in rows]==expected
 assert {r['seed'] for r in rows}=={1601,1607,1613}
 assert all(abs(x)<=.15 for r in rows for x in r['ball_offset_xy'])
 assert c['safety']['heldout_not_for_tuning']
def test_all_field_parity():
 a={'trace':[{'q':[0.,1.],'com':[0.,0.,1.], 'contacts':[]}]}
 b={'trace':[{'q':[0.,1.],'com':[0.,0.,1.], 'contacts':[]}]}
 assert _max_abs(a,b)==0
 b['trace'][0]['contacts'].append({'actor':'hip','force':1.})
 assert _max_abs(a,b)==float('inf')
 b={'trace':[{'q':[0.,1.],'com':[0.,0.,1.001], 'contacts':[]}]}
 assert 0<_max_abs(a,b)<.01
