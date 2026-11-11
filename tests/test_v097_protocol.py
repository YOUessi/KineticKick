"""Frozen new G/H source eligibility and fail-closed hard-gate topology."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def config():
 return json.loads((ROOT/'configs/v097_frozen_matrix.json').read_text())
def test_six_unique_heldout_pre_registered_runs():
 c=config()
 assert c['version']=='v0.1.97' and len(c['cases'])==6
 assert len({r['name'] for r in c['cases']})==6
 assert [r['scene'] for r in c['cases']]==['G']*3+['H']*3
 assert [r['control'] for r in c['cases']]==['v32','cmu02','cmu05']*2
 assert {r['seed'] for r in c['cases']}=={1409,1423}
 assert all(-.15<=v<=.15 for r in c['cases'] for v in r['ball_offset_xy'])
 assert not any(r['native_poll'] for r in c['cases'])
def test_hard_fail_closed_not_evaluation_tuning():
 c=config()
 assert c['gate_protocol']['min_instep_fraction']==.95
 assert c['gate_protocol']['min_ball_speed_mps']==3.5
 assert c['gate_protocol']['min_full_s']==13.995
 assert c['settings']['min_free_gib']>=20
 assert c['settings']['max_new_G1_episodes']==6
 assert c['causal_contract']['no_test_result_used_to_tune_actions']
 assert c['gate_protocol']['zero_native_callback_means']=='UNKNOWN'
def test_parent_and_control_source_paths_are_frozen():
 import hashlib
 c=config()
 assert c['parent_commit']=='f143a1cc70f5e5c92c6ed2c803c16b1a1934faea'
 for path in c['controls'].values():
  obj=json.loads((ROOT/path).read_text())
  for k in ('template','reference','expert_plan'):
   p=ROOT/obj[k]
   assert p.is_file()
   assert hashlib.sha256(p.read_bytes()).hexdigest()==obj[k+'_sha256']
