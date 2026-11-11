#!/usr/bin/env python3
"""Independent *postregistered* physics pairing, accepting only hash-checked native GPU traces.

Keeps wrapper-valid and early-native nonfoot distinct. 5ms first 100 frames 25-field
physics parity required for causal comparability; no claim of human model efficacy.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_v067_directionality_batch import sha,PROTOCOL,PROGRESS,config
from audit_v059_observer_parity import compare_raw
def analyze():
 c=config();d=json.loads(PROGRESS.read_text())
 if d['protocol_sha256']!=sha(PROTOCOL) or len(d['runs'])!=9:raise ValueError('Missing frozen nine trials')
 raw={}
 for case,row in zip(c['cases'],d['runs']):
  if case['id']!=row['id']:raise ValueError('Case order changed')
  file=ROOT/case['output'];log=file.with_suffix('.log')
  if sha(file)!=row.get('raw_sha256') or sha(log)!=row['log_sha256']:
   raise RuntimeError('Original physics evidence integrity failed: '+case['id'])
  x=json.loads(file.read_text())
  from wholebody_safety_interface_v055 import G1_JOINT_ORDER
  if x.get('seed')!=case['seed'] or x.get('root_fixed') is not False or tuple(x.get('joint_names',[]))!=G1_JOINT_ORDER:
   raise RuntimeError('Original G1 seed/29DoF identity drift: '+case['id'])
  if len(x['traces'][0])<100:raise RuntimeError('Original physical 100 states unavailable')
  logfile=log.read_text(errors='replace')
  if not ('V058_NATIVE_RETURN cuda:0' in logfile and 'V058_BACKEND_ARG_FORWARDED cuda:0' in logfile
          and 'Environment device    : cuda:0' in logfile):
   raise RuntimeError('Not native CUDA G1 physics')
  raw[case['id']]=x
 pairs=[]
 for group in ['new_stance_a','new_stance_b','new_stance_c']:
  a=[raw[group+'_'+p] for p in ['v32','synth','reverse']]
  pre=[compare_raw({'traces':[a[0]['traces'][0][:100]]},
                   {'traces':[x['traces'][0][:100]]}) for x in a[1:]]
  pairs.append({'group':group,
   'same_seed_preaction_100_frames_all_original_fields_parity':all(x['matches_at_abs_tol_1e_6'] for x in pre),
   'raw_parity_checks':pre,
   'native_physical_rows':[len(x['traces'][0]) for x in a],
   'native_original_goal':[x['records'][0].get('goal') for x in a],
   'native_original_recovered':[x['records'][0].get('recovered') for x in a],
   'native_nonfoot_termination':['physical_nonfoot' in x['termination_reasons'][0] for x in a],
   'native_inst_impulse_fraction':[x['records'][0].get('requested_surface_impulse_fraction') for x in a],
   'no_claim_of_full_wrapper_validation_for_censored':True})
 return {'version':'v0.1.67','scope':'POSTREGISTERED original native CUDA first 100 frame parity only; censored data never imputed',
   'frozen_protocol_sha256':sha(PROTOCOL),
   'real_original_native_episodes_used':len(raw),
   'same_seed_initial_parity_groups':sum(x['same_seed_preaction_100_frames_all_original_fields_parity'] for x in pairs),
   'no_professional_human_motion_groundtruth':True,
   'groups':pairs}
def main():
 out=analyze()
 p=ROOT/'reports/v067_all_native_pair_parity_postregistered.json'
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
 print('ORIGINAL_G1_PARITY',out['real_original_native_episodes_used'],out['same_seed_initial_parity_groups'])
 for row in out['groups']:
  print(row['group'],row['same_seed_preaction_100_frames_all_original_fields_parity'],
        'steps',row['native_physical_rows'],'nonfoot',row['native_nonfoot_termination'])
if __name__=='__main__':main()
