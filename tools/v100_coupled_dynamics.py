#!/usr/bin/env python3
"""Independent, read-only v100 full-body time-coupling physics audit (not naturalness ground truth)."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from audit_v067_hard_gates import physically_scored
ROOT=Path(__file__).resolve().parents[1]
GROUPS={'waist':('waist_',),'left_leg':('left_hip_','left_knee_','left_ankle_'),
        'right_leg':('right_hip_','right_knee_','right_ankle_'),
        'left_arm':('left_shoulder_','left_elbow_','left_wrist_'),
        'right_arm':('right_shoulder_','right_elbow_','right_wrist_')}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for buf in iter(lambda:f.read(1<<20),b''):h.update(buf)
 return h.hexdigest()
def require_joint_groups(names):
 if len(names)!=29 or len(set(names))!=29:raise ValueError('NOT_ORIGINAL_FULL29_JOINTS')
 ix={g:[i for i,n in enumerate(names) if n.startswith(prefixes)] for g,prefixes in GROUPS.items()}
 if sorted(i for k in ix for i in ix[k])!=list(range(29)):raise ValueError('INCOMPLETE_OR_OVERLAPPING_29DOF_GROUPS')
 if {g:len(v) for g,v in ix.items()}!={'waist':3,'left_leg':6,'right_leg':6,'left_arm':7,'right_arm':7}:
  raise ValueError('INCORRECT_KINEMATIC_JOINT_GROUPS')
 return ix
def physical_grid(trace):
 if not trace:raise ValueError('NO_200HZ_TRACE')
 t=np.asarray([a['time_s'] for a in trace],float)
 if not np.isfinite(t).all():raise ValueError('NONFINITE_TIME')
 dt=np.diff(t)
 if (dt<0).any():raise ValueError('REVERSED_PHYSICAL_TIME')
 # The original recorder can append one termination annotation at t_end+1e-6.
 # This annotation is NOT a new 200 Hz PhysX step; never insert synthetic time.
 terminal=np.flatnonzero(dt<0.0025)
 if len(terminal)>1 or (len(terminal)==1 and terminal[0]!=len(dt)-1):
  raise ValueError('NON_TERMINAL_OR_MULTIPLE_MICROSTEP_MARKERS')
 unique=np.r_[True,dt>=0.0025]
 grid=t[unique]
 if len(grid)>1 and (np.abs(np.diff(grid)-.005)>.00015).any():raise ValueError('NOT_TRUE_200HZ_GRID')
 return np.flatnonzero(unique),int(len(t)-len(grid))
def _as_float(row,key,shape):
 a=np.asarray(row[key],dtype=float)
 if a.shape!=shape or not np.isfinite(a).all():raise ValueError('INVALID_'+key)
 return a
def describe_episode(raw):
 names=raw['joint_names'];ix=require_joint_groups(names)
 idx,duplicates=physical_grid(raw['traces'][0])
 tr=[raw['traces'][0][i] for i in idx]
 t=np.asarray([r['time_s'] for r in tr],float)
 q=np.stack([_as_float(r,'joint_pos',(29,)) for r in tr])
 dq=np.stack([_as_float(r,'joint_vel',(29,)) for r in tr])
 com=np.stack([_as_float(r,'whole_body_com_xyz',(3,)) for r in tr])
 limits=np.asarray(raw['joint_velocity_limits'],float).reshape(-1)
 if limits.shape!=(29,) or not np.isfinite(limits).all() or (limits<=0).any():raise ValueError('INVALID_ORIGINAL_ALL29_SPEED_LIMITS')
 vmax=float(np.max(np.abs(dq)/limits[None,:]))
 rec=raw['records'][0]
 ct=rec.get('contact_time_s')
 if ct is not None:ct=float(ct)
 if ct is not None and not np.isfinite(ct):raise ValueError('NONFINITE_CONTACT_TIME')
 contacted=ct is not None
 pre=contacted and bool(t[0]<=ct-.4+1e-4)
 post=contacted and bool(t[-1]>=ct+1.25-1e-4)
 pre_mask=np.logical_and(t>=ct-.4-1e-4,t<=ct+1e-6) if contacted else np.zeros(len(t),bool)
 post_mask=np.logical_and(t>ct,t<=ct+1.25+1e-4) if contacted else np.zeros(len(t),bool)
 if pre and pre_mask.sum()<79:raise ValueError('INVALID_PRE_CONTACT_SAMPLE_COVERAGE')
 if post and post_mask.sum()<249:raise ValueError('INVALID_POST_CONTACT_SAMPLE_COVERAGE')
 energy={g:np.mean(np.abs(dq[:,indices])/limits[None,indices],axis=1) for g,indices in ix.items()}
 peaks={}
 for g,a in energy.items():
  if pre and pre_mask.any():
   subidx=np.flatnonzero(pre_mask)
   k=int(subidx[np.argmax(a[pre_mask])])
   peaks[g]=dict(peak_time_rel_contact_s=round(float(t[k]-ct),5),
                 peak_mean_abs_normalized_qdot=round(float(a[k]),7),
                 precontact_mean_abs_normalized_qdot=round(float(np.mean(a[pre_mask])),7))
  else:peaks[g]=None
 support={}
 for side in ('left','right'):
  x=np.asarray([float(r['support_foot_force_N'][side]) for r in tr])
  if not np.isfinite(x).all():raise ValueError('INVALID_SUPPORT_SENSOR_FORCE')
  support[side]=dict(precontact_contact_fraction_20N=float(np.mean(x[pre_mask]>20)) if pre else None,
                      post_1p25_contact_fraction_20N=float(np.mean(x[post_mask]>20)) if post else None,
                      peak_sensor_force_N=float(np.max(x)),
                      not_force_plate_GRF=True)
 nonfoot=np.asarray([float(r['nonfoot_contact_N']) for r in tr],float)
 if not np.isfinite(nonfoot).all():raise ValueError('INVALID_NONFOOT')
 j=int(np.argmax(nonfoot))
 return dict(raw_trace_rows=len(raw['traces'][0]),actual_unique_200hz_steps=len(tr),
    terminal_duplicate_rows=duplicates,recorded_duration_s=float(t[-1]),completed_full_14s=bool(t[-1]>=13.999-1e-5),
    actual29_qdot_limit_max_ratio=vmax,actual29_qdot_all_within_limits=bool(vmax<=1.000001),
    contact_time_s=ct,pre_400ms_observed=bool(pre),post_1p25_observed=bool(post),
    post_contact_window_right_censored=bool(contacted and not post),
    support_sensor_N_not_expert_GRF=support,
    group_phase_energy_peaks=peaks,
    arm_kick_leg_peak_lag_s=(None if not pre else round(peaks['left_arm']['peak_time_rel_contact_s']-peaks['right_leg']['peak_time_rel_contact_s'],5)),
    waist_kick_leg_peak_lag_s=(None if not pre else round(peaks['waist']['peak_time_rel_contact_s']-peaks['right_leg']['peak_time_rel_contact_s'],5)),
    com_height_min_m=float(com[:,2].min()),com_planar_span_m=float(np.linalg.norm(com[-1,:2]-com[0,:2])),
    nonfoot_peak_N=float(nonfoot[j]),nonfoot_peak_time_s=float(t[j]),
    late_recovery_unknown=bool(t[-1]<13.999-1e-5),
    descriptive_only_not_elite_human_naturalness=True)
def canonical_hash(x):
 return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def historical_audit(source_roots:dict, out:Path):
 """SHA-bound independent re-score. This DOES NOT constitute heldout training."""
 if out.exists():raise FileExistsError('WRITE_ONCE_HISTORICAL_AUDIT')
 import collections
 rows={}
 manifest_hashes={}
 for version,source in source_roots.items():
  source=Path(source)
  main=source/'reports'/f'{version}_g1_physics_final.audit.json'
  if not main.is_file():raise ValueError('MISSING_MASTER_AUDIT:'+str(main))
  master=json.loads(main.read_text())
  if master['version']!='v0.1.'+version[-2:]:raise ValueError('VERSION_DRIFT:'+version)
  manifest_hashes[version]=sha(main)
  for e in master['runs']:
   if not e['true_original_200Hz_G1_CUDA_PhysX']:
    raise ValueError('UNCERTIFIED_HISTORICAL_G1:'+e['name'])
   name=version+'_'+e['name']
   if name in rows:raise ValueError('DUPLICATE_EPISODE:'+name)
   p=source/'reports'/(name+'.json')
   if not p.is_file() or sha(p)!=e['raw_sha256']:
    raise ValueError('RAW_SHA_DRIFT:'+name)
   raw=json.loads(p.read_text())
   original_score=physically_scored(raw,wrapper_certified=True)
   if bool(original_score['hard_gate_pass'])!=bool(e['gates']['hard_gate_pass']):
    raise ValueError('INDEPENDENT_SCORE_MISMATCH:'+name)
   d=describe_episode(raw)
   record=raw['records'][0]
   rows[name]=dict(source_sha256=e['raw_sha256'],original_sha_verified=True,
     hard_gate_pass=bool(original_score['hard_gate_pass']),
     failed_gates=list(original_score['failed_gates']),
     goal=bool(record['goal']),recovered=bool(record['recovered']),
     instep_impulse_fraction=float(record['requested_surface_impulse_fraction']),
     **d)
 expected_counts={'v097':6,'v098':9}
 for version,count in expected_counts.items():
  if sum(name.startswith(version+'_') for name in rows)!=count:raise ValueError('HISTORICAL_CASE_MISSING')
 result=dict(version='v0.1.100',scope='SHA-verified prior v97 G/H, v98 I/J/K only; deliberately not paper blind',
   original_source_audit_sha256=manifest_hashes,
   certified_original_G1_episode_count=len(rows),
   certified_strict_pass_count=sum(v['hard_gate_pass'] for v in rows.values()),
   true_physics_step_count=sum(v['actual_unique_200hz_steps'] for v in rows.values()),
   fully_observed_contact_window_count=sum(v['pre_400ms_observed'] and v['post_1p25_observed'] for v in rows.values()),
   right_censored_contact_window_count=sum(v['post_contact_window_right_censored'] for v in rows.values()),
   early_termination_count=sum(v['late_recovery_unknown'] for v in rows.values()),
   original_29dof_speed_violations=sum(not v['actual29_qdot_all_within_limits'] for v in rows.values()),
   not_human_dynamics_ground_truth=True,
   missing_native_actor_collider_contact_manifold=True,
   missing_synchronous_true_human_GRF=True,
   no_policy_training_or_motor_execution=True,episodes=rows)
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('V100_HISTORICAL',json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
 return result
def plot_historical(result:dict, png:Path):
 """Scientific visualization of already SHA-verified historical physics; local PNG."""
 if png.exists():raise FileExistsError('WRITE_ONCE_PLOT')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=list(result['episodes'])
 vals=[result['episodes'][k] for k in names]
 fig,axes=plt.subplots(2,1,figsize=(12,10),sharex=True)
 axes[0].bar(range(len(names)),[v['instep_impulse_fraction'] for v in vals])
 axes[0].axhline(.95,ls='--',label='95% legal instep gate')
 axes[0].set_ylabel('Measured instep impulse fraction')
 axes[0].legend()
 axes[1].bar(range(len(names)),[v['nonfoot_peak_N'] for v in vals])
 axes[1].axhline(20.,ls='--',label='20 N non-foot bound (exclusive)')
 axes[1].set_ylabel('Measured max non-foot contact (N)')
 axes[1].set_xticks(range(len(names)),names,rotation=65,ha='right')
 axes[1].legend()
 fig.suptitle('Historical original G1 PhysX: full gate negative controls (not human motion)')
 fig.tight_layout()
 png.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(png,dpi=140)
 plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--source-v97',type=Path,required=True)
 p.add_argument('--source-v98',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--plot',type=Path)
 args=p.parse_args()
 res=historical_audit({'v097':args.source_v97,'v098':args.source_v98},args.output)
 if args.plot:plot_historical(res,args.plot)
