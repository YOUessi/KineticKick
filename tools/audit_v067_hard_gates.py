#!/usr/bin/env python3
"""Independent, POSTREGISTERED hard-gate recheck over all nine source-hashed G1 Physics episodes.

Original evaluation's stride_shot_success may omit >=95% actual instep impulse.
Do NOT silently treat any original goal or legacy PASS as compliant with requested
hard gate. Missing output/partial controller fingerprint => no positive PASS.
"""
from __future__ import annotations
import hashlib,json,sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_v067_directionality_batch import config,sha,PROGRESS,PROTOCOL
from audit_v060_observer_factorial import assert_complete_physics_fields
MIN_INST=.95
MIN_BALL=3.5
MAX_QDOT_RATIO=1.000001

def physically_scored(raw, wrapper_certified):
    t=raw.get('traces',[[]])[0]
    if len(t)<100:raise ValueError('Missing real 200Hz sample stream')
    assert_complete_physics_fields(t)
    record=raw['records'][0]
    tt=np.asarray([r['time_s'] for r in t],float)
    dq=np.asarray([r['joint_vel'] for r in t],float)
    vel_lim=np.asarray(raw['joint_velocity_limits'],float).reshape(-1)
    if dq.shape!=(len(t),29) or vel_lim.shape!=(29,) or np.any(vel_lim<=0):
        raise ValueError('Missing original 29DoF actual qdot gate')
    if not np.isfinite(dq).all() or not np.isfinite(tt).all():
        raise ValueError('Nonfinite real physics')
    contact=record.get('contact_time_s')
    speed_samples=[]
    if contact is not None:
        for row in t:
            if contact+.0049<=row['time_s']<=contact+.1101:
                speed_samples.append(float(np.linalg.norm(row['ball_velocity'])))
    speed=max(speed_samples) if speed_samples else None
    goal=record.get('goal') is True
    instep=record.get('requested_surface_impulse_fraction')
    fraction_good=(isinstance(instep,(float,int)) and math.isfinite(instep)
                and instep>=MIN_INST and record.get('measured_surface')=='instep')
    recovered=record.get('recovered') is True
    full=tt[-1]>=13.995 and tt[0]<=.005
    speed_ok=speed is not None and math.isfinite(speed) and speed>=MIN_BALL
    qdot_max=float(np.max(np.abs(dq)/vel_lim))
    safe_speed=qdot_max<=MAX_QDOT_RATIO
    nonfoot=max(float(r['nonfoot_contact_N']) for r in t)
    terminated=any('physical_nonfoot' in x for x in raw.get('termination_reasons',[]))
    safe_nonfoot=(not terminated and nonfoot<20)
    # True only when ALL measured physical and wrapper-source gates pass.
    strict=bool(wrapper_certified and goal and fraction_good and recovered and full
                and speed_ok and safe_speed and safe_nonfoot)
    failures=[]
    for key,gate in [('original_G1_wrapper_source_certified',wrapper_certified),('goal',goal),
                     ('original_measured_instep_impulse_ge_95pct',fraction_good),
                     ('ball_speed_3p5mps_5_to_110ms_proxy',speed_ok),
                     ('original_29_qdot_within_limit',safe_speed),('no_nonfoot',safe_nonfoot),
                     ('full_14s',full),('recovered',recovered)]:
        if not gate:failures.append(key)
    return dict(hard_gate_pass=strict,original_native_wrapper_source_certified=wrapper_certified,
        observed_ball_contact_s=contact,ball_speed_peak_mps_post_5_to_110ms_proxy=speed,
        measured_surface=record.get('measured_surface'),measured_instep_fraction=instep,
        original_goal_entered=goal,recovered=recovered,whole_14s_observed=bool(full),
        original_29_qdot_max_ratio=qdot_max,measured_nonfoot_max_net_N=nonfoot,
        original_physical_nonfoot_terminated=bool(terminated),
        original_raw_stride_shot_success=raw.get('stride_shot_success'),
        source_version_original_stride_pass_is_not_95pct_instep_hard_gate=True,
        physical_rows=len(t),last_true_sample_s=float(tt[-1]),failed_gates=failures)
def audit():
 cfg=config();p=json.loads(PROGRESS.read_text())
 if p['protocol_sha256']!=sha(PROTOCOL) or len(p['runs'])!=9:raise ValueError('Frozen 9 sources missing')
 from wholebody_safety_interface_v055 import G1_JOINT_ORDER
 out=[]
 for case,row in zip(cfg['cases'],p['runs']):
    if row['id']!=case['id']:raise ValueError('case order changed')
    path=ROOT/case['output'];log=path.with_suffix('.log')
    if not path.exists() or sha(path)!=row['raw_sha256'] or sha(log)!=row['log_sha256']:
        raise RuntimeError('Original physical evidence hash drift')
    raw=json.loads(path.read_text())
    if tuple(raw.get('joint_names',()))!=G1_JOINT_ORDER or raw.get('seed')!=case['seed'] or raw.get('root_fixed') is not False:
        raise RuntimeError('Not original free-root 29DoF G1')
    txt=log.read_text(errors='replace')
    if ('V058_NATIVE_RETURN cuda:0' not in txt or 'Environment device    : cuda:0' not in txt
            or 'V058_BACKEND_ARG_FORWARDED cuda:0' not in txt):
        raise RuntimeError('Not confirmed original CUDA G1')
    certified=(row['status']=='VALID_G1' and raw.get('stride_shot_success') is True
              and raw.get('policy_sha256')==cfg['policy_sha256'][cfg['policies'][case['policy']]])
    score=physically_scored(raw,certified)
    out.append(dict(id=case['id'],group=case['ball_case'],policy=case['policy'],
      original_runner_status=row['status'],seed=case['seed'],raw_sha256=row['raw_sha256'],
      independent_postregistered_hard_gates=score))
 return dict(version='v0.1.67',audit_type='POSTREGISTERED_STRICT_HARD_GATE_INCLUDING_INSTEP_95pct',
     original_frozen_protocol_sha256=sha(PROTOCOL),
     physical_original_GPU_G1_attempts=len(out),
     original_wrapper_VALID_G1=sum(row['original_runner_status']=='VALID_G1' for row in out),
     original_raw_wrapper_stride_PASS=sum(row['independent_postregistered_hard_gates']['original_raw_stride_shot_success'] is True for row in out),
     independently_confirmed_all_hard_gate_PASS=sum(row['independent_postregistered_hard_gates']['hard_gate_pass'] is True for row in out),
     independent_hard_gate_scoring_for_partial_native_disabled_for_positive=True,
     original_source_physx_PD_and_rule_unchanged=True,
     postregistered_ball_velocity_5_to_110ms_is_proxy_not_original_ball_power_truth=True,
     no_human_foot_groundtruth_or_GPU_shape_pair_certified=True,
     episodes=out)
def main():
 out=audit()
 path=ROOT/'reports/v067_independent_hard_gate_audit_postregistered.json'
 if path.exists():raise FileExistsError(path)
 path.write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item())+'\n')
 print('HARD_GATE_AUDIT_PHYSX',out['physical_original_GPU_G1_attempts'],
      'ORIGINAL_WRAPPER_PASS',out['original_raw_wrapper_stride_PASS'],
      'INSTEP95_HARD_PASS',out['independently_confirmed_all_hard_gate_PASS'])
 for row in out['episodes']:
  q=row['independent_postregistered_hard_gates']
  print(row['id'],'original',q['original_raw_stride_shot_success'],'HARD',q['hard_gate_pass'],
      'instep',q['measured_instep_fraction'],'failed',q['failed_gates'])
if __name__=='__main__':main()
