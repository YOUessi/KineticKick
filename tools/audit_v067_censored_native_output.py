#!/usr/bin/env python3
"""Postregistered, explicitly LIMITED audit of native-G1 early exits.

When Isaac returns a saved original trajectory but wrapper finalization did not
stamp policy SHA or strict task verdict, DO NOT relabel the file VALID_G1.
Classify indisputable physical nonfoot failure separately; preserve original
runner's INVALID_G1 status. Do not fold into 9 verified paired outcomes.
"""
from __future__ import annotations
import json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_v067_directionality_batch import sha,config,PROGRESS,PROTOCOL
from audit_v060_observer_factorial import assert_complete_physics_fields

def qualify_partial(raw, logtext, case):
    if not isinstance(raw,dict) or raw.get('root_fixed') is not False or raw.get('seed')!=case['seed']:
        raise ValueError('Cannot assign original G1 physical event to known initial state')
    if 'V058_NATIVE_RETURN cuda:0' not in logtext or 'Environment device    : cuda:0' not in logtext:
        raise ValueError('Original GPU backend not explicitly observed')
    if 'V058_BACKEND_ARG_FORWARDED cuda:0' not in logtext:
        raise ValueError('CUDA was not forwarded to original simulator')
    from wholebody_safety_interface_v055 import G1_JOINT_ORDER
    if tuple(raw.get('joint_names',[]))!=G1_JOINT_ORDER:raise ValueError('29DoF original joint identity not verified')
    trace=raw.get('traces',[[]])[0]
    if len(trace)<100:raise ValueError('Insufficient real 200Hz physical samples')
    assert_complete_physics_fields(trace)
    tt=np.asarray([x['time_s'] for x in trace])
    if np.diff(tt).min()<0 or not np.isfinite(tt).all():raise ValueError('time disorder')
    qdot=np.asarray([x['joint_vel'] for x in trace],dtype=float)
    lim=np.asarray(raw['joint_velocity_limits'],dtype=float)
    if qdot.shape!=(len(trace),29) or lim.shape!=(29,) or np.any(lim<=0):
        raise ValueError('Missing measured original 29DoF velocity gate')
    rec=raw.get('records',[{}])[0]
    termination=raw.get('termination_reasons',[[]])[0]
    nonfoot_N=max(float(x['nonfoot_contact_N']) for x in trace)
    witnessed=('physical_nonfoot' in termination and nonfoot_N>=20
        and rec.get('recovered') is False and rec.get('failed') is True)
    if not witnessed:raise ValueError('Does not demonstrate definite physical nonfoot failure')
    contact=rec.get('contact_time_s')
    return dict(status='PARTIAL_NATIVE_G1_CONFIRMED_NONFOOT_FAILURE',
       original_frozen_runner_status_stays_INVALID_G1=True,
       strict_pass_by_irrevocable_original_nonfoot_gate=False,
       cannot_certify_policy_config_sha_since_final_wrapper_absent=True,
       physical_GPU_G1_backend_verified=True,seed=case['seed'],
       original_PhyxX_real_200hz_rows=len(trace),last_real_frame_s=float(tt[-1]),
       physical_nonfoot_terminated=True,
       observed_impact_s=contact,
       pre_0p4s_observed=contact is not None and tt[0]<=contact-.4+.005,
       post_1p25s_observed=contact is not None and tt[-1]>=contact+1.25-.005,
       full_14s_recovery_observed=tt[-1]>=13.995,
       all_29_real_qdot_max_ratio=float(np.max(np.abs(qdot)/lim)),
       nonfoot_actual_net_force_peak_N=nonfoot,
       original_requested_instep_fraction=rec.get('requested_surface_impulse_fraction'),
       original_ball_entered_goal=rec.get('goal'),
       original_recovered=rec.get('recovered'),
       physical_motion_not_human_contact_kinetics=True)
def main():
    c=config()
    progress=json.loads(PROGRESS.read_text())
    if progress['protocol_sha256']!=sha(PROTOCOL):
        raise ValueError('protocol drift')
    findings=[]
    for case,run in zip(c['cases'],progress['runs']):
        if run['status']!='INVALID_G1':continue
        p=ROOT/case['output'];l=p.with_suffix('.log')
        if not p.exists() or sha(p)!=run.get('raw_sha256') or sha(l)!=run['log_sha256']:
            raise ValueError('old physical bytes changed')
        raw=json.loads(p.read_text())
        try: result=qualify_partial(raw,l.read_text(errors='replace'),case)
        except Exception as e:result=dict(status='PARTIAL_NO_SCIENTIFIC_FAIL_CLASSIFICATION',reason=str(e))
        findings.append(dict(id=case['id'],original_runner_status=run['status'],
                             independent_evidence=result,raw_sha256=run.get('raw_sha256')))
    out=dict(version='v0.1.67',independent_POSTREGISTERED=True,
         frozen_physical_source_or_evaluation_rules_never_changed=True,
         partial_native_attempts=len(findings),
         independently_certified_nonfoot_failures=sum(x['independent_evidence']['status']=='PARTIAL_NATIVE_G1_CONFIRMED_NONFOOT_FAILURE' for x in findings),
         findings=findings)
    path=ROOT/'reports/v067_postregistered_censored_native_audit.json'
    if path.exists():raise FileExistsError('cannot overwrite prior postregistered audit')
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print('V067_CENSORED_NATIVE_AUDIT',out['partial_native_attempts'],
          out['independently_certified_nonfoot_failures'])
    for f in findings:print(f['id'],f['independent_evidence']['status'])
if __name__=='__main__':main()
