#!/usr/bin/env python3
"""Frozen-phase analysis of actual G1 PhysX, not a human GRF/CoP or contact-manifold claim."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from v113_200hz_coupled_audit import group_indices, strip_terminal_event_only, require_200hz

ROOT=Path(__file__).resolve().parents[1]
DT=.005

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def load_physical_arrays(raw):
    if raw.get('root_fixed') is not False: raise ValueError('NOT_FREE_ROOT_G1')
    names=raw['joint_names']
    groups=group_indices(names)
    tr,sentinel=strip_terminal_event_only(raw['traces'][0])
    t=require_200hz(tr)
    def vec(key,dim):
        v=np.asarray([r[key] for r in tr],dtype=np.float64)
        if v.shape!=(len(t),dim) or not np.isfinite(v).all():
            raise ValueError('INVALID_200HZ_'+key.upper())
        return v
    vel=vec('joint_vel',29)
    pos=vec('joint_pos',29)
    tau=vec('joint_torque_estimate',29)
    com=vec('whole_body_com_xyz',3)
    root=vec('root_xyz',3)
    support=np.asarray([[r['support_foot_force_N']['left'],r['support_foot_force_N']['right']] for r in tr],float)
    nonfoot=np.asarray([r['nonfoot_contact_N'] for r in tr],float)
    if support.shape!=(len(t),2) or not np.isfinite(support).all() or np.any(support<0):
        raise ValueError('INVALID_SUPPORT_PROXY')
    if nonfoot.shape!=(len(t),) or not np.isfinite(nonfoot).all() or np.any(nonfoot<0):
        raise ValueError('INVALID_NONFOOT')
    lim=np.asarray(raw['joint_velocity_limits'],float)
    if lim.shape!=(29,) or not np.isfinite(lim).all() or np.any(lim<=0):
        raise ValueError('INVALID_ORIGINAL_29_QDOT_LIMIT')
    impact=float(raw['records'][0]['contact_time_s'])
    if not np.isfinite(impact) or impact<.405 or impact>t[-1]+.005:
        raise ValueError('NO_VALID_IMPACT')
    if len(t)<100: raise ValueError('NO_PHYSICAL_STEPS')
    return dict(t=t,names=names,groups=groups,vel=vel,pos=pos,tau=tau,com=com,root=root,
                support=support,nonfoot=nonfoot,lim=lim,impact=impact,
                terminal_event_markers=sentinel)

def phase_metrics(a):
    t=a['t']; impact=a['impact']
    windows={
        'approach':(t<impact-.4-1e-8),
        'final_pre_400ms':((t>=impact-.4-1e-8)&(t<impact)),
        'post_1250ms':((t>=impact)&(t<=impact+1.25+1e-8)),
        'later_recovery':(t>impact+1.25+1e-8),
    }
    if not windows['final_pre_400ms'].sum(): raise ValueError('MISSING_PREIMPACT_PHASE')
    result={}
    power=a['tau']*a['vel']
    for key,m in windows.items():
        ix=np.flatnonzero(m)
        if not len(ix):
            result[key]={'observed_rows':0,'right_censored':bool(t[-1]<13.995), 'observed':False}
            continue
        by_group={}
        for label,indices in a['groups'].items():
            p=power[ix][:,indices]
            by_group[label]={
                'qdot_rms_radps':float(np.sqrt(np.mean(a['vel'][ix][:,indices]**2))),
                'estimated_positive_motor_work_J':float(np.maximum(p,0).sum()*DT),
                'estimated_negative_motor_work_J':float(np.minimum(p,0).sum()*DT),
                'estimated_absolute_motor_work_J':float(np.abs(p).sum()*DT),
            }
        expected_end={
            'approach':impact-.4-.005,
            'final_pre_400ms':impact-.005,
            'post_1250ms':impact+1.25,
            'later_recovery':13.995,
        }[key]
        complete=bool(t[ix[-1]]+0.005>=expected_end and (key!='later_recovery' or t[-1]>=13.995))
        result[key]={
            'observed':True,'observed_rows':int(len(ix)),
            'time_start_s':float(t[ix[0]]),'time_end_s':float(t[ix[-1]]),
            'right_censored':not complete,
            'all29_qdot_peak_ratio':float(np.max(np.abs(a['vel'][ix])/a['lim'])),
            'com_horizontal_displacement_m':float(np.linalg.norm(a['com'][ix[-1],:2]-a['com'][ix[0],:2])),
            'root_horizontal_displacement_m':float(np.linalg.norm(a['root'][ix[-1],:2]-a['root'][ix[0],:2])),
            'support_left_impulse_proxy_Ns':float(a['support'][ix,0].sum()*DT),
            'support_right_impulse_proxy_Ns':float(a['support'][ix,1].sum()*DT),
            'nonfoot_peak_N':float(a['nonfoot'][ix].max()),
            'groups':by_group,
        }
    return dict(impact_time_s=impact,physics_rows=int(len(t)),physics_duration_s=float(t[-1]),
                actual_14s_recovery_coverage=bool(t[-1]>=13.995),
                recorded_phase_metrics=result,original_29_joint_order_verified=True,
                native_actor_collider_manifold_certified=False,
                human_grf_cop_or_true_impact_power_certified=False,
                motor_work_is_simulated_estimated_torque=True)

def paired_controller_effect(a,b):
    if a['names']!=b['names']: raise ValueError('DIFFERENT_SOURCE_JOINT_ORDER')
    if not np.array_equal(a['t'][:min(len(a['t']),len(b['t']))],b['t'][:min(len(a['t']),len(b['t']))]):
        raise ValueError('MISALIGNED_REAL_PHYSICAL_TIME')
    t=a['t'][:min(len(a['t']),len(b['t']))]
    av=a['vel'][:len(t)]
    bv=b['vel'][:len(t)]
    delta=bv-av
    pre=t<min(a['impact'],b['impact'])
    pre0=t<min(a['impact'],b['impact'])-.4
    near=pre&~pre0
    results={}
    for label,idx in a['groups'].items():
        dist=np.max(np.abs(delta[:,idx]),axis=1)
        altered=np.flatnonzero(dist>1e-7)
        results[label]={
            'first_measurable_difference_s':float(t[altered[0]]) if len(altered) else None,
            'approach_rms_qdot_delta_radps':float(np.sqrt(np.mean(delta[pre0][:,idx]**2))) if np.any(pre0) else None,
            'pre_400ms_rms_qdot_delta_radps':float(np.sqrt(np.mean(delta[near][:,idx]**2))) if np.any(near) else None,
        }
    return dict(shared_actual_physics_frames=int(len(t)),both_complete_14s=bool(len(t)>=2800),
                post_impact_prefix_censored=(len(t)<2800),
                all29_rms_qdot_delta_radps=float(np.sqrt(np.mean(delta**2))),
                groups=results, not_proof_of_human_coordination=True)

def run_audit():
    master=ROOT/'reports/v113_original_g1_batch.audit.json'
    batch=json.loads(master.read_text())
    if batch.get('version')!='v0.1.113':raise ValueError('OTHER_FROZEN_BATCH')
    runs=[];refs={}
    for row in batch['runs']:
        if not row.get('certified_original_G1_cuda_200Hz'):continue
        path=ROOT/'reports'/('v113_'+row['name']+'.json')
        if sha256(path)!=row['raw_sha256']: raise ValueError('PHYSICAL_RAW_SHA_MISMATCH:'+row['name'])
        raw=json.loads(path.read_text())
        arr=load_physical_arrays(raw)
        d=phase_metrics(arr)
        runs.append({'name':row['name'],'scene':row['scene'],'control':row['control'],
                     'observer':row['observer'],'original_sha256':row['raw_sha256'],
                     'strict_hard_gate_pass':row['hard_gates']['hard_gate_pass'],
                     'all_original_failed_gates':row['hard_gates']['failed_gates'],
                     'measured':d})
        if not row['observer']:refs[(row['scene'],row['control'])]=arr
    paired=[]
    for scene in ('AA','AB','AC'):
        base=refs.get((scene,'v32'))
        if base is None:continue
        for ctl in ('cmu02','cmu05'):
            comp=refs.get((scene,ctl))
            if comp is None:continue
            paired.append({'scene':scene,'control':ctl,'baseline':'v32',
                           'actual_common_prefix':paired_controller_effect(base,comp)})
    return {'version':'v0.1.113','source_batch_sha256':sha256(master),
            'new_certified_original_G1_physx':len(runs),
            'independent_trial_results':len(refs),
            'strict_success_count_independent':sum(x['strict_hard_gate_pass'] for x in runs if not x['observer']),
            'new_raw_200hz_physics_frames':sum(x['measured']['physics_rows'] for x in runs),
            'native_pair_certified':False,'human_impact_force_truth_certified':False,
            'new_learned_policy_motor_execution':False,
            'runs':runs,'paired_original_controller_contrasts':paired}

def main():
    dst=ROOT/'reports/v113_phase_biomechanics.audit.json'
    if dst.exists() or dst.is_symlink():raise FileExistsError('WRITE_ONCE_PHASE_AUDIT')
    data=run_audit()
    with dst.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    print('V113_ACTUAL_PHASE_AUDIT',json.dumps({k:v for k,v in data.items() if k not in ('runs','paired_original_controller_contrasts')}))
if __name__=='__main__':main()
