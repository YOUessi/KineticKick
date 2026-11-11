#!/usr/bin/env python3
"""Independent original G1 200Hz physics audit for frozen synthetic phase reversal.

NO human impact/contact/GRF or GPU wrist-hip manifold truth is claimed.
Preserves every valid raw and failure log byte. Does not silently score absent data.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_v067_directionality_batch import config,check_real,sha,PROTOCOL,PROGRESS
from audit_v059_observer_parity import compare_raw
from audit_v063_post_registered import dynamic_metrics

def trace_compare(a,b):
    return compare_raw({'traces':[a]},{'traces':[b]})

def independent_dynamics(raw):
    metric=dynamic_metrics(raw)
    tr=raw['traces'][0]
    ct=raw['records'][0].get('contact_time_s')
    tim=np.asarray([r['time_s'] for r in tr],float)
    if len(tim)<2 or not np.isfinite(tim).all() or np.any(np.diff(tim)<0):
        raise ValueError('corrupt real physics timestamps')
    # Last 1 microsecond termination snapshot is not claimed to be a 200Hz step.
    expected_diffs=np.diff(tim)
    if not np.all(((expected_diffs>.0049)&(expected_diffs<.0051)) |
                  ((expected_diffs>=0)&(expected_diffs<.00001))):
        raise ValueError('non-200Hz physics sequence detected')
    period=(ct-.4,ct+1.25) if ct is not None else (None,None)
    coverage=ct is not None and tim[0]<=period[0]+.005 and tim[-1]>=period[1]-.005
    q=np.asarray([r['joint_pos'] for r in tr],dtype=float)
    dq=np.asarray([r['joint_vel'] for r in tr],dtype=float)
    measured_qdot_limit=np.asarray(raw['joint_velocity_limits'],dtype=float)
    if q.shape!=(len(tim),29) or dq.shape!=q.shape or measured_qdot_limit.shape!=(29,):
        raise ValueError('missing original G1 all 29 joints')
    if np.any(measured_qdot_limit<=0) or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError('nonfinite/invalid G1 state/limit')
    speed_ratio=np.abs(dq)/measured_qdot_limit
    # Force and support are simulator measurements; no synthetic human GRF.
    feet=np.array([[x['support_foot_force_N']['left'],x['support_foot_force_N']['right']] for x in tr])
    nf=np.array([x['nonfoot_contact_N'] for x in tr])
    observers=raw['safe_body_v062']['rows']
    wrist=np.asarray([x['left_wrist_hip_link_origin_distance_m'] for x in observers],float)
    if not (np.isfinite(feet).all() and np.isfinite(nf).all() and np.isfinite(wrist).all()):
        raise ValueError('nonfinite G1 contact/support sensors')
    for i,(r,x) in enumerate(zip(tr,observers)):
        if x['state_alignment_source_index']!=i or abs(x['state_aligned_time_s']-r['time_s'])>1e-7:
            raise ValueError('observer physical clock offset')
        if max(abs(float(j)-float(k)) for j,k in zip(x['joint_pos_rad'],r['joint_pos']))>1e-6:
            raise ValueError('observer 29-DoF not real same-frame pose')
        if max(abs(float(j)-float(k)) for j,k in zip(x['joint_vel_rad_s'],r['joint_vel']))>1e-6:
            raise ValueError('observer 29-DoF not real same-frame velocity')
    impact_mask=np.zeros(len(tim),bool) if ct is None else (tim>=ct-.4-.000001)&(tim<=ct+1.25+.000001)
    if coverage and impact_mask.sum()<300:raise ValueError('nominal impact window lacks expected 200Hz rows')
    return dict(**metric, actual_200Hz_impact_window_available=bool(coverage),
                actual_impact_window_rows=int(impact_mask.sum()),
                last_real_state_s=float(tim[-1]),
                all_29_physically_measured_qdot_peak_ratio=float(np.max(speed_ratio)),
                actual_nonfoot_net_force_peak_N=float(np.max(nf)),
                min_left_wrist_hip_link_origin_distance_m=float(wrist.min()),
                support_left_N_precontact_mean=float(np.mean(feet[(tim<ct)&(tim>=ct-.4),0])) if ct is not None else None,
                support_right_N_precontact_mean=float(np.mean(feet[(tim<ct)&(tim>=ct-.4),1])) if ct is not None else None,
                valid_original_G1_com_200Hz_all_frames=all(len(r.get('whole_body_com_xyz',[]))==3 for r in tr),
                no_claim_of_gpu_native_pair_manifold=True)

def audit():
    c=config();p=json.loads(PROGRESS.read_text())
    if p['protocol_sha256']!=sha(PROTOCOL):raise ValueError('protocol drift')
    if len(p['runs'])!=9:raise ValueError('pre-registered batch incomplete')
    sources={}
    episodes=[]
    for case,entry in zip(c['cases'],p['runs']):
        if case['id']!=entry['id']:raise ValueError('case order drift')
        path=ROOT/case['output'];log=path.with_suffix('.log')
        if not log.exists() or sha(log)!=entry['log_sha256']:raise ValueError('evidence log drift')
        if entry['status']!='VALID_G1':
            episodes.append(dict(id=case['id'],policy=case['policy'],seed=case['seed'],
                status=entry['status'],strict_pass=None,reason=entry.get('reason'),not_scorable=True))
            continue
        if not path.exists() or sha(path)!=entry['raw_sha256']:raise ValueError('evidence JSON drift')
        raw=json.loads(path.read_text())
        check=check_real(case,raw,log.read_text(errors='replace'),entry['return_code'],c)
        metric=independent_dynamics(raw)
        if check['strict_pass']!=entry['strict_pass']:raise ValueError('original physics evaluation drift')
        record=raw['records'][0]
        motion=raw.get('phase_v063_summary',{})
        episodes.append(dict(id=case['id'],policy=case['policy'],seed=case['seed'],
          context=case['ball_case'],status='VALID_G1',strict_pass=check['strict_pass'],
          goal=record.get('goal'),surface=record.get('measured_surface'),
          instep_impulse_fraction=record.get('requested_surface_impulse_fraction'),
          recovered=record.get('recovered'),termination=raw['termination_reasons'],
          actual_action_steps=motion.get('actual_nonzero_action_steps',0),
          actual_max_joint_delta_rad=motion.get('actual_max_joint_delta_rad',0.0),
          gate_counts=motion.get('safety_gate_true_counts',{}),
          original_real_dynamics=metric,raw_sha256=sha(path)))
        sources[case['id']]=raw
    matched=[]
    for k in ('new_stance_a','new_stance_b','new_stance_c'):
        vals=[sources.get(k+'_'+p) for p in ('v32','synth','reverse')]
        if not all(vals):
            matched.append(dict(context=k,matched=False,reason='one or more original G1 trajectories missing'))
            continue
        before=[trace_compare(vals[0]['traces'][0][:100],v['traces'][0][:100]) for v in vals[1:]]
        if not all(v['matches_at_abs_tol_1e_6'] for v in before):
            matched.append(dict(context=k,matched=False,reason='original same-seed pre-intervention physical states differed',comparisons=before))
            continue
        matched.append(dict(context=k,matched=True,initial_100_physics_frames_all_25_field_match=True,
          original_pass=[v['stride_shot_success'] for v in vals],
          eligible_steps=[v.get('phase_v063_summary',{}).get('eligible_steps',0) for v in vals],
          actual_nonzero_action_steps=[v.get('phase_v063_summary',{}).get('actual_nonzero_action_steps',0) for v in vals],
          applied_rad=[v.get('phase_v063_summary',{}).get('actual_max_joint_delta_rad',0) for v in vals]))
    return dict(version='v0.1.67',evaluation='synthetic_directionality_qualifier_not_human_skill',
       protocol_sha256=sha(PROTOCOL),
       pre_physics_git_sha=json.loads((ROOT/'reports/v067_protocol_lock.json').read_text())['frozen_git_sha'],
       attempted=len(episodes),scorable=sum(x['status']=='VALID_G1' for x in episodes),
       strict_pass=sum(x.get('strict_pass') is True for x in episodes),
       scientific_active=sum(x.get('actual_action_steps',0)>0 for x in episodes if x['policy']!='v32'),
       GPU_left_wrist_to_hip_direct_collider_manifold_certified=False,
       human_foot_ball_contact_and_GRF_truth=False,body_link_origin_m_not_mesh_clearance=True,
       observed_200Hz_duration_not_imputed_when_early_terminated=True,
       pairs=matched,episodes=episodes)

def main():
    p=ROOT/'reports/v067_independent_200hz_audit.json'
    if p.exists():raise FileExistsError('independent audit must be write-once')
    out=audit()
    p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print('V067_AUDITED',out['attempted'],out['scorable'],out['strict_pass'],'active',out['scientific_active'])
    for pair in out['pairs']:
        print(pair)
if __name__=='__main__':main()
