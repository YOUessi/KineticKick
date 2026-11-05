#!/usr/bin/env python3
"""Analyze frozen physical reports and export scientific figures without running Isaac Sim."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from strike_baseline_metrics import summarize_strike
from strike_baseline_protocol import case_plan, verify_files
from strike_baseline_evidence import verify_case_evidence
ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_comparison(a, b):
    fields = ['root_xyz','ball_xyz','ball_velocity','joint_pos','joint_vel','joint_torque_estimate']
    result = {}
    for name in fields:
        x = np.asarray([r[name] for r in a['traces'][0]])
        y = np.asarray([r[name] for r in b['traces'][0]])
        if x.shape != y.shape:
            result[name] = dict(equal_shape=False)
            continue
        error = np.max(np.abs(x-y), axis=1)
        indices = np.flatnonzero(error > 1e-8)
        result[name] = dict(equal_shape=True, max_abs_difference=float(error.max()),
            first_difference_over_1e_minus8_s=a['traces'][0][indices[0]]['time_s'] if len(indices) else None)
    return result


def draw_figures(summary, records, archived, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    labels = {'center':'Original', 'ball_x_minus_5cm':'Ball x -5 cm', 'ball_x_plus_5cm':'Ball x +5 cm',
              'ball_near_10cm':'Ball nearer 10 cm','ball_far_10cm':'Ball farther 10 cm',
              'initial_slower_02':'Initial speed -0.20 m/s','initial_faster_02':'Initial speed +0.20 m/s'}
    cases = summary['cases']
    y = np.arange(len(cases)); colors = ['#225b83' if c['metrics']['outcome']['success'] else '#ad5148' for c in cases]
    fig, axes = plt.subplots(1,2,figsize=(12.8,5.8),layout='constrained')
    for ax,key,title,xlabel in [(axes[0],'speed','Outgoing ball speed','Horizontal speed (m/s), first zero-force sample'),
                               (axes[1],'standing','Final stable-standing duration','Duration (s), original v0.1.15 criterion')]:
        values = [(c['metrics']['contact'] or {}).get('outgoing_speed_xy_mps') if key=='speed' else c['metrics']['final_standing_seconds'] for c in cases]
        ax.barh(y,[v if v is not None else 0 for v in values],color=colors,height=.62)
        ax.set_yticks(y,[labels[c['id']] for c in cases]);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.set_ylabel('Frozen initial-condition case');ax.set_title(title,loc='left',weight='bold')
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        upper=max([v for v in values if v is not None]+[1]);ax.set_xlim(0,upper*1.2)
        for i,v in enumerate(values):ax.text((v or 0)+upper*.02,i,('No contact' if cases[i]['metrics']['contact'] is None else 'Outgoing unavailable') if v is None else f'{v:.2f}',va='center')
    fig.suptitle('Frozen v0.1.15 sensitivity baseline: blue = complete pass, red = task failure',fontsize=13)
    fig.supxlabel('7 prescribed conditions, one rollout each; seed 1201; 14 s limit. This is not a population success-rate estimate.',fontsize=9)
    outputs=[]
    for ext in ('png','svg'):
        out=Path(str(prefix)+'_sensitivity.'+ext);fig.savefig(out);outputs.append(str(out))
    plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12.8,11),layout='constrained')
    colors2={'normal':'#777777','strong':'#225b83'}
    for label,result in archived.items():
        m=summarize_strike(result);contact=m['contact']['first_force_s']
        rows=[r for r in result['traces'][0] if contact-.30<=r['time_s']<=contact+.10]
        x=np.array([r['time_s']-contact for r in rows])*1000
        axes[0,0].plot(x,[np.linalg.norm(r['right_foot_point_velocity']) for r in rows],label=label,color=colors2[label])
        axes[1,1].plot(x,[r['support_foot_force_N']['left'] for r in rows],label=label,color=colors2[label])
        poses=[r for r in result['kinematics'][0] if contact-.30<=r['time_s']<=contact+.10]
        from scipy.spatial.transform import Rotation
        q=np.array([k['root_pose'][3:] for k in poses]);h=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()[:,:2,0]
        yaw=np.unwrap(np.arctan2(h[:,1],h[:,0]));axes[2,0].plot([(k['time_s']-contact)*1000 for k in poses],np.degrees(yaw-yaw[0]),label=label,color=colors2[label])
        joints=['right_hip_pitch_joint','right_knee_joint','right_ankle_pitch_joint']
        w=[m['precontact']['joints'][j]['positive_work_J'] for j in joints]
        axes[2,1].bar(np.arange(3)+(-.18 if label=='normal' else .18),w,width=.36,label=label,color=colors2[label])
    strong=archived['strong'];contact=summarize_strike(strong)['contact']['first_force_s']
    rows=[r for r in strong['traces'][0] if contact-.30<=r['time_s']<=contact+.10]
    x=np.array([r['time_s']-contact for r in rows])*1000
    for name,label,color in [('right_hip_pitch_joint','Hip pitch','#225b83'),('right_knee_joint','Knee','#bd772d'),('right_ankle_pitch_joint','Ankle pitch','#50765d')]:
        i=strong['joint_names'].index(name)
        axes[0,1].plot(x,[r['joint_vel'][i] for r in rows],label=label,color=color)
        axes[1,0].plot(x,[r['joint_vel'][i]*r['joint_torque_estimate'][i] for r in rows],label=label,color=color)
    meta=[(axes[0,0],'Fixed foot reference-point speed','Speed (m/s)'),(axes[0,1],'Strong: signed joint velocities','Angular velocity (rad/s)'),
          (axes[1,0],'Strong: estimated signed joint power','Estimated power (W)'),(axes[1,1],'Ground-filtered left-foot support','Force (N)'),
          (axes[2,0],'Pelvis heading change, measured at 50 Hz','Yaw relative to window start (deg)')]
    plant=-summarize_strike(strong)['support']['last_continuous_left_support_before_contact_s']*1000
    for ax,title,ylabel in meta:
        ax.set_title(title,loc='left',weight='bold');ax.set_xlabel('Time relative to first foot-ball force (ms)');ax.set_ylabel(ylabel)
        ax.axvline(0,color='black',linewidth=.8,linestyle='--');ax.axvline(plant,color='#999999',linewidth=.8,linestyle=':');ax.legend(fontsize=8);ax.grid(alpha=.13)
    axes[2,1].set_title('Estimated positive joint work, preceding 300 ms',loc='left',weight='bold')
    axes[2,1].set_xticks(np.arange(3),['Hip pitch','Knee','Ankle pitch']);axes[2,1].set_xlabel('Right-leg joint');axes[2,1].set_ylabel('Estimated positive work (J)');axes[2,1].legend(fontsize=8)
    fig.suptitle('Archived normal vs strong shot: contact-aligned diagnostics, not a human-naturalness score',fontsize=13)
    fig.supxlabel('Sources: v014 normal / v015 strong physics reports. Power = estimated clipped PD torque x joint speed.\nBlack line: contact; dotted line: strong left-support onset (80 N). Foot point [0.06, 0, -0.025] m is fixed, not the actual collision point.',fontsize=9)
    for ext in ('png','svg'):
        out=Path(str(prefix)+'_coordination.'+ext);fig.savefig(out);outputs.append(str(out))
    plt.close(fig)
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,default=ROOT/'configs/strike_baseline_v1.json')
    p.add_argument('--input-prefix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure-prefix',type=Path)
    p.add_argument('--unwrapped',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.figure_prefix and any(a.figure_prefix.parent.glob(a.figure_prefix.name+'_*')):
        raise FileExistsError('Use a new figure prefix to retain earlier analyses')
    matrix=json.loads(a.matrix.read_text());digest=fingerprint(a.matrix)
    verify_files(ROOT,matrix['frozen_files_sha256'])
    baseline=json.loads((ROOT/matrix['baseline_plan']).read_text())
    source_hashes={field:fingerprint(ROOT/path) for field,path in [('evaluator_sha256','tools/run_strike_baseline_case.py'),('protocol_sha256','tools/strike_baseline_protocol.py')]}
    center=None
    summary=dict(schema_version=1,matrix_sha256=digest,analysis_source_sha256=fingerprint(__file__),
        metrics_source_sha256=fingerprint(ROOT/'tools/strike_baseline_metrics.py'),
        evidence_source_sha256=fingerprint(ROOT/'tools/strike_baseline_evidence.py'),cases=[])
    records={}
    for case in matrix['cases']:
        path=Path(str(a.input_prefix)+'_'+case['id']+'.json');r=json.loads(path.read_text())
        verify_case_evidence(r,case,digest,source_hashes,case_plan(baseline,case),center)
        if case['id']=='center':center=r
        records[case['id']]=r
        summary['cases'].append(dict(id=case['id'],condition=case,report=str(path),sha256=fingerprint(path),
            metrics=summarize_strike(r,matrix['ball_mass_kg'],matrix['precontact_window_s'])))
    historical=ROOT/matrix['historical_report']
    if fingerprint(historical)!=matrix['historical_report_sha256']:raise ValueError('Historical baseline changed')
    archived={'normal':json.loads((ROOT/'reports/v014_stride_instep_preview_final.json').read_text()),'strong':json.loads(historical.read_text())}
    summary['archived_comparison']={name:summarize_strike(r) for name,r in archived.items()}
    summary['replay_vs_historical']=replay_comparison(records['center'],archived['strong'])
    if a.unwrapped:
        r=json.loads(a.unwrapped.read_text());summary['replay_vs_original_headless']=replay_comparison(records['center'],r)
        summary['original_headless_report']=dict(path=str(a.unwrapped),sha256=fingerprint(a.unwrapped))
    summary['completed_cases']=len(summary['cases']);summary['complete_passes']=sum(c['metrics']['outcome']['success'] for c in summary['cases'])
    summary['failure_counts']=dict(Counter(reason for c in summary['cases'] for reason in c['metrics']['outcome']['failure_reasons']))
    if a.figure_prefix:summary['figures']=[dict(path=path,sha256=fingerprint(path)) for path in draw_figures(summary,records,archived,a.figure_prefix)]
    a.output.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('completed_cases','complete_passes','failure_counts')},ensure_ascii=False))


if __name__=='__main__':main()
