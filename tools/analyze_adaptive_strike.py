#!/usr/bin/env python3
"""Audit the v0.1.24 negative-result study and draw its frozen-control comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from strike_baseline_metrics import summarize_strike
from strike_baseline_protocol import verify_files,case_plan
from strike_baseline_evidence import verify_case_evidence
from adaptive_strike_evidence import verify_candidate_runtime,validate_feedback_trace
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.figure.exists():raise FileExistsError('Preserve existing analysis')
    matrix_path=ROOT/'configs/strike_baseline_v1.json';matrix=json.loads(matrix_path.read_text())
    verify_files(ROOT,matrix['frozen_files_sha256'])
    plan=json.loads((ROOT/matrix['baseline_plan']).read_text())
    adaptive_plan_path=ROOT/'configs/adaptive_strike_v1.json';adaptive_plan=json.loads(adaptive_plan_path.read_text())
    hashes={field:sha(ROOT/path) for field,path in [('evaluator_sha256','tools/run_strike_baseline_case.py'),('protocol_sha256','tools/strike_baseline_protocol.py')]}
    sources={n:sha(ROOT/'tools'/n) for n in ['run_adaptive_stride.py','adaptive_strike.py']}
    center=None;rows=[];records={}
    diagnostic_path=ROOT/'experiments/running_v024/velocity_jitter_pilot/velocity_diagnostic_matrix.json'
    diagnostic=json.loads(diagnostic_path.read_text())
    for i,case in enumerate(matrix['cases']+diagnostic['cases']):
        name=case['id'];extra=i>=7
        bp=ROOT/'reports'/((f'v024_velocity_baseline_{name}' if extra else f'v023_baseline_{name}')+'.json')
        if extra:cp=ROOT/'reports'/f'v024_velocity_candidate_{name}.json'
        else:cp=ROOT/'reports'/((f'v024_ankle_preserved_{name}' if name in ['center','initial_slower_02','initial_faster_02'] else f'v024_ball_positions_{name}')+'.json')
        baseline=json.loads(bp.read_text());candidate=json.loads(cp.read_text())
        verify_case_evidence(baseline,case,sha(diagnostic_path if extra else matrix_path),hashes,case_plan(plan,case),center)
        if center is None:center=baseline
        verify_candidate_runtime(candidate,baseline,sources)
        validate_feedback_trace(candidate,adaptive_plan['feedback'])
        ev=candidate['adaptive_control']
        if ev['source_sha256']!=sources or candidate['stride_shot_plan_sha256']!=sha(adaptive_plan_path):raise ValueError('Mixed candidate source/configuration')
        if candidate['running_checkpoint_sha256']!=sha(ROOT/adaptive_plan['running_checkpoint']):raise ValueError('Candidate checkpoint changed')
        if ev['template_sha256']!=sha(ROOT/adaptive_plan['feedback']['template']):raise ValueError('Calibration changed')
        expected=baseline['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0]
        if not np.allclose(ev['observed_initial_root_velocity'],expected,atol=1e-6,rtol=0):raise ValueError('Mismatched reset velocity')
        if candidate['initial_robot_state']!=baseline['initial_robot_state']:raise ValueError('Mismatched initial pose')
        for key in ['asset_sha256','policy_sha256','joint_effort_limits','joint_velocity_limits','points','goals','goal_width_m','goal_height_m']:
            if candidate[key]!=baseline[key]:raise ValueError('Changed physics/task: '+key)
        b=summarize_strike(baseline);c=summarize_strike(candidate)
        rows.append(dict(case=case,exploratory_extra=extra,baseline=b,candidate=c,
            baseline_report=str(bp.relative_to(ROOT)),baseline_sha256=sha(bp),candidate_report=str(cp.relative_to(ROOT)),candidate_sha256=sha(cp)))
        records[name]=(baseline,candidate)
    result=dict(status='rejected_as_robustness_improvement',cases=rows,
        frozen_seven=dict(baseline_passed=sum(r['baseline']['outcome']['success'] for r in rows[:7]),candidate_passed=sum(r['candidate']['outcome']['success'] for r in rows[:7]),total=7),
        extra_velocity_diagnostics=dict(baseline_passed=sum(r['baseline']['outcome']['success'] for r in rows[7:]),candidate_passed=sum(r['candidate']['outcome']['success'] for r in rows[7:]),total=2),
        interpretation='Prescribed development cases, one rollout each; not population success probabilities or a held-out test',
        training=json.loads((ROOT/'experiments/running_v024/velocity_jitter_pilot/result.json').read_text()),
        analysis_sha256=sha(__file__),evidence_validator_sha256=sha(ROOT/'tools/adaptive_strike_evidence.py'),candidate_sources=sources)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    labels=['Original','Ball x -5 cm','Ball x +5 cm','Ball near 10 cm','Ball far 10 cm','Speed -0.20','Speed +0.20','Speed -0.10*','Speed +0.10*']
    y=np.arange(9)
    for method,offset,color in [('baseline',-.17,'#486579'),('candidate',.17,'#ca714b')]:
        v=[r[method]['final_standing_seconds'] for r in rows]
        axes[0].barh(y+offset,v,height=.31,label=method,color=color)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlabel('Final standing duration (s)');axes[0].set_title('Same physical success gates');axes[0].legend()
    for n,color in [('center','#344e5c'),('initial_slower_02','#4c8b70'),('initial_faster_02','#b45d4c')]:
        for method,ls in [(0,'--'),(1,'-')]:
            rr=[r for r in records[n][method]['traces'][0] if r['time_s']<=1.6 and not r['terminated']]
            axes[1].plot([r['time_s'] for r in rr],[r['root_xyz'][0] for r in rr],color=color,ls=ls,label=n.replace('initial_','').replace('_02','')+(' frozen' if method==0 else ' candidate'))
        fb=records[n][1]['adaptive_control']['feedback']
        axes[2].plot([r['time_s'] for r in fb if r['time_s']<=1.8],[1000*r['phase_shift_s'] for r in fb if r['time_s']<=1.8],color=color,label=n.replace('initial_','').replace('_02',''))
    axes[1].set_xlabel('Physical time (s)');axes[1].set_ylabel('Root lateral x (m)');axes[1].set_title('Approach drift changes, task still fails');axes[1].legend(fontsize=7)
    axes[2].set_xlabel('Physical time (s)');axes[2].set_ylabel('Command phase shift (ms)');axes[2].set_title('State-dependent timing is active');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('v0.1.24 candidate rejected: full task passes 1/7, frozen control 1/7',weight='bold')
    fig.supxlabel('* Extra ±0.10 m/s diagnostic cases: both methods 0/2. One rollout per condition; no statistical generalization claim.',fontsize=9)
    fig.savefig(a.figure,dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['status','frozen_seven','extra_velocity_diagnostics']},indent=2))

if __name__=='__main__':main()
