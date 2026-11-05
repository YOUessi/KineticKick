#!/usr/bin/env python3
"""Audit fitted placement episodes against the immutable physical baseline."""
import argparse,hashlib,json,math
from pathlib import Path
from strike_baseline_protocol import verify_files,case_plan
from strike_baseline_evidence import verify_case_evidence
from strike_baseline_metrics import summarize_strike,validated_rows
from closed_loop_acceptance import full_task_success
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['run_placement_strike.py','placement_strike.py','closed_loop_strike.py','closed_loop_expert.py','joint_strike_core.py','paid_actor_warmstart.py','closed_loop_acceptance.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(candidate,baseline,case,config,config_sha,sources,nominal_velocity):
    ev=candidate['joint_strike']
    if ev['mode']!='fitted_placement' or ev['config']!=config or ev['config_sha256']!=config_sha or ev['source_sha256']!=sources:raise ValueError('Execution model/source mismatch')
    if candidate['policy_sha256']!=config_sha or ev['expert_backend']!='onnx':raise ValueError('Wrong active fitted model or expert backend')
    if ev['ball_offset_xy']!=case['ball_offset_xy_m'] or ev['velocity_y_delta']!=case['root_velocity_y_delta_mps']:raise ValueError('Requested condition differs')
    expected=baseline['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0]
    velocity=ev['initial_root_velocity']
    if len(velocity)!=3 or any(abs(a-b)>1e-6 for a,b in zip(velocity,expected)):raise ValueError('Measured initial velocity differs')
    for key in ['initial_robot_state','asset_sha256','joint_effort_limits','joint_velocity_limits','points','goals','goal_width_m','goal_height_m','controller_sources_sha256','motion_sources_sha256','sequence_sources_sha256','termination_policy','contact_geometry','config_sha256']:
        if candidate.get(key)!=baseline[key]:raise ValueError('Frozen physics/runtime differs: '+key)
    if candidate['root_fixed'] is not False or candidate['post_initialization_state_writes'] is not False:raise ValueError('Physical integrity violated')
    if candidate['stride_shot_success'] is not full_task_success(candidate['running_evidence'],candidate['stride_shot_evidence'],candidate['sequence_records'][0],ev['expert_handoffs']):raise ValueError('Original full acceptance gates differ')
    feedback=ev['feedback'];rows=validated_rows(candidate['traces'][0]);poses=candidate['kinematics'][0]
    if len(feedback)!=len(poses) or len(feedback)!=math.ceil(rows[-1]['time_s']/.02-1e-7):raise ValueError('Feedback does not cover executed physics')
    states={round(r['time_s'],6):(r['root_xyz'],r['ball_xyz']) for r in rows}
    states[0.]=(candidate['initial_robot_state']['root_position'][0],[*candidate['points'][0],.113])
    first=min((r['time_s'] for r in rows if r['robot_ball_contact_force_N'].get('right_ankle_roll_link',0)>2),default=math.inf)
    delta=[x-y for x,y in zip(velocity,nominal_velocity)];shift=case['ball_offset_xy_m'];dead=lambda x:0. if abs(x)<=1e-4 else x
    gate=min(1.,math.sqrt((dead(shift[0])/.05)**2+(dead(shift[1])/.10)**2+sum((dead(v)/.20)**2 for v in delta)))
    last_phase=None;last_shift=0.
    for i,row in enumerate(feedback):
        t=i*.02;phase=row['phase_s'];actual_shift=phase-t
        scalars=[row['time_s'],phase,row['context_gate'],*row['measured_root_xy'],*row['measured_ball_xyz'],*row['residual_rad'],*row['stance_correction_rad'],*row['prefix_correction_rad'],*row['placement_command_xyz']]
        if not all(math.isfinite(x) for x in scalars):raise ValueError('Nonfinite control evidence')
        if abs(row['time_s']-t)>2e-6 or abs(poses[i]['time_s']-t)>2e-6:raise ValueError('Shifted feedback timestamp')
        if abs(actual_shift)>.120003 or abs(actual_shift-last_shift)>.008003 or (last_phase is not None and phase<last_phase):raise ValueError('Phase correction bound/causality violated')
        if row['contact_latched']!=(first<=t+1e-8):raise ValueError('Contact latch differs from prior physics')
        if (t>=2.1 or first<=t-.02+1e-8) and abs(actual_shift-last_shift)>3e-6:raise ValueError('Phase changed after contact/update window')
        if max(map(abs,row['residual_rad']))>config.get('maximum_swing_correction_rad',.25)+3e-6 or max(map(abs,row['prefix_correction_rad']))>.200003 or max(map(abs,row['stance_correction_rad']))>.150003 or max(map(abs,row['placement_command_xyz']))>.400003:raise ValueError('Command bound exceeded')
        measured=states[round(t,6)]
        if any(abs(a-b)>1e-5 for a,b in zip(row['measured_root_xy'],measured[0][:2])) or any(abs(a-b)>1e-5 for a,b in zip(row['measured_ball_xyz'],measured[1])):raise ValueError('Feedback state differs from measured physics')
        if abs(row['context_gate']-gate)>1e-5 or any(abs(a-b)>1e-6 for a,b in zip(row['measured_initial_velocity_error'],delta)):raise ValueError('Context differs from measured reset')
        if gate==0 and (any(row['residual_rad']) or any(row['stance_correction_rad']) or any(row['prefix_correction_rad']) or any(row['placement_command_xyz']) or abs(actual_shift)>2e-6):raise ValueError('Nominal expert protection violated')
        last_phase=phase;last_shift=actual_shift


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--prefix',required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if Path(a.prefix).name!=a.prefix:p.error('Prefix must be a filename component')
    mp=ROOT/'configs/strike_baseline_v1.json';matrix=json.loads(mp.read_text());verify_files(ROOT,matrix['frozen_files_sha256'])
    config=json.loads(a.config.read_text());cases={c['id']:c for c in matrix['cases']};plan=json.loads((ROOT/matrix['baseline_plan']).read_text())
    if len(set(a.cases))!=len(a.cases) or any(n not in cases for n in a.cases):p.error('Unique frozen case ids required')
    for key in ['template','reference','expert_plan']:
        if sha(ROOT/config[key])!=config[key+'_sha256']:raise ValueError('Changed '+key)
    old_sources={k:sha(ROOT/v) for k,v in [('evaluator_sha256','tools/run_strike_baseline_case.py'),('protocol_sha256','tools/strike_baseline_protocol.py')]}
    sources={n:sha(ROOT/'tools'/n) for n in SOURCES};center=json.loads((ROOT/'reports/v023_baseline_center.json').read_text())
    verify_case_evidence(center,cases['center'],sha(mp),old_sources,case_plan(plan,cases['center']))
    nominal=center['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0];rows=[]
    for name in a.cases:
        case=cases[name];bp=ROOT/'reports'/f'v023_baseline_{name}.json';rp=ROOT/'reports'/f'{a.prefix}_{name}.json'
        baseline=json.loads(bp.read_text());candidate=json.loads(rp.read_text())
        verify_case_evidence(baseline,case,sha(mp),old_sources,case_plan(plan,case),center)
        verify(candidate,baseline,case,config,sha(a.config),sources,nominal)
        rows.append(dict(case=case,baseline=summarize_strike(baseline),candidate=summarize_strike(candidate),baseline_report=str(bp.relative_to(ROOT)),candidate_report=str(rp.relative_to(ROOT)),baseline_sha256=sha(bp),candidate_sha256=sha(rp)))
    result=dict(version='0.1.26',cases=rows,total=len(rows),baseline_passed=sum(x['baseline']['outcome']['success'] for x in rows),candidate_passed=sum(x['candidate']['outcome']['success'] for x in rows),sources=sources,model_sha256=sha(a.config),analysis_sha256=sha(__file__),unexecuted_cases=[n for n in cases if n not in a.cases],interpretation='Prespecified development cases used in fitting; not held-out success rates. Protected original condition is expert retention, not newly learned robustness.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['baseline_passed','candidate_passed','total']}))

if __name__=='__main__':main()
