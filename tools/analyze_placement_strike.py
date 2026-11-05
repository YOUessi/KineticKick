#!/usr/bin/env python3
"""Audit fitted placement episodes against the immutable physical baseline."""
import argparse,hashlib,json,math,subprocess
from strike_cases import load_cases
from pathlib import Path
from strike_baseline_protocol import verify_files,case_plan
from strike_baseline_evidence import verify_case_evidence
from strike_baseline_metrics import summarize_strike,validated_rows
from closed_loop_acceptance import full_task_success
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['run_placement_strike.py','placement_strike.py','closed_loop_strike.py','closed_loop_expert.py','joint_strike_core.py','paid_actor_warmstart.py','closed_loop_acceptance.py','hip_clearance.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(candidate,baseline,case,config,config_sha,sources,nominal_velocity):
    ev=candidate['joint_strike']
    if ev['mode']!='fitted_placement' or ev['config']!=config or ev['config_sha256']!=config_sha or ev['source_sha256']!=sources:raise ValueError('Execution model/source mismatch')
    if candidate['policy_sha256']!=config_sha or ev['expert_backend']!='onnx':raise ValueError('Wrong active fitted model or expert backend')
    if ev['ball_offset_xy']!=case['ball_offset_xy_m'] or ev['velocity_y_delta']!=case['root_velocity_y_delta_mps']:raise ValueError('Requested condition differs')
    expected=list(nominal_velocity);expected[1]+=case['root_velocity_y_delta_mps']
    expected_points=[[v+d for v,d in zip(config['ball_xy'],case['ball_offset_xy_m'])]]
    if candidate['points']!=expected_points:raise ValueError('Measured initial ball position differs')
    velocity=ev['initial_root_velocity']
    if len(velocity)!=3 or any(abs(a-b)>1e-6 for a,b in zip(velocity,expected)):raise ValueError('Measured initial velocity differs')
    for key in ['initial_robot_state','asset_sha256','joint_effort_limits','joint_velocity_limits','goals','goal_width_m','goal_height_m','controller_sources_sha256','motion_sources_sha256','sequence_sources_sha256','termination_policy','contact_geometry','config_sha256']:
        if candidate.get(key)!=baseline[key]:raise ValueError('Frozen physics/runtime differs: '+key)
    if candidate['root_fixed'] is not False or candidate['post_initialization_state_writes'] is not False:raise ValueError('Physical integrity violated')
    if candidate['stride_shot_success'] is not full_task_success(candidate['running_evidence'],candidate['stride_shot_evidence'],candidate['sequence_records'][0],ev['expert_handoffs']):raise ValueError('Original full acceptance gates differ')
    feedback=ev['feedback'];rows=validated_rows(candidate['traces'][0]);poses=candidate['kinematics'][0]
    if len(feedback)!=len(poses) or len(feedback)!=math.ceil(rows[-1]['time_s']/.02-1e-7):raise ValueError('Feedback does not cover executed physics')
    physical_by_time={round(r['time_s'],6):r for r in rows}
    states={round(r['time_s'],6):(r['root_xyz'],r['ball_xyz']) for r in rows}
    states[0.]=(candidate['initial_robot_state']['root_position'][0],[*candidate['points'][0],.113])
    first=min((r['time_s'] for r in rows if r['robot_ball_contact_force_N'].get('right_ankle_roll_link',0)>2),default=math.inf)
    delta=[x-y for x,y in zip(velocity,nominal_velocity)];shift=case['ball_offset_xy_m'];dead=lambda x:0. if abs(x)<=1e-4 else x
    gate=min(1.,math.sqrt((dead(shift[0])/.05)**2+(dead(shift[1])/.10)**2+sum((dead(v)/.20)**2 for v in delta)))
    last_phase=None;last_shift=0.
    pulse_end=json.loads((ROOT/config['expert_plan']).read_text())['strike_velocity']['window'][-1]
    for i,row in enumerate(feedback):
        t=i*.02;phase=row['phase_s'];actual_shift=phase-t
        scalars=[row['time_s'],phase,row['context_gate'],*row['measured_root_xy'],*row['measured_ball_xyz'],*row['residual_rad'],*row['stance_correction_rad'],*row['prefix_correction_rad'],*row['placement_command_xyz'],row.get('follow_through_correction_rad',0.),row.get('thigh_pelvis_gap_m',0.),row.get('hip_stopping_angle_rad',0.),row.get('hip_clearance_floor_rad',0.)]
        if config.get('project_swing_target'):
            fraction=row.get('swing_reach_fraction')
            if fraction is None or not math.isfinite(fraction) or not 0.<=fraction<=1.:raise ValueError('Missing or invalid feasible swing fraction')
            if fraction>0. and (row['ik_error_m']>=.025 or row['ik_angle_error_rad']>=.12):raise ValueError('Projected target violates original IK acceptance')
        if not all(math.isfinite(x) for x in scalars):raise ValueError('Nonfinite control evidence')
        if abs(row['time_s']-t)>2e-6 or abs(poses[i]['time_s']-t)>2e-6:raise ValueError('Shifted feedback timestamp')
        if abs(actual_shift)>.120003 or abs(actual_shift-last_shift)>.008003 or (last_phase is not None and phase<last_phase):raise ValueError('Phase correction bound/causality violated')
        if row['contact_latched']!=(first<=t+1e-8):raise ValueError('Contact latch differs from prior physics')
        if (t>=2.1 or first<=t-.02+1e-8) and abs(actual_shift-last_shift)>3e-6:raise ValueError('Phase changed after contact/update window')
        follow=row.get('follow_through_correction_rad',0.);follow_cfg=config.get('follow_through_clearance',{})
        if follow_cfg and any(key not in row for key in ['follow_through_active','follow_through_correction_rad','thigh_pelvis_gap_m','hip_stopping_angle_rad','hip_clearance_floor_rad']):raise ValueError('Missing post-contact clearance evidence')
        if follow<0. or follow>follow_cfg.get('maximum_correction_rad',0.)+3e-6:raise ValueError('Post-contact hip correction bound exceeded')
        swing=list(row['residual_rad']);pitch=candidate['joint_names'].index('right_hip_pitch_joint');swing[pitch]-=follow
        if follow>0.:
            physical=physical_by_time[round(t,6)]
            command_state=physical_by_time[round(t+physical['physics_dt_s'],6)]
            if not row['follow_through_active'] or phase/.02+1.<pulse_end-1e-6 or first>t or physical['contact_force_N']>2. or gate==0. or t>=follow_cfg['latest_s'] or command_state['joint_velocity_target'][pitch]<-1e-6:raise ValueError('Hip braking before separation, in nominal case, or against strike pulse')
            q=physical['joint_pos'][pitch];v=physical['joint_vel'][pitch]
            predicted=q-min(v,0.)**2/(2.*follow_cfg['deceleration_rad_s2'])
            if v>=0. or abs(predicted-row.get('hip_stopping_angle_rad',0.))>3e-6:raise ValueError('Hip braking prediction differs from measured state')
        if max(map(abs,swing))>config.get('maximum_swing_correction_rad',.25)+3e-6 or max(map(abs,row['prefix_correction_rad']))>.200003 or max(map(abs,row['stance_correction_rad']))>.150003 or max(map(abs,row['placement_command_xyz']))>.400003:raise ValueError('Command bound exceeded')
        measured=states[round(t,6)]
        if any(abs(a-b)>1e-5 for a,b in zip(row['measured_root_xy'],measured[0][:2])) or any(abs(a-b)>1e-5 for a,b in zip(row['measured_ball_xyz'],measured[1])):raise ValueError('Feedback state differs from measured physics')
        if abs(row['context_gate']-gate)>1e-5 or any(abs(a-b)>1e-6 for a,b in zip(row['measured_initial_velocity_error'],delta)):raise ValueError('Context differs from measured reset')
        if gate==0 and (any(row['residual_rad']) or any(row['stance_correction_rad']) or any(row['prefix_correction_rad']) or follow!=0. or any(row['placement_command_xyz']) or abs(actual_shift)>2e-6):raise ValueError('Nominal expert protection violated')
        last_phase=phase;last_shift=actual_shift


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--cases',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-version',choices=['v0.1.15','v0.1.26'],default='v0.1.15')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if Path(args.prefix).name!=args.prefix:parser.error('Prefix must be a filename component')
    mp=ROOT/'configs/strike_baseline_v1.json';matrix=json.loads(mp.read_text());verify_files(ROOT,matrix['frozen_files_sha256'])
    config=json.loads(args.config.read_text());cases=load_cases(ROOT)
    frozen_cases={case['id']:case for case in matrix['cases']}
    plan=json.loads((ROOT/matrix['baseline_plan']).read_text())
    if len(set(args.cases))!=len(args.cases) or any(name not in cases for name in args.cases):parser.error('Unique declared case ids required')
    for key in ['template','reference','expert_plan']:
        if sha(ROOT/config[key])!=config[key+'_sha256']:raise ValueError('Changed '+key)
    old_sources={key:sha(ROOT/path) for key,path in [('evaluator_sha256','tools/run_strike_baseline_case.py'),('protocol_sha256','tools/strike_baseline_protocol.py')]}
    sources={name:sha(ROOT/'tools'/name) for name in SOURCES}
    center=json.loads((ROOT/'reports/v023_baseline_center.json').read_text())
    verify_case_evidence(center,cases['center'],sha(mp),old_sources,case_plan(plan,cases['center']))
    nominal=center['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0]
    prior_commit='92141f640982f370e5fc78f0e154ce5cb7ed6f67'
    if args.reference_version=='v0.1.26':
        def content(path):return subprocess.check_output(['git','show',prior_commit+':'+path],cwd=ROOT)
        prior_config_bytes=content('configs/state_conditioned_strike_v1.json')
        prior_config=json.loads(prior_config_bytes);prior_sha=hashlib.sha256(prior_config_bytes).hexdigest()
        prior_sources={name:hashlib.sha256(content('tools/'+name)).hexdigest() for name in SOURCES if name!='hip_clearance.py'}
    rows=[]
    for name in args.cases:
        case=cases[name];rp=ROOT/'reports'/f'{args.prefix}_{name}.json'
        candidate=json.loads(rp.read_text())
        # The declared perturbation changes only initial ball/velocity. All other
        # physical inputs are checked against the verified original center run.
        verify(candidate,center,case,config,sha(args.config),sources,nominal)
        bp=ROOT/'reports'/f"{'v023_baseline' if args.reference_version=='v0.1.15' else 'v026_final'}_{name}.json"
        baseline=None
        if args.reference_version=='v0.1.15' and name in frozen_cases:
            baseline=json.loads(bp.read_text())
            verify_case_evidence(baseline,case,sha(mp),old_sources,case_plan(plan,case),center)
        elif args.reference_version=='v0.1.26' and (name in frozen_cases or name in ['combined_left_slower','combined_right_faster']):
            baseline=json.loads(bp.read_text())
            verify(baseline,center,case,prior_config,prior_sha,prior_sources,nominal)
        rows.append(dict(case=case,case_group='new_combination' if name.startswith('heldout_') else 'development',
            baseline=None if baseline is None else summarize_strike(baseline),candidate=summarize_strike(candidate),
            baseline_report=None if baseline is None else str(bp.relative_to(ROOT)),candidate_report=str(rp.relative_to(ROOT)),
            baseline_sha256=None if baseline is None else sha(bp),candidate_sha256=sha(rp)))
    comparable=[row for row in rows if row['baseline'] is not None]
    result=dict(version=config['version'],cases=rows,total=len(rows),
        baseline_version=args.reference_version,baseline_total=len(comparable),
        baseline_passed=sum(row['baseline']['outcome']['success'] for row in comparable),
        candidate_passed=sum(row['candidate']['outcome']['success'] for row in rows),
        matched_candidate_passed=sum(row['candidate']['outcome']['success'] for row in comparable),
        sources=sources,model_sha256=sha(args.config),analysis_sha256=sha(__file__),
        case_registry_sha256=sha(ROOT/'tools/strike_cases.py'),unexecuted_cases=[name for name in cases if name not in args.cases],
        interpretation='Declared deterministic development/new combinations, not population success rates. Missing historical comparisons are explicitly null, not inferred failures.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['baseline_passed','baseline_total','candidate_passed','total']}))

if __name__=='__main__':main()
