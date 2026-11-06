#!/usr/bin/env python3
"""Audit fitted placement episodes against the immutable physical baseline."""
import argparse,hashlib,json,math,subprocess
from strike_cases import load_cases,case_group
from pathlib import Path
from strike_baseline_protocol import verify_files,case_plan
from strike_baseline_evidence import verify_case_evidence
from strike_baseline_metrics import summarize_strike,validated_rows
from closed_loop_acceptance import full_task_success
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['run_placement_strike.py','placement_strike.py','closed_loop_strike.py','closed_loop_expert.py','joint_strike_core.py','paid_actor_warmstart.py','closed_loop_acceptance.py','hip_clearance.py','shank_clearance.py','approach_adapter.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def verify_approach(row,config,previous,joint_names,velocity_delta,expected_feature):
    """Independently reproduce the bounded request and original-limit projection."""
    cfg=config['approach_adapter'];feature=row['approach_feature']
    if not math.isfinite(feature) or abs(feature)>min(1.,max(0.,velocity_delta/.2))+3e-6:raise ValueError('Invalid measured approach feature')
    if not math.isfinite(expected_feature) or abs(feature-expected_feature)>2e-5:raise ValueError('Approach feature differs from measured state')
    weights=[]
    for start,rise,fall,end in config['fast_prefix_windows_s']:
        t=row['time_s'];f=(t-start)/(rise-start) if t<=rise else (end-t)/(end-fall)
        weights.append(.5-.5*math.cos(math.pi*min(1.,max(0.,f))))
    indices=[joint_names.index(side+'_'+part+'_joint') for side in ['left','right'] for part in ['hip_pitch','hip_roll','hip_yaw','ankle_pitch','ankle_roll']]
    actual=row['approach_correction_rad'];requests=row['approach_requested_rad'];before=row['approach_target_before_rad'];limits=row['approach_soft_limits_rad']
    if len(actual)!=len(joint_names) or any(len(x)!=10 for x in [requests,before,limits]):raise ValueError('Incomplete approach evidence')
    if any(not math.isfinite(x) for x in actual) or any(abs(x)>1e-7 for j,x in enumerate(actual) if j not in indices):raise ValueError('Unexpected approach joint')
    for j,index in enumerate(indices):
        old=previous[j];wanted=sum(w*c[j] for w,c in zip(weights,cfg['coefficients']))*feature
        wanted=min(cfg['maximum_rad'],max(-cfg['maximum_rad'],wanted))
        requested=old+min(cfg['maximum_step_rad'],max(-cfg['maximum_step_rad'],wanted-old))
        lo,hi=limits[j];target=before[j]
        if not all(math.isfinite(x) for x in [requests[j],lo,hi,target]) or lo>=hi:raise ValueError('Invalid approach projection evidence')
        expected=min(max(requested,min(lo-target,0.)),max(hi-target,0.))
        if abs(requested-requests[j])>3e-6 or abs(expected-actual[index])>3e-6:raise ValueError('Approach command/projection differs')
    return [actual[j] for j in indices]


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
    if config.get('approach_adapter'):
        import numpy as np
        reference=np.load(ROOT/config['template'])
        state_inputs=ev.get('state_inputs',[])
        if len(state_inputs)!=len(feedback):raise ValueError('Missing independent approach state samples')
    last_phase=None;last_shift=0.;last_shank=0.;last_ankle=0.;last_approach=[0.]*10
    pulse_end=json.loads((ROOT/config['expert_plan']).read_text())['strike_velocity']['window'][-1]
    for i,row in enumerate(feedback):
        t=i*.02;phase=row['phase_s'];actual_shift=phase-t
        if config.get('approach_adapter'):
            inputs=state_inputs[i];v=inputs['root_velocity_w']
            if abs(inputs['time_s']-t)>2e-6 or len(v)!=3 or not all(math.isfinite(x) for x in v):raise ValueError('Invalid independent approach state sample')
            if i==0 and any(abs(a-b)>1e-6 for a,b in zip(v,velocity)):raise ValueError('Approach reset velocity differs')
            root,ball=states[round(t,6)];observed=shift[0] if first<=t+1e-8 else ball[0]-config['ball_xy'][0]
            observed=min(.15,max(-.15,observed));ramp=min(1.,t/1.2);ramp=ramp*ramp*(3.-2.*ramp)
            k=min(i,len(reference['root_pose'])-1);ac=config['approach_adapter']
            error=float(reference['root_pose'][k,0])+observed*ramp-root[0]+ac['lookahead_s']*(float(reference['root_velocity'][k,0])-v[0])
            expected_feature=min(1.,max(-1.,error/ac['scale_m']))*min(1.,max(0.,delta[1]/.2))
            last_approach=verify_approach(row,config,last_approach,candidate['joint_names'],delta[1],expected_feature)
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
        shank=config.get('shank_clearance')
        if shank:
            keys=['shank_correction_rad','shank_ankle_correction_rad','shank_gap_m','shank_predicted_gap_m','shank_closest_time_s']
            if any(k not in row or not math.isfinite(row[k]) for k in keys) or 'shank_active' not in row:raise ValueError('Missing shank path evidence')
            sd=row['shank_correction_rad'];sa=row['shank_ankle_correction_rad']
            if max(abs(sd),abs(sa))>shank['maximum_correction_rad']+3e-6:raise ValueError('Shank command bound violated')
            requests=row['shank_requested_rad'];before=row['shank_target_before_rad'];limits=row['shank_soft_limits_rad']
            if len(requests)!=2 or len(before)!=2 or len(limits)!=2:raise ValueError('Missing two-channel shank projection evidence')
            for current,request,old,target,limit in zip([sd,sa],requests,[last_shank,last_ankle],before,limits):
                if len(limit)!=2 or not all(math.isfinite(x) for x in [request,target,*limit]) or limit[0]>=limit[1]:raise ValueError('Invalid shank actuator projection evidence')
                if abs(request-old)>shank['maximum_step_rad']+3e-6 or abs(request)>shank['maximum_correction_rad']+3e-6:raise ValueError('Shank requested rate/bound violated')
                expected=min(max(request,min(limit[0]-target,0.)),max(limit[1]-target,0.))
                if abs(current-expected)>3e-6:raise ValueError('Shank actuator-limit projection differs')
                if not row['shank_active'] and (abs(current)>abs(old)+3e-6 or current*old<-1e-8):raise ValueError('Inactive shank channel starts a new correction')
            if not 0.<=row['shank_closest_time_s']<=shank['horizon_s']+1e-6:raise ValueError('Invalid swept horizon')
            if row['shank_active']:
                physical=physical_by_time[round(t,6)];support=physical['support_foot_force_N']
                if gate==0 or first<=t or t<shank['start_s']-1e-6 or t>=shank['end_s']+1e-6 or support['left']<shank['support_min_N']-1e-4 or support['right']>shank['swing_max_N']+1e-4:raise ValueError('Shank correction without precontact support')
            elif abs(sd)>abs(last_shank)+3e-6 or sd*last_shank<-1e-8:raise ValueError('Inactive shank control is not releasing')
            if gate==0. and (sd or sa):raise ValueError('Nominal shank protection violated')
            swing[candidate['joint_names'].index('right_hip_roll_joint')]-=sd
            swing[candidate['joint_names'].index('right_ankle_roll_joint')]-=sa
            last_shank=sd;last_ankle=sa

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
    parser.add_argument('--reference-version',choices=['v0.1.15','v0.1.26','v0.1.27'],default='v0.1.15')
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
    prior_commit='a0dfb47cf00a1cb198dce857825c13b2d8dc01b3' if args.reference_version=='v0.1.27' else '92141f640982f370e5fc78f0e154ce5cb7ed6f67'
    if args.reference_version in ['v0.1.26','v0.1.27']:
        def content(path):return subprocess.check_output(['git','show',prior_commit+':'+path],cwd=ROOT)
        prior_config_bytes=content('configs/state_conditioned_strike_v4.json' if args.reference_version=='v0.1.27' else 'configs/state_conditioned_strike_v1.json')
        prior_config=json.loads(prior_config_bytes);prior_sha=hashlib.sha256(prior_config_bytes).hexdigest()
        prior_sources={name:hashlib.sha256(content('tools/'+name)).hexdigest() for name in SOURCES if name not in ['shank_clearance.py','approach_adapter.py'] and (name!='hip_clearance.py' or args.reference_version=='v0.1.27')}
    rows=[]
    for name in args.cases:
        case=cases[name];rp=ROOT/'reports'/f'{args.prefix}_{name}.json'
        candidate=json.loads(rp.read_text())
        # The declared perturbation changes only initial ball/velocity. All other
        # physical inputs are checked against the verified original center run.
        verify(candidate,center,case,config,sha(args.config),sources,nominal)
        bp=ROOT/'reports'/f"{'v023_baseline' if args.reference_version=='v0.1.15' else ('v027_final' if args.reference_version=='v0.1.27' else 'v026_final')}_{name}.json"
        baseline=None
        if args.reference_version=='v0.1.15' and name in frozen_cases:
            baseline=json.loads(bp.read_text())
            verify_case_evidence(baseline,case,sha(mp),old_sources,case_plan(plan,case),center)
        elif (args.reference_version=='v0.1.27' and not name.startswith('heldout_v028_')) or (args.reference_version=='v0.1.26' and (name in frozen_cases or name in ['combined_left_slower','combined_right_faster'])):
            baseline=json.loads(bp.read_text())
            verify(baseline,center,case,prior_config,prior_sha,prior_sources,nominal)
        rows.append(dict(case=case,case_group=case_group(name,config['version']),
            baseline=None if baseline is None else summarize_strike(baseline),candidate=summarize_strike(candidate),
            baseline_report=None if baseline is None else str(bp.relative_to(ROOT)),candidate_report=str(rp.relative_to(ROOT)),
            baseline_sha256=None if baseline is None else sha(bp),candidate_sha256=sha(rp)))
    comparable=[row for row in rows if row['baseline'] is not None]
    evaluation=[row for row in rows if row['case_group']=='evaluation']
    result=dict(version=config['version'],cases=rows,total=len(rows),
        evaluation_total=len(evaluation),evaluation_passed=sum(row['candidate']['outcome']['success'] for row in evaluation),
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
