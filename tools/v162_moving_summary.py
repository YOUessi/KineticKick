"""Recompute paired rolling-football results from preserved 200 Hz CPU evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from v162_moving_batch import ROOT, digest, readout


def xy_before_kick(folder):
    names=json.loads((folder/'tracking_result.json').read_text())['source_body_names']
    with np.load(folder/'rollout.npz',allow_pickle=False) as tr:
        i=int(np.abs(tr['time_s']-2.2).argmin())
        ankle=tr['body_position_w_m'][i,names.index('left_ankle_roll_link'),:2]
    return ankle.tolist()


def follow_at_impact(folder, score):
    """For a moving ball, also align the frozen follow criteria from impact ball to goal."""
    from v161_evaluate import measured_follow
    first=score['earliest_loaded_right_foot_s']
    if first is None:
        return {'passed':False,'reason':'no right foot impact'}
    names=json.loads((folder/'tracking_result.json').read_text())['source_body_names']
    with np.load(folder/'rollout.npz',allow_pickle=False) as tr:
        time=tr['time_s']; i=int(np.searchsorted(time,first))
        direction=np.r_[np.array(score['goal_xyz_m'])[:2]-tr['ball_position_w_m'][i-1,:2],0.]
        direction/=np.linalg.norm(direction)
        active=np.zeros(len(time),bool)
        for event in score['contact_episodes']:
            active|=(time>=event['first_loaded_bin_s'])&(time<=event['last_loaded_bin_s'])
        foot=tr['body_position_w_m'][:,names.index('right_ankle_roll_link')]
        return measured_follow(time,foot,tr['root_pose_wxyz'],first,direction,active)


def summarize(folder):
    protocol_path=ROOT/'configs/v162_moving_protocol.json'
    selection_path=ROOT/'configs/v162_moving_selection.json'
    protocol=json.loads(protocol_path.read_text()); selections=json.loads(selection_path.read_text())
    rows={}
    for phase in ('development','holdout'):
        for case in protocol[phase]:
            prefix='dev' if phase=='development' else 'held'
            arms=('baseline','candidate','planner') if prefix=='dev' else ('baseline','candidate')
            for arm in arms:
                run=folder/f"{prefix}_{case['id']}_{arm}"
                r=readout(run);old=json.loads((run/'moving_score.json').read_text())
                if r['score']['input_sha256']!=old['score']['input_sha256'] or r['strict_cpu_pass']!=old['strict_cpu_pass']:
                    raise ValueError(f'Physical evidence changed after scoring {run}')
                if arm!='baseline' and (phase=='holdout' or arm=='planner'):
                    if old['selected_strike_sha256']!=selections['candidate_sha256']:
                        raise ValueError('Independent candidate differs from frozen selector')
                scr=r['score'];surface=r['surface'];phys=r['physical'];touch=scr['native_pair_events']['first_wrong_ball_contact']
                rows[run.name]=dict(phase=phase,case=case['id'],arm=arm,
                    strict_pass=r['strict_cpu_pass'],failed=scr['failed_checks'],
                    v161_pass=scr['cpu_proxy_task_pass'],requested_velocity_xy_mps=r['requested_velocity'],
                    measured=phys,all_surface_min=r['surface_minimum'],surface=surface,
                    exit_speed_xy_mps=scr['last_exit_speed_xy_mps'],goal=scr['checks']['goal'],
                    whole_ball_crossing=scr['whole_ball_crossing'],first_disallowed_ball_pair=touch,
                    continuous_forward_s=scr['follow_through'].get('continuous_initial_forward_s'),
                    follow_through_aligned_impact_ball_to_goal=follow_at_impact(run,scr),
                    forward_advance_m=scr['follow_through'].get('forward_advance_m'),
                    terminal_standing_s=scr['terminal_standing_seconds'],
                    q_position_violation_count=scr['q_position_violation_cells'],
                    left_foot_xy_at_2p20s_m=xy_before_kick(run),
                    first_foot_contact_s=scr['earliest_loaded_right_foot_s'],
                    original_source_sha256=scr['input_sha256'])
    pairs={}
    for phase in ('development','holdout'):
        prefix='dev' if phase=='development' else 'held';final='planner' if prefix=='dev' else 'candidate'
        pairs[phase]={}
        for case in protocol[phase]:
            a=rows[f"{prefix}_{case['id']}_baseline"];b=rows[f"{prefix}_{case['id']}_{final}"]
            delta=np.array(b['left_foot_xy_at_2p20s_m'])-a['left_foot_xy_at_2p20s_m']
            pairs[phase][case['id']]=dict(baseline_pass=a['strict_pass'],selected_pass=b['strict_pass'],
                                          left_foot_xy_difference_at_2p20s_m=delta.tolist(),
                                          left_foot_net_difference_at_2p20s_m=float(np.linalg.norm(delta)))
    return {'phase_protocol_sha256':digest(protocol_path),'pre_holdout_selection_sha256':digest(selection_path),
            'scope':'Single official-001 capture area in 14s SAPIEN CPU free-root rollouts. The trained PAiD recurrent gait actor is used as-is; a causal upstream planner supplies its intercept task point. All ball positions and velocities are native physics. v161 12 gates plus all four local instep fractions >=0.95. Development and holdout scenarios are separate, all 12 episodes included.',
            'rows':rows,'pairs':pairs}


def fast_extension(folder):
    fast=ROOT/'configs/v162_moving_fast_holdout.json'
    cfg=json.loads(fast.read_text())
    def compact(run):
        new=readout(run);old=json.loads((run/'moving_score.json').read_text());s=new['score']
        if old['score']['input_sha256']!=s['input_sha256'] or old['strict_cpu_pass']!=new['strict_cpu_pass']:
            raise ValueError(f'Run evidence drifted: {run}')
        return dict(strict=new['strict_cpu_pass'],asked=new['requested_velocity'],
                    incoming=new['physical'],exit_speed_xy_mps=s['last_exit_speed_xy_mps'],
                    foot=new['surface'],goal=s['whole_ball_crossing'],failed=s['failed_checks'],
                    wrong=s['native_pair_events']['first_wrong_ball_contact'],
                    follow_original=s['follow_through'],follow_impact_aligned=follow_at_impact(run,s),
                    standing_s=s['terminal_standing_seconds'],input_sha256=s['input_sha256'])
    pairs={}
    for case in cfg['holdout']:
        pairs[case['id']]={arm:compact(folder/f"fast2_{case['id']}_{arm}")
                           for arm in ('baseline','candidate')}
    return {'holdout_protocol_sha256':digest(fast),'frozen_unchanged_s0_study':'summary.json',
            'prior_failure_reused_as_development':{
                'clearance_off':compact(folder/'held_fast_candidate'),
                'clearance_on':compact(folder/'reused_fast_support')},
            'separate_new_blind_fast_pairs':pairs}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--folder',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--fast-extension',action='store_true')
    args=cli.parse_args()
    result=fast_extension(args.folder) if args.fast_extension else summarize(args.folder)
    with args.output.open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write('\n')
    if args.fast_extension:
        print({case:{arm:x['strict'] for arm,x in arms.items()}
               for case,arms in result['separate_new_blind_fast_pairs'].items()})
    else:
        print(json.dumps(result['pairs'],indent=2))
