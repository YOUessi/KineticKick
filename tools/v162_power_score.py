"""Paired command outcome, using full physical evidence and unmodified v161 safety rules."""
import argparse
import json
from pathlib import Path
import numpy as np

from v161_evaluate import report
from v161_surface_audit import summarize as surface_report
from v160_tracking_ball_outcomes import crossing
from v160_tracking_cpu import PROTOCOL


def measure(folder, speed, aim):
    r=report(folder);surface=surface_report(folder)
    cfg=json.loads(PROTOCOL.read_text())
    with np.load(folder/'rollout.npz') as data:
        t=np.r_[0.,data['time_s']]; p=np.vstack((r['initial_ball_xyz_m'],data['ball_position_w_m']))
        center=crossing(p,t,np.array(r['goal_xyz_m']),np.array(r['goal_forward_xy']),0.,cfg['ball_radius_m'],r['goal_width_height_m'])
    x=center['ball_centre_xyz_m'][0] if center else None
    v=r['last_exit_speed_xy_mps'];fractions=surface['instep_normal_impulse_fractions'].values()
    checks=dict(physical=r['cpu_proxy_task_pass'],surface=all(z is not None and z>=.95 for z in fractions),
                within_speed=v is not None and abs(v-speed)<=.18,within_aim=x is not None and abs(x-aim)<=.1)
    return {'request':[speed,aim],'speed_mps':v,'goal_plane_x_m':x,'checks':checks,'passes':all(checks.values()),
            'score':r,'surface':surface}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--speed',type=float)
    parser.add_argument('--aim',type=float)
    args=parser.parse_args();trial=json.loads((args.run/'trial.json').read_text())
    speed=trial.get('shot_speed_mps') if args.speed is None else args.speed
    aim=trial.get('aim_x_m') if args.aim is None else args.aim
    if speed is None or aim is None:
        parser.error('Provide --speed and --aim when the run has no recorded command')
    result=measure(args.run,speed,aim)
    with args.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(result['checks'],result['speed_mps'],result['goal_plane_x_m'],flush=True)
