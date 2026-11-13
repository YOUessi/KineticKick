"""Run and score matched 14-second physical rolling-ball trajectories; retain every trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import v161_evaluate
import v161_surface_audit

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(case, protocol):
    velocity = np.array(case['velocity_xy_mps'])
    offset = np.asarray(case['crossing_offset_xy_m']) - protocol['arrival_time_s']*velocity
    return offset, velocity


def readout(path):
    score = v161_evaluate.report(path)
    surface = v161_surface_audit.summarize(path)
    trial = json.loads((path/'trial.json').read_text())
    with np.load(path/'rollout.npz', allow_pickle=False) as tr:
        t = tr['time_s']
        pos = tr['ball_position_w_m']
        vel = tr['ball_velocity_w_mps']
        first = score['earliest_loaded_right_foot_s']
        i = np.searchsorted(t, first) - 1 if first is not None else np.abs(t-2.395).argmin()
        raw_touch = np.flatnonzero(tr['robot_ball_contact'])
        prior_to_any = int(raw_touch[0]-1) if len(raw_touch) else i
        if i < 0:
            raise ValueError('No previous physical observation before first contact')
        init = np.array(trial['ball_initial_xyz_m'])
        asked = np.array(trial['ball_initial_velocity_xy_mps'])
        if np.max(abs(vel[0,:2] - asked)) > 1e-5:
            raise ValueError('Measured first-bin velocity differs from requested physical motion')
        measured = dict(first_t_s=t[0],first_velocity_world_mps=vel[0].tolist(),
                        first_spin_world_rad_s=tr['ball_angular_velocity_w_rad_s'][0].tolist(),
                        first_any_robot_ball_s=float(t[raw_touch[0]]) if len(raw_touch) else None,
                        pre_any_robot_ball_velocity_xy_mps=vel[prior_to_any,:2].tolist(),
                        pre_any_robot_ball_travel_xy_m=(pos[prior_to_any,:2]-init[:2]).tolist(),
                        pre_any_robot_ball_speed_mps=float(np.linalg.norm(vel[prior_to_any,:2])),
                        precontact_or_nominal_t_s=t[i],was_precontact=first is not None,
                        precontact_velocity_xy_mps=vel[i,:2].tolist(),
                        precontact_speed_xy_mps=float(np.linalg.norm(vel[i,:2])),
                        precontact_ball_xy_m=pos[i,:2].tolist(),
                        travel_xy_m=(pos[i,:2]-init[:2]).tolist(),
                        travelled_horizontal_distance_m=float(np.linalg.norm(pos[i,:2]-init[:2])))
    values=surface['instep_normal_impulse_fractions']
    minimum=min(values.values()) if all(v is not None for v in values.values()) else None
    return dict(run=path.name, requested_velocity=asked.tolist(), ball_initial_xyz_m=init.tolist(),
                physical=measured, surface=surface, surface_minimum=minimum, score=score,
                strict_cpu_pass=bool(score['cpu_proxy_task_pass'] and minimum is not None and minimum>=.95))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path,default=ROOT/'configs/v162_moving_protocol.json')
    parser.add_argument('--phase', choices=('development','holdout'), required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--arm', choices=('baseline','candidate'), required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    case=next(c for c in protocol[args.phase] if c['id']==args.case)
    cfg=args.candidate if args.arm=='candidate' and args.candidate else ROOT/protocol[args.arm]
    offset,velocity=geometry(case,protocol)
    cmd=[sys.executable,str(ROOT/'tools/v161_run.py'),'--strike-config',str(cfg),
         '--recovery-config',str(ROOT/protocol['recovery']),'--ball-offset',*map(str,offset),
         '--ball-velocity',*map(str,velocity),'--output',str(args.output)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
             CUDA_VISIBLE_DEVICES='')
    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True)
    if args.output.is_dir():
        with (args.output/'runner_stdout.log').open('x') as out:
            out.write(proc.stdout)
    proc.check_returncode()
    result=readout(args.output)
    result.update(case=case,phase=args.phase,arm=args.arm,
                  protocol_sha256=digest(args.protocol),batch_script_sha256=digest(Path(__file__)),
                  selected_strike_sha256=digest(cfg))
    with (args.output/'moving_score.json').open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write('\n')
    print(json.dumps(dict(case=args.case,arm=args.arm,strict=result['strict_cpu_pass'],
                          failed=result['score']['failed_checks'],surface=result['surface_minimum'],
                          measured=result['physical'])))


if __name__=='__main__':
    main()
