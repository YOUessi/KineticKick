#!/usr/bin/env python3
"""Quick physical loft profiles; measured flight, not calibrated 3-D aim."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]


def summarize_loft(result,index=0):
    rows=result['traces'][index]
    contact=next((r['time_s'] for r in rows if r['contact_force_N']>2),None)
    flight=[];start=land=None
    if contact is not None:
        for row in rows:
            if row['time_s']<contact:continue
            if start is None and row['ball_xyz'][2]>.13 and row['ball_velocity'][2]>0:
                start=row['time_s']
            if start is not None:
                flight.append(row)
                if row['ball_xyz'][2]<=.12 and row['ball_velocity'][2]<=0:
                    land=row['time_s'];break
    apex=max((r['ball_xyz'][2] for r in flight),default=.11)
    duration=(land if land is not None else flight[-1]['time_s'])-start if flight else 0.
    contacts=[r for r in rows if contact is not None and contact<=r['time_s']<contact+.25 and r['contact_force_N']>2]
    released=next((r for r in rows if contacts and r['time_s']>contacts[-1]['time_s']),None)
    direct=any(r['ball_velocity'][2]>.2 for r in contacts)
    illegal=any(force>2 and body not in ('right_ankle_roll_link','left_ankle_roll_link')
        for row in rows for body,force in row.get('robot_ball_contact_force_N',{}).items())
    fell=any(r.get('terminated',False) or r['root_xyz'][2]<.45 or r['root_tilt_deg']>60 for r in rows)
    standing=result.get('sequence_records',[{}]*len(result['traces']))[index].get('recovery',{}).get('final_standing_seconds',0.)
    speed=math.sqrt(sum(v*v for v in released['ball_velocity'])) if released else None
    # Sphere contact-normal estimate, not a measured contact-point reconstruction.
    sum_force=weighted_z=0.
    for row in contacts:
        for event in row.get('contacts',[]):
            q=event.get('foot_quat_w');n=event.get('normal_foot');f=event['force_N']
            if q is None or n is None:continue
            w,x,y,z=q
            normal_z=2*(x*z-w*y)*n[0]+2*(y*z+w*x)*n[1]+(1-2*(x*x+y*y))*n[2]
            weighted_z+=f*normal_z;sum_force+=f
    physical=result.get('root_fixed') is False and result.get('post_initialization_state_writes') is False
    return dict(contact_time_s=contact,first_flight_apex_m=apex,ball_bottom_clearance_m=max(0.,apex-.11),
        first_flight_duration_s=duration,first_landing_s=land,release_speed_mps=speed,
        release_vertical_speed_mps=released['ball_velocity'][2] if released else None,
        estimated_contact_below_center_m=.11*weighted_z/sum_force if sum_force else None,
        direct_foot_lift=direct,illegal_body_ball_contact=illegal,fell=fell,final_standing_seconds=standing,
        loft_success=bool(physical and direct and not illegal and not fell and apex>=.30 and duration>=.30 and standing>=.5),
        scope='Quick demonstration only; no robustness, three-level calibration or 3-D target accuracy claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',default='loft')
    p.add_argument('--config',default=str(ROOT/'configs/lofted_shooting_v1.json'))
    p.add_argument('--seed',type=int,default=1903)
    p.add_argument('--headless',action='store_true');p.add_argument('--video-dir');p.add_argument('--output')
    args=p.parse_args();config_path=Path(args.config).resolve();config=json.loads(config_path.read_text())
    if args.profile not in config['profiles']:p.error('Available profiles: '+', '.join(config['profiles']))
    from show_runup_shooting import verify_runup_config
    verify_runup_config(config,ROOT)
    trial=copy.deepcopy(config['profiles'][args.profile]['trial'])
    trial.update(loft_config=str(config_path),loft_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    output=Path(args.output).resolve() if args.output else ROOT/'reports'/f'loft_{args.profile}_{time.time_ns()}.json'
    if output.exists():p.error('Choose a new result path')
    cmd=[sys.executable,'-u',str(ROOT/'tools/eval_free_shooting.py'),'--sequence-evidence','--steps',str(config['trial_steps']),
         '--seed',str(args.seed),'--style',trial['style'],'--output',str(output)]
    if args.video_dir:cmd+=['--video-dir',str(Path(args.video_dir).resolve())]
    elif not args.headless:cmd+=['--gui','--hold']
    print(config['profiles'][args.profile]['label'],flush=True)
    with tempfile.TemporaryDirectory(prefix='kinetickick-loft-') as d:
        manifest=Path(d)/'trial.json';manifest.write_text(json.dumps([trial]))
        subprocess.run(cmd+['--trials',str(manifest)],cwd=ROOT,check=True)
    if not output.is_file():raise RuntimeError('Simulator produced no completed trial')
    result=json.loads(output.read_text());summary=summarize_loft(result)
    result['loft_records']=[summary];output.write_text(json.dumps(result,indent=2))
    print('MEASURED_LOFT_RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
