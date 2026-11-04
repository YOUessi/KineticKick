"""Measure physical leg follow-through separately from ball motion and labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from shooting_posture import audit_posture, check_posture_acceptance


FOLLOW_THROUGH_CRITERIA = dict(advance_window_s=.25, forward_advance_min_m=.22,
    pelvis_relative_advance_min_m=.15, forward_speed_time_s=.12, forward_speed_min_mps=.60,
    earliest_sustained_reversal_s=.20, reversal_speed_threshold_mps=.15,
    peak_return_speed_max_mps=2., return_window_start_s=.22, return_window_end_s=.45,
    recovery_window_end_s=.75, recovery_return_speed_max_mps=2.5,
    late_contact_start_s=.08, late_contact_end_s=.75, late_contact_force_N=2.)


def _motion(run, index, contact_time, direction):
    samples = run['kinematics'][index]
    end_time = FOLLOW_THROUGH_CRITERIA['recovery_window_end_s']
    times = np.array([row['time_s'] for row in samples], dtype=float)
    if (not len(times) or not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0) or
            times[0]>contact_time-.04+1e-8 or times[-1]<contact_time+end_time-1e-8):
        raise ValueError('Missing finite bracketing kinematics')
    first=max(int(np.searchsorted(times,contact_time-.04+1e-8,side='right'))-1,0)
    last=int(np.searchsorted(times,contact_time+end_time-1e-8))
    if np.any(np.diff(times[first:last+1])>.021):
        raise ValueError('Missing20ms motion samples')
    window=samples[first:last+1]
    feet=np.array([row['feet']['right']['position'] for row in window],dtype=float)
    roots=np.array([row['root_pose'][:3] for row in window],dtype=float)
    if feet.shape!=(len(window),3) or roots.shape!=feet.shape or not np.all(np.isfinite([feet,roots])):
        raise ValueError('Invalid measured foot/root positions')
    times=times[first:last+1]-contact_time
    # Rebase first: each environment's constant root/world offset cancels.
    foot=(feet-feet[0])@direction
    root=(roots-roots[0])@direction
    foot-=np.interp(0.,times,foot)
    root-=np.interp(0.,times,root)
    end=FOLLOW_THROUGH_CRITERIA['advance_window_s']
    knots=np.r_[0.,times[(times>0)&(times<end)],end]
    progress=np.interp(knots,times,foot)
    relative=progress-np.interp(knots,times,root)
    speed_time=FOLLOW_THROUGH_CRITERIA['forward_speed_time_s']
    speed=(np.interp(speed_time+.01,times,foot)-np.interp(speed_time-.01,times,foot))/.02
    grid=np.r_[np.arange(0.,end_time,.02),end_time]
    velocities=np.diff(np.interp(grid,times,foot))/np.diff(grid)
    centers=(grid[:-1]+grid[1:])/2
    backwards=velocities < -FOLLOW_THROUGH_CRITERIA['reversal_speed_threshold_mps']
    # Two consecutive20ms intervals distinguish reversal from one noisy sample.
    reversal=next((float(grid[i]) for i in range(len(backwards)-1)
                   if backwards[i] and backwards[i+1]),None)
    selected=(centers>=FOLLOW_THROUGH_CRITERIA['return_window_start_s']) & (centers<=FOLLOW_THROUGH_CRITERIA['return_window_end_s'])
    recovery = centers >= FOLLOW_THROUGH_CRITERIA['return_window_start_s']
    return dict(forward_advance_m=float(max(progress)), pelvis_relative_advance_m=float(max(relative)),
                forward_speed_at_120ms_mps=float(speed), reversal_after_s=reversal,
                peak_return_speed_mps=float(max(0.,-min(velocities[selected]))),
                peak_recovery_return_speed_mps=float(max(0.,-min(velocities[recovery]))))


def audit_follow_through(run, index):
    result=audit_posture(run,index)
    result.update(follow_through_ok=False,follow_through_and_full_chain=False,follow_through_failures=[])
    ct=result['contact_time_s']
    if ct is None or 'toe_world_z' not in result:
        result['follow_through_failures']=['no_valid_right_foot_contact']
        return result
    ball,goal=np.asarray(run['points'][index],float),np.asarray(run['goals'][index],float)
    direction=np.r_[goal-ball,0.]
    direction/=np.linalg.norm(direction)
    try:
        measured=_motion(run,index,ct,direction)
    except (KeyError,TypeError,ValueError,IndexError):
        result['follow_through_failures']=['missing_motion_coverage']
        return result
    result.update(measured)
    c=FOLLOW_THROUGH_CRITERIA
    late=any(ct+c['late_contact_start_s']<=row['time_s']<=ct+c['late_contact_end_s'] and
             row['contact_force_N']>c['late_contact_force_N'] for row in run['traces'][index])
    result['late_ball_contact']=late
    failed=[]
    if measured['forward_advance_m']<c['forward_advance_min_m']:failed.append('short_forward_swing')
    if measured['pelvis_relative_advance_m']<c['pelvis_relative_advance_min_m']:failed.append('short_relative_swing')
    if measured['forward_speed_at_120ms_mps']<c['forward_speed_min_mps']:failed.append('early_deceleration')
    if measured['reversal_after_s'] is not None and measured['reversal_after_s']<c['earliest_sustained_reversal_s']-1e-8:failed.append('early_reversal')
    if measured['peak_return_speed_mps']>c['peak_return_speed_max_mps']:failed.append('fast_retraction')
    if measured['peak_recovery_return_speed_mps']>c['recovery_return_speed_max_mps']:failed.append('fast_recovery_retraction')
    if late:failed.append('additional_ball_contact')
    if not result['posture_ok']:failed.append('contact_posture_failed')
    result.update(follow_through_failures=failed,follow_through_ok=not failed,
                  follow_through_and_full_chain=bool(not failed and result['full_chain_success']))
    return result


def check_follow_through_acceptance(config,reports,config_sha256):
    if any(run.get('diagnostic_only') for run in reports):
        raise ValueError('Diagnostic reports cannot certify acceptance')
    if config.get('follow_through_criteria')!=FOLLOW_THROUGH_CRITERIA:
        raise ValueError('Missing or changed follow-through criteria')
    required = config.get('follow_through_required_styles',
        [style for style in config['styles'] if style in ('instep', 'outside')])
    if (not isinstance(required, list) or not required or len(set(required)) != len(required)
            or any(style not in config['styles'] or style not in ('instep', 'outside') for style in required)):
        raise ValueError('Declare a nonempty subset of the measured instep/outside styles')
    if 'inside' in config['styles'] and config['styles']['inside'] != {'strike': {}}:
        raise ValueError('Only unchanged inside control can be exempt from the follow-through gate')
    for key in ('controller_sources_sha256', 'motion_file', 'motion_sha256'):
        if not config.get(key) or any(run.get(key) != config[key] for run in reports):
            raise ValueError(f'Missing or changed frozen {key}')
    result=check_posture_acceptance(config,reports,config_sha256)
    rows=[];counts={style:0 for style in required}
    for run in reports:
        for i,trial in enumerate(run['trials']):
            row=audit_follow_through(run,i)
            row.update(seed=run['seed'],index=i,name=trial.get('name',trial['style']))
            rows.append(row)
            row['follow_through_required'] = trial['style'] in required
            if row['follow_through_required']:
                counts[trial['style']]+=int(row['follow_through_and_full_chain'])
    result.update(follow_through_criteria=FOLLOW_THROUGH_CRITERIA,follow_through_trials=rows,
                  follow_through_required_styles=required,follow_through_successes=counts,
                  passed=bool(result['passed'] and all(n>=5 for n in counts.values())))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+');parser.add_argument('--config');parser.add_argument('--output',required=True)
    args=parser.parse_args();runs=[json.loads(Path(path).read_text()) for path in args.runs]
    if args.config:
        raw=Path(args.config).read_bytes()
        result=check_follow_through_acceptance(json.loads(raw),runs,hashlib.sha256(raw).hexdigest())
        result['sources']=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in args.runs]
        print(json.dumps({k:result[k] for k in ['passed','styles','follow_through_successes']},indent=2))
    else:
        result=[]
        for path,run in zip(args.runs,runs):
            for i,trial in enumerate(run['trials']):
                row=audit_follow_through(run,i);row.update(source=path,index=i,name=trial.get('name',row['style']))
                result.append(row)
                print(row['name'],'combined',row['follow_through_and_full_chain'],
                      'advance',round(row.get('forward_advance_m',0),3),
                      'relative',round(row.get('pelvis_relative_advance_m',0),3),
                      'v120',round(row.get('forward_speed_at_120ms_mps',0),3),
                      'return',round(row.get('peak_return_speed_mps',0),3),row['follow_through_failures'])
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    return 0 if not args.config or result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
