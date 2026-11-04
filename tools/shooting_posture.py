"""Audit actual kicking posture independently of requested style labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from free_shooting_metrics import ShotTracker


POSTURE_CRITERIA = dict(instep_top_forward_min=.75, instep_toe_world_z_max=-.60,
    outside_abs_hip_yaw_max_rad=1.90, outside_heading_error_max_deg=45.,
    outside_axis_forward_min=.75, required_complete_trials_per_style=5,
    trials_per_style=6, allowed_falls=0)


def rotate_vector(quaternion_wxyz, vector):
    q, v = np.asarray(quaternion_wxyz, float), np.asarray(vector, float)
    if q.shape != (4,) or v.shape != (3,) or not np.all(np.isfinite([*q,*v])) or np.linalg.norm(q)<1e-8:
        raise ValueError('Expected a finite nonzero wxyz quaternion and a 3-vector')
    q = q/np.linalg.norm(q)
    return v + 2*np.cross(q[1:], np.cross(q[1:],v)+q[0]*v)


def audit_posture(run, index):
    trial = run['trials'][index]
    style = trial['style'] if 'style' in trial else run['requested_style']
    ball, goal = np.asarray(run['points'][index], float), np.asarray(run['goals'][index], float)
    aim = goal.copy()
    if 'target' in trial: aim[0] = trial['target']['target_x']
    direction = np.r_[aim-ball,0.]
    if np.linalg.norm(direction)<1e-8: raise ValueError('Shot direction must be nonzero')
    direction /= np.linalg.norm(direction)
    tracker = ShotTracker(initial_root=run['initial_robot_state']['root_position'][index],
                          initial_ball=[*ball,.113], goal=goal,
                          forward=trial.get('goal_forward',goal-ball),style=style)
    traces = run['traces'][index]
    for row in traces: tracker.update(row)
    base = tracker.summary()
    first = next((row for row in traces if row['contact_force_N']>2 and row.get('contacts')), None)
    result = dict(style=style, full_chain_success=base['success'], surface_match=base['surface_match'],
                  goal=base['goal'], recovered=base['recovered'], failed=base['failed'], posture_ok=False,
                  posture_and_full_chain=False, posture_failure=None,
                  contact_time_s=base['contact_time_s'])
    if first is None:
        result['posture_failure'] = 'no_contact'
        return result
    contact = max(first['contacts'],key=lambda event:event['force_N'])
    if contact['foot']!='right':
        result['posture_failure'] = 'wrong_kicking_foot'
        return result
    kin = run['kinematics'][index]
    if not kin: raise ValueError('Missing measured root/joint kinematics')
    near = min(kin,key=lambda sample:abs(sample['time_s']-first['time_s']))
    time_error = abs(near['time_s']-first['time_s'])
    if time_error > .021: raise ValueError('Root-pose sample is too far from first contact')
    q = contact['foot_quat_w']
    top, toe, outside = (rotate_vector(q,axis) for axis in ([0,0,1],[1,0,0],[0,-1,0]))
    heading = rotate_vector(near['root_pose'][3:7],[1,0,0]); heading[2]=0.
    heading /= np.linalg.norm(heading)
    error = math.degrees(math.acos(float(np.clip(heading@direction,-1,1))))
    yaw_index = run['joint_names'].index('right_hip_yaw_joint')
    yaw = float(contact['joint_pos'][yaw_index])
    peak_yaw = max(abs(float(k['joint_pos'][yaw_index])) for k in kin)
    point = contact['point_foot']
    weighted_top, total_impulse = 0., 0.
    for row in traces:
        for event in row.get('contacts',[]):
            impulse = event['force_N']*row['physics_dt_s']
            if event['foot']=='right' and impulse>0:
                weighted_top += float(rotate_vector(event['foot_quat_w'],[0,0,1])@direction)*impulse
                total_impulse += impulse
    top_forward, outside_forward = float(top@direction), float(outside@direction)
    average_top = weighted_top/total_impulse if total_impulse else None
    central = bool(abs(point[1])<=.018 and -.01<=point[0]<=.11)
    if style=='instep':
        pose = top_forward>=.75 and toe[2]<=-.60 and average_top is not None and average_top>=.75
    elif style=='outside':
        pose = abs(yaw)<=1.90 and peak_yaw<=1.90 and error<=45 and outside_forward>=.75
    else:
        pose = True
    result.update(top_axis_world=top.tolist(), toe_axis_world=toe.tolist(), outside_axis_world=outside.tolist(),
                  top_forward=top_forward, impulse_weighted_top_forward=average_top,
                  toe_world_z=float(toe[2]), outside_forward=outside_forward,
                  hip_yaw_rad=yaw, peak_abs_hip_yaw_rad=peak_yaw,
                  pelvis_heading_error_deg=error, root_sample_offset_s=time_error,
                  contact_point_local=point, central_contact=central,
                  posture_ok=bool(pose and base['surface_match']),
                  posture_failure=None if pose and base['surface_match'] else
                      ('wrong_contact_surface' if not base['surface_match'] else 'pose_limits'),
                  posture_and_full_chain=bool(pose and base['success']))
    return result


def check_posture_acceptance(config, reports, config_sha256):
    """Replay the frozen three-position/two-seed matrix, including every failure."""
    from check_free_shooting_acceptance import validate_matrix
    validate_matrix(config, reports)
    if len(config['acceptance_seeds']) != 2 or len(config['acceptance_conditions']) != 3:
        raise ValueError('Posture acceptance requires three positions and two seeds')
    if config.get('posture_criteria', POSTURE_CRITERIA) != POSTURE_CRITERIA:
        raise ValueError('Configuration differs from the implemented posture criteria')
    counts = {style: dict(trials=0, full_chain=0, posture_and_full_chain=0, falls=0)
              for style in config['styles']}
    rows = []
    for run in reports:
        if run['config_sha256'] != config_sha256:
            raise ValueError('Run does not match the supplied frozen configuration')
        if run.get('goal_width_m') != 1.6 or run.get('goal_height_m') != 1.1:
            raise ValueError('Unexpected physical goal aperture')
        for i, trial in enumerate(run['trials']):
            control = config['styles'][trial['style']]
            for key in ('strike', 'reference', 'cartesian', 'policy_goal', 'goal_forward', 'target'):
                if trial.get(key) != control.get(key):
                    raise ValueError(f'A trial changes the frozen {key}')
            if (run['points'][i] != trial['ball'] or run['goals'][i] != trial['goal'] or
                    run['policy_goals'][i] != control.get('policy_goal', trial['goal'])):
                raise ValueError('Manifest differs from the actual ball/goal inputs')
            trace = run['traces'][i]
            if not trace:
                raise ValueError('Missing physical trace')
            physical = trace[:-1] if trace[-1]['terminated'] else trace
            times = np.array([row['time_s'] for row in physical])
            if (not len(times) or not np.allclose(times, np.arange(1, len(times)+1)*.005, atol=1e-7, rtol=0)
                    or any(abs(row['physics_dt_s']-.005)>1e-9 for row in physical)):
                raise ValueError('Physical trace must preserve every 5 ms sample')
            if trace[-1]['terminated'] and abs(trace[-1]['time_s']-times[-1]-1e-6)>1e-8:
                raise ValueError('Invalid terminal marker')
            if (not trace[-1]['terminated'] and
                    abs(times[-1]-config['trial_steps']*config['policy_dt_s'])>.001):
                raise ValueError('Incomplete nonterminal trial')
            result = audit_posture(run, i)
            if i in run.get('automatic_resets_suppressed', []) and not result['failed']:
                raise ValueError('Suppressed reset lacks a recorded terminal failure')
            count = counts[trial['style']]
            count['trials'] += 1
            count['full_chain'] += int(result['full_chain_success'])
            count['posture_and_full_chain'] += int(result['posture_and_full_chain'])
            count['falls'] += int(result['failed'])
            result.update(seed=run['seed'], index=i, name=trial.get('name', trial['style']))
            rows.append(result)
    passed = all(c['trials']==6 and c['posture_and_full_chain']>=5 and c['falls']==0 for c in counts.values())
    return dict(passed=passed, config_sha256=config_sha256, criteria=POSTURE_CRITERIA,
                styles=counts, trials=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+')
    parser.add_argument('--output',required=True)
    parser.add_argument('--config', help='Also certify the complete frozen posture matrix')
    args=parser.parse_args()
    if args.config:
        raw = Path(args.config).read_bytes()
        reports = [json.loads(Path(path).read_text()) for path in args.runs]
        result = check_posture_acceptance(json.loads(raw), reports, hashlib.sha256(raw).hexdigest())
        result['sources'] = [dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                             for path in args.runs]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('passed','styles','config_sha256')}, indent=2))
        return 0 if result['passed'] else 1
    rows=[]
    for path in args.runs:
        run=json.loads(Path(path).read_text())
        sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for index in range(len(run['trials'])):
            result=audit_posture(run,index)
            result.update(source=path,source_sha256=sha,index=index,name=run['trials'][index].get('name',result['style']))
            rows.append(result)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:
        print(r['name'], 'pose',r['posture_ok'],'chain',r['full_chain_success'],
              'top_forward',round(r.get('top_forward',0),3),'hip_yaw_deg',round(math.degrees(r.get('hip_yaw_rad',0)),1),
              'heading_error',round(r.get('pelvis_heading_error_deg',0),1))
    return 0


if __name__=='__main__': raise SystemExit(main())
