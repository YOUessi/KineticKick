"""Measure historical vertical-instep pose at actual loaded CPU contact in a complete shot."""
import argparse
import json
from pathlib import Path

import numpy as np

from v160_tracking_ball_impulse import collect_contacts
from v161_evaluate import report as shot_report, quat_rotation, digest
from v161_surface_audit import summarize as audit_surface
from vertical_instep import FOOT_CORNERS, VERTICAL_CRITERIA


def measured_pose(times, positions, rotations, loads, points, direction):
    if not loads.any() or not points:
        return {'pass':False,'reason':'No loaded right-foot contact'}
    direction=np.asarray(direction,dtype=float).copy();direction[2]=0.
    direction/=np.linalg.norm(direction)
    first=int(np.flatnonzero(loads)[0])
    i=np.asarray([p[0]-1 for p in points]);w=np.asarray([p[3] for p in points],dtype=float)
    if i.min()<1 or first<8 or first+4>=len(times):
        return {'pass':False,'reason':'Contact window lacks measured poses'}
    matrices=rotations.as_matrix()
    axis_down=-matrices[:,2,0]
    dorsal=matrices[:,:,2]@direction
    def view(offset):
        a=float(axis_down[first+offset]);b=float(np.average(axis_down[i+offset],weights=w))
        first_f=float(dorsal[first+offset]);weighted_f=float(np.average(dorsal[i+offset],weights=w))
        return {'first_toe_down_deg':float(np.rad2deg(np.arcsin(np.clip(a,-1.,1.)))),
                'all_impulse_weighted_toe_down_deg':float(np.rad2deg(np.arcsin(np.clip(b,-1.,1.)))),
                'first_dorsal_forward':first_f,'all_impulse_weighted_dorsal_forward':weighted_f}
    lo,hi=first-8,first+4
    if not np.isclose(times[hi]-times[lo],.060):
        raise ValueError('Missing native 5ms near-impact pose clock')
    clearance=positions[:,2]+(matrices[:,2]@np.asarray(FOOT_CORNERS).T).min(axis=1)
    primary=view(0);pre=view(-1);threshold=90.-VERTICAL_CRITERIA['max_angle_from_down_deg']
    gate={'first_toe_down':primary['first_toe_down_deg']>=threshold,
          'weighted_toe_down':primary['all_impulse_weighted_toe_down_deg']>=threshold,
          'first_dorsal_forward':primary['first_dorsal_forward']>=VERTICAL_CRITERIA['dorsal_forward_min'],
          'weighted_dorsal_forward':primary['all_impulse_weighted_dorsal_forward']>=VERTICAL_CRITERIA['dorsal_forward_min'],
          'padded_clearance':float(clearance[lo:hi+1].min())>=VERTICAL_CRITERIA['near_contact_clearance_min_m']}
    return {'pass':all(gate.values()),'checks':gate,'primary_post_step':primary,'pre_step_sensitivity':pre,
            'first_loaded_time_s':float(times[first]),'positive_point_count':int(len(points)),
            'window_s':times[[lo,hi]].tolist(),'sample_count':hi-lo+1,'clearance_at_contact_m':float(clearance[first]),
            'minimum_padded_clearance_m':float(clearance[lo:hi+1].min()),
            'weight':'Arithmetic mean of toe-down direction cosine/dorsal forward weighted by each reported positive native normal scalar impulse; angle is asin of weighted direction cosine.'}


def audit(folder):
    r=shot_report(folder);dt=.005
    surface=audit_surface(folder)['instep_normal_impulse_fractions']
    both=all(surface[k] is not None and surface[k]>=.95 for k in ('pre_foot','post_foot'))
    with np.load(folder/'rollout.npz') as tr:
        times=tr['time_s'];loaded,_,_,points=collect_contacts(folder/'native_contacts.jsonl.gz',times,dt)
        foot=json.loads((folder/'tracking_result.json').read_text())['source_body_names'].index('right_ankle_roll_link')
        pose=measured_pose(times,tr['body_position_w_m'][:,foot],quat_rotation(tr['body_quat_wxyz'][:,foot]),
                           loaded,points,np.asarray(r['goal_xyz_m'])-r['initial_ball_xyz_m'])
    return dict(run=folder.name,input_sha256=r['input_sha256'],v161_checks=r['checks'],v161_pass=r['cpu_proxy_task_pass'],
                v161_failures=r['failed_checks'],speed_xy_mps=r['last_exit_speed_xy_mps'],instep_proxy=r['instep_proxy'],
                follow=r['follow_through'],terminal_standing_s=r['terminal_standing_seconds'],vertical_instep=pose,
                all_source_patch_fractions=surface, both_corrected_witness_pose_checks=both,
                pass_combined=bool(r['cpu_proxy_task_pass'] and pose['pass'] and both), original_gpu_or_hardware_verified=None,
                audit_code_sha256=digest(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();row=audit(a.run)
    with a.output.open('x') as target:
        json.dump(row,target,indent=2,allow_nan=False);target.write('\n')
    print(row['run'],row['speed_xy_mps'],row['v161_failures'],row['vertical_instep'].get('primary_post_step'),row['pass_combined'])
