"""Post-registered ball-event readout on the frozen SAPIEN trajectory only."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


FEET = {'left_ankle_roll_link', 'right_ankle_roll_link'}


def episode_readout(contact, times, horizontal_velocity):
    """Use first unqualified sample after each contiguous loaded interval, including the last."""
    edges = np.diff(np.r_[False, contact, False].astype(int))
    runs = zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)
    result = []
    for first, last in runs:
        after = last + 1
        result.append({
            'first_loaded_bin_s': float(times[first]), 'last_loaded_bin_s': float(times[last]),
            'loaded_bins': int(last-first+1),
            'first_unqualified_bin_s': float(times[after]) if after < len(times) else None,
            'velocity_xy_mps_after_interval': float(np.linalg.norm(horizontal_velocity[after])) if after < len(times) else None,
        })
    return result


def crossing(positions, times, goal, forward, distance, radius, size):
    projection = (positions[:, :2]-goal[:2]) @ forward
    transitions = np.flatnonzero((projection[:-1] < distance) & (projection[1:] >= distance))
    if not len(transitions):
        return None
    i = int(transitions[0]); fraction = (distance-projection[i])/(projection[i+1]-projection[i])
    point = positions[i] + fraction*(positions[i+1]-positions[i])
    lateral = (point[:2]-goal[:2]) @ [-forward[1],forward[0]]
    fits = abs(lateral) <= size[0]/2-radius and radius-1e-4 <= point[2] <= size[1]-radius
    return {'interpolated_time_s':float(times[i]+fraction*(times[i+1]-times[i])),
            'source_bracket_s':[float(times[i]),float(times[i+1])], 'ball_centre_xyz_m':point.tolist(),
            'lateral_from_goal_centre_m':float(lateral),'whole_sphere_within_aperture':bool(fits)}


def summarize(folder, config):
    with np.load(folder/'rollout.npz', allow_pickle=False) as tr:
        times=tr['time_s']; speed_xy=tr['ball_velocity_w_mps'][:,:2]; body_contact=tr['robot_ball_contact']
        positions=np.vstack((config['ball_initial_xyz_m'],tr['ball_position_w_m']))
    contact = np.zeros(len(times),dtype=bool); shapes=set(); step=0
    with gzip.open(folder/'native_contacts.jsonl.gz','rt') as stream:
        for step, line in enumerate(stream,start=1):
            index,pairs=json.loads(line)
            if index!=step:raise ValueError('Invalid contact clock')
            for pair in pairs:
                if 'SoccerBall' not in pair['actors'] or not FEET.intersection(pair['actors']):continue
                impulse=np.asarray(pair['points'],dtype=float)[:,6:9].sum(axis=0)
                if np.linalg.norm(impulse)/config['physics_dt_s']>2.:
                    contact[step-1]=True
                    shapes.update(shape for name,shape in zip(pair['actors'],pair['shapes']) if name in FEET)
    if step!=len(times):raise ValueError('Truncated native contact log')
    common=(positions,np.r_[0.,times],np.array(config['goal_xyz_m']),np.array(config['goal_forward_xy']))
    radius=config['ball_radius_m'];size=config['goal_width_height_m']
    return {'loaded_rule':'native foot/ball pair norm(sum of reported 3D contact-point impulses)/dt >2 N; subthreshold does not imply geometric separation',
            'foot_and_any_robot_loaded_bins_equal':bool(np.array_equal(contact,body_contact)),
            'loaded_foot_shapes':sorted(shapes),'requested_instep_fraction':None,
            'surface_attribution':'No certified instantaneous contact-point instep classifier for this CPU backend',
            'foot_ball_intervals':episode_readout(contact,times,speed_xy),'centre_plane':crossing(*common,0.,radius,size),
            'whole_sphere_plane':crossing(*common,radius,radius,size),'new_physical_runs':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/v160_tracking_protocol.json').read_text())
    result=summarize(a.run,config)
    with a.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps(result,indent=2))
