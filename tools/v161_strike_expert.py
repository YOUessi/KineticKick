"""Transfer the historical v32 whole-body actuator sequence over the support transition."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from v161_strike_point import smooth


def make_controller(ref, schema, config):
    evidence = Path(config['expert_report']).read_bytes()
    if hashlib.sha256(evidence).hexdigest() != '17d294d6563b5695edb2444e768650e26b145a67d08626693e3012fa2cd981b0':
        raise ValueError('Historical successful controller record differs')
    record = json.loads(evidence)
    order = [record['joint_names'].index(name) for name in schema['joint_names']]
    samples = record['traces'][0]
    seconds = np.array([row['time_s'] for row in samples])
    pos = np.array([row['joint_target'] for row in samples])[:,order]
    vel = np.array([row['joint_velocity_target'] for row in samples])[:,order]
    # A physical velocity-target pulse, when present, can be expressed as a drive-position
    # offset D/K * v_target. Original source D/K = 4/(2*pi*10) except rounded metadata.
    pos += 4/(2*np.pi*10)*vel
    times = np.array([a['time_s'] for a in record['kinematics'][0]])
    measured_q = np.array([row['joint_pos'] for row in samples])[:,order]
    left_roll = schema['joint_names'].index('left_hip_roll_joint')
    right_roll = schema['joint_names'].index('right_hip_roll_joint')
    reference_center = np.array(record['points'][0])
    source_root = np.array([row['root_xyz'] for row in samples])
    source_rootv = np.array([row['root_velocity_w'][:3] for row in samples])
    angles = Rotation.from_quat(np.array([a['root_pose'][3:] for a in record['kinematics'][0]])[:,[1,2,3,0]]).as_euler('xyz')[:,2]
    yaw_shift = -.15676
    yaw_rot = Rotation.from_rotvec([0.,0.,yaw_shift])
    ball = None
    def control(ctx, target):
        nonlocal ball
        if ball is None:
            ball = np.array(ctx['ball_pos'], copy=True)
        t = ctx['t']
        begin, full, hold, end = config['window_s']
        if t <= begin or t >= end:
            return target
        thist = 1.57+(t-config['contact_s'])*config['phase_rate']
        i = int(np.clip(np.searchsorted(seconds, thist),0,len(seconds)-1))
        if config['source'] == 'target':
            q = pos[i].copy()
        else:
            q = measured_q[min(i+config.get('lead_ticks',2),len(seconds)-1)].copy()
        # Symmetric pre-plant hip-roll feedback shifts the pelvis toward the historical
        # ball-relative running path. The source world translation is never a state write.
        desired_p = ball[:2] + yaw_rot.apply(np.r_[source_root[i,:2]-reference_center[:2],0])[:2]
        desired_v = yaw_rot.apply(source_rootv[i])[:2]
        if 'placement' in config:
            lateral = np.clip((desired_p[0]-ctx['root_pos'][0])*config['placement']+
                              (desired_v[0]-ctx['root_velocity'][0])*.06, -.15,.15)
            q[[left_roll,right_roll]] += lateral
        gain = smooth((t-begin)/(full-begin))*smooth((end-t)/(end-hold))*config['weight']
        return target + gain*(q-target)
    return control
