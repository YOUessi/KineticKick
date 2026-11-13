"""Closed-loop right-instep material-point path and surface-normal IK with a free base."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cartesian_strike import LegKinematics


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)


def solve_point(leg, root, seed, point, normal, desired_point, desired_normal):
    """Five task constraints leave foot twist free; solve in original position bounds."""
    q = np.clip(seed, leg.limits[:, 0], leg.limits[:, 1]).copy()
    for _ in range(16):
        pose, jac = leg.forward(q, root)
        arm = pose[:3, :3] @ point
        n = pose[:3, :3] @ normal
        dp = desired_point - (pose[:3, 3]+arm)
        dn = desired_normal - n
        if np.linalg.norm(dp) < .0006 and np.linalg.norm(dn) < .015:
            break
        J = np.vstack((jac[:3]+np.cross(jac[3:].T, arm).T, .13*np.cross(jac[3:].T, n).T))
        H = J.T @ J + np.eye(6)*.00018
        step = np.linalg.solve(H, J.T @ np.r_[dp,.13*dn] + .00002*(seed-q))
        q = np.clip(q + np.clip(step, -.21, .21), leg.limits[:, 0], leg.limits[:, 1])
    return q


def make_controller(ref, schema, config):
    leg = LegKinematics(Path(config['urdf']))
    idx = [schema['joint_names'].index(n) for n in leg.names]
    point = np.asarray(config['material_point_ankle_m'])
    normal = np.asarray(config['desired_normal_ankle'])
    normal = normal / np.linalg.norm(normal)
    ball_anchor = None
    previous = None
    foot_history = []
    foot = schema['body_names'].index('right_ankle_roll_link')
    arm = schema['joint_names'].index('left_shoulder_roll_joint')
    def control(ctx, target):
        nonlocal ball_anchor, previous
        t = ctx['t']
        if 'return_velocity_gain' in config:
            foot_history.append(ctx['body_pos'][foot].copy())
            if len(foot_history)>2:
                foot_history.pop(0)
        if 'arm_clearance_rad' in config and t > 2.44:
            start = smooth((t-2.44)/.20)
            release = smooth((4.10-t)/.28)
            target = target.copy()
            target[arm] += start*release*(max(target[arm],config['arm_clearance_rad'])-target[arm])
        begin, full, hold, end = config['window_s']
        if t <= begin or t >= end:
            return target
        if ball_anchor is None or ctx['first_contact_s'] is None:
            ball_anchor = ctx['ball_pos'].copy()
        forward = ctx['goal'][:2] - ball_anchor[:2]
        forward = forward / np.linalg.norm(forward)
        angle = config.get('yaw_rad', 0.)
        forward = np.r_[Rotation.from_rotvec([0,0,angle]).apply([*forward,0])[:2], 0.]
        elev = config['elevation_rad']
        outward = np.cos(elev)*forward + [0,0,np.sin(elev)]
        lead = config['lead_s']
        phase = t + lead - config['target_contact_s']
        distance = .40*np.tanh(config['sweep_mps']*phase/.40)
        if 'return_start_s' in config:
            elapsed = max(0., phase-config['return_start_s'])
            # Release smoothly toward a landing ahead of the body, with a finite retreat speed.
            ramp = .045
            distance -= config['return_mps']*(elapsed-ramp*(1.-np.exp(-elapsed/ramp)))
        desired = ball_anchor - .11*outward + forward*distance
        lateral = np.array([-forward[1],forward[0],0.])
        desired += lateral*config.get('point_lateral_shift_m',0.)
        # Add swing-leg clearance while the distal shank passes the ball.
        desired[2] += config.get('clearance_lift_m',0.) * np.exp(-.5*(phase/.105)**2)
        desired[2] += config['backswing_lift_m']*max(-distance/.40,0.) + config['follow_lift_m']*max(distance/.40,0.)
        root = np.eye(4)
        root[:3,3] = ctx['root_pos'] + ctx['root_velocity'][:3]*lead
        rot = Rotation.from_quat(ctx['root_quat'][[1,2,3,0]])
        root[:3,:3] = (Rotation.from_rotvec(ctx['root_velocity'][3:]*lead)*rot).as_matrix()
        if 'landing_duration_s' in config:
            grounded = smooth((phase-.19)/config['landing_duration_s'])
            desired[2] += grounded*(.02-desired[2])
            outward = outward*(1.-grounded) + np.array([0.,0.,1.])*grounded
            outward /= np.linalg.norm(outward)
        seed = ctx['q'][idx] if previous is None else previous
        q = solve_point(leg, root, seed, point, normal, desired, outward)
        previous = q.copy()
        if 'hip_drive_rad' in config:
            approach = smooth((phase+.115)/.075) * smooth((.125-phase)/.08)
            if ctx['foot_load'][0] >= 80.:
                q[0] = np.clip(q[0]-approach*config['hip_drive_rad'],*leg.limits[0])
        if config.get('hip_safety') and phase > .090:
            # During follow-through, abduct the lifted thigh as its pitch closes on the pelvis.
            strength = smooth((phase-.090)/.080)
            hip_guard = -.95 if q[1] > .18 else -1.25
            q[0] += strength * max(hip_guard-q[0], 0.)
            q[1] -= strength * max(q[1]-.04, 0.)
        weight = smooth((t-begin)/(full-begin))*smooth((end-t)/(end-hold))
        result = target.copy()
        result[idx] += weight*(q-result[idx])
        if ('return_velocity_gain' in config and ctx['first_contact_s'] is not None and
            t-ctx['first_contact_s'] > .18 and len(foot_history) == 2 and ctx['foot_load'][1] < 80):
            measured_speed = (foot_history[1]-foot_history[0])@forward / ctx['dt']
            force = min(max(-config['allow_return_mps']-measured_speed,0.)*config['return_velocity_gain'],95.)
            measured_root = root.copy()
            measured_root[:3,3] = ctx['root_pos']
            measured_root[:3,:3] = rot.as_matrix()
            _, jac = leg.forward(ctx['q'][idx], measured_root)
            # J^T along the goal axis, converted to a PD setpoint through the released gains.
            ks = np.array([.01017752,.025101925,.01017752,.025101925,.00721945,.00721945])*(20*np.pi)**2
            delta = np.clip(jac[:3].T@forward*force/ks,-.36,.36)
            result[idx] = np.clip(result[idx]+delta,leg.limits[:,0],leg.limits[:,1])
            if 'landing_knee_flex_rad' in config:
                # Flex for loading and absorb the swing-foot landing during the return.
                knee = idx[3]
                blend = smooth((t-ctx['first_contact_s']-.27)/.08)
                result[knee] += blend*max(config['landing_knee_flex_rad']-result[knee],0.)
        return result
    return control
