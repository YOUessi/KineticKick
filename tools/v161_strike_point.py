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
    def control(ctx, target):
        nonlocal ball_anchor, previous
        t = ctx['t']
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
        desired = ball_anchor - .11*outward + forward*distance
        desired[2] += config['backswing_lift_m']*max(-distance/.40,0.) + config['follow_lift_m']*max(distance/.40,0.)
        root = np.eye(4)
        root[:3,3] = ctx['root_pos'] + ctx['root_velocity'][:3]*lead
        rot = Rotation.from_quat(ctx['root_quat'][[1,2,3,0]])
        root[:3,:3] = (Rotation.from_rotvec(ctx['root_velocity'][3:]*lead)*rot).as_matrix()
        seed = ctx['q'][idx] if previous is None else previous
        q = solve_point(leg, root, seed, point, normal, desired, outward)
        previous = q
        weight = smooth((t-begin)/(full-begin))*smooth((end-t)/(end-hold))
        result = target.copy()
        result[idx] += weight*(q-result[idx])
        return result
    return control
