"""Causal, bounded ankle material-point correction for the frozen 29-axis strike."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cartesian_strike import LegKinematics
from v161_strike_search import make_controller as base_controller

REPO = Path(__file__).resolve().parents[1]


def limited(x, maximum):
    return x * min(1., maximum / max(1e-12, np.linalg.norm(x)))


def forecast(ctx, cfg):
    horizon = np.clip(cfg['nominal_contact_s'] - ctx['t'], 0., cfg['prediction_horizon_s'])
    return np.asarray(ctx['ball_pos'])[:2] + horizon * np.asarray(ctx['ball_velocity'])[:2]


def make_controller(ref, schema, cfg):
    parent = REPO / cfg['parent']
    if hashlib.sha256(parent.read_bytes()).hexdigest() != cfg['parent_sha256']:
        raise ValueError('The frozen v161 parent controller changed')
    base = base_controller(ref, schema, json.loads(parent.read_text()))
    leg = LegKinematics(cfg['urdf'])
    indices = [schema['joint_names'].index(n) for n in leg.names]
    point = np.asarray(cfg['material_point_m'])
    nominal = np.asarray(cfg['nominal_ball_xy_m'])
    prior = np.zeros((2,))
    close = None

    def control(ctx, target):
        nonlocal prior, close
        target = base(ctx, target)
        t = ctx['t']
        weights = cfg['time_scale_knots']
        mix = np.interp(t, weights[0], weights[1], left=0., right=0.)
        if mix <= 0.:
            return target
        if close is None:
            if ctx['first_contact_s'] is None:
                estimate = limited(forecast(ctx, cfg) - nominal, cfg['max_displacement_m'])
                prior += limited(estimate-prior, ctx['dt']*cfg['rate_mps'])
            else:
                close = t
        root = np.eye(4)
        root[:3,:3] = Rotation.from_quat(np.asarray(ctx['root_quat'])[[1,2,3,0]]).as_matrix()
        root[:3,3] = ctx['root_pos']
        pose, jac = leg.forward(np.asarray(ctx['q'])[indices], root)
        arm = pose[:3,:3] @ point
        transl = jac[:3] + np.cross(jac[3:].T, arm).T
        task = np.vstack((transl, cfg['orientation_length_m']*jac[3:]))
        desired = np.r_[prior * np.asarray(cfg['xy_gain']) * mix, 0., 0., 0., 0.]
        dq = np.linalg.solve(task.T@task + np.eye(6)*cfg['damping_m2'], task.T@desired)
        target[indices] += limited(dq, cfg['max_joint_norm_rad'])
        return target

    return control
