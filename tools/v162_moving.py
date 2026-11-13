"""Causal, bounded ankle material-point correction for the frozen 29-axis strike."""
import hashlib
import json
import atexit
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from cartesian_strike import LegKinematics
from v161_strike_search import make_controller as base_controller
import v160_tracking_cpu as physical

REPO = Path(__file__).resolve().parents[1]


def limited(x, maximum):
    return x * min(1., maximum / max(1e-12, np.linalg.norm(x)))


def forecast(ctx, cfg):
    horizon = np.clip(cfg['nominal_contact_s'] - ctx['t'], 0., cfg['prediction_horizon_s'])
    return np.asarray(ctx['ball_pos'])[:2] + horizon * np.asarray(ctx['ball_velocity'])[:2]


def stance_planner(ref, schema, cfg):
    protocol = json.loads(physical.PROTOCOL.read_text())
    assets = Path(cfg.get('asset_root', physical.DEFAULT_ASSETS))
    if physical.digest(assets / '.pretrained_checkpoints/paid_official/policy_30000.onnx') != protocol['asset_hashes']['.pretrained_checkpoints/paid_official/policy_30000.onnx']:
        raise ValueError('Unqualified gait planner')
    context = physical.cpu_policy(assets, protocol)
    meta, step = context.__enter__()
    atexit.register(context.__exit__, None, None, None)
    anchor = schema['body_names'].index(meta['anchor_body_name'])
    scale = np.fromstring(meta['action_scale'], sep=',')
    previous = np.zeros(29)
    decay = cfg['rolling_decay_per_s']

    def step_toward_intercept(ctx):
        nonlocal previous
        # Same actor and temporal reference, but the task point is explicitly a
        # planned rendezvous, distinct from the measured soccer ball fed to the parent.
        point = np.array(ctx['ball_pos'], dtype=float)
        if ctx['first_contact_s'] is None:
            horizon = max(0., cfg['nominal_contact_s']-ctx['t'])
            flight = -np.expm1(-decay*horizon)/decay
            point[:2] += flight*np.asarray(ctx['ball_velocity'])[:2]
        root = SimpleNamespace(p=ctx['root_pos'],q=ctx['root_quat'])
        obs = physical.observation(ctx['q'],ctx['dq'],root,ctx['root_velocity'][3:],
                                   point,ctx['goal'],previous,ref,ctx['frame'],anchor,ctx['default_q'])
        previous = np.clip(step(ctx['frame'],obs), -100., 100.)
        return ctx['default_q']+scale*previous

    return step_toward_intercept


def support_clearance(schema, cfg):
    """Use 20ms left toe history to forecast closest passage of the real rolling ball."""
    leg = OrthogonalSupport(schema, cfg)
    ankle = schema['body_names'].index('left_ankle_roll_link')
    prior_toe = None
    shift = 0.
    fallback = None
    def clear(ctx, target):
        nonlocal prior_toe, shift, fallback
        t, dt = ctx['t'], ctx['dt']
        toe = ctx['body_pos'][ankle] + physical.rotation(ctx['body_quat'][ankle]).apply(cfg['toe_local_m'])
        old = prior_toe
        prior_toe = toe
        start, end, release = cfg['window_s']
        if old is None or not start <= t < release:
            return target
        ahead = ctx['goal'][:2] - ctx['ball_pos'][:2]
        ahead /= np.linalg.norm(ahead)
        side = np.array([-ahead[1], ahead[0]])
        if t<=end and ctx['first_contact_s'] is None and ctx['foot_load'][0]<cfg['max_airborne_load_N']:
            rel = toe[:2] - ctx['ball_pos'][:2]
            rate = (toe[:2]-old[:2])/dt - ctx['ball_velocity'][:2]
            horizon = np.clip(-np.dot(rel,rate)/(np.dot(rate,rate)+1e-8), 0., cfg['lookahead_s'])
            near = rel + rate*horizon
            gap = np.linalg.norm(near)
            if horizon > 0. and gap < cfg['guard_m']:
                # Keep the natural supporting side of the ball's line; do not cross it.
                if fallback is None:
                    fallback = side * (1. if np.dot(near,side)>=0. else -1.)
                required = np.clip(cfg['clearance_gain']*(cfg['guard_m']-gap),0.,cfg['maximum_side_step_m'])
                shift = max(shift,required)
        if fallback is None:
            return target
        phase = np.clip((release-t)/(release-end), 0.,1.)
        return leg.translate(ctx,target,np.r_[fallback*shift*phase,0.])

    return clear


class OrthogonalSupport:
    """Compute the material-point Jacobian from actual left joint pivots and URDF axes."""
    def __init__(self, schema, cfg):
        import xml.etree.ElementTree as ET
        xml = ET.parse(cfg['urdf']).getroot()
        names = ['left_'+j+'_joint' for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        joints = [xml.find(f"joint[@name='{n}']") for n in names]
        self.ids = np.array([schema['joint_names'].index(n) for n in names])
        self.links = np.array([schema['body_names'].index(j.find('child').get('link')) for j in joints])
        self.axes = np.array([np.fromstring(j.find('axis').get('xyz'),sep=' ') for j in joints])
        self.cfg = cfg

    def translate(self,ctx,target,delta):
        rotations = physical.rotation(ctx['body_quat'][self.links])
        axes = rotations.apply(self.axes)
        pivots = ctx['body_pos'][self.links]
        toe = pivots[-1] + rotations[-1].apply(self.cfg['toe_local_m'])
        jac = np.vstack((np.cross(axes, toe-pivots).T, self.cfg['orientation_weight_m']*axes.T))
        desired = np.r_[delta,0.,0.,0.]
        joints = np.linalg.solve(jac.T@jac + np.eye(6)*.001, jac.T@desired)
        target[self.ids] += limited(joints, self.cfg['max_joint_rad'])
        return target


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
    planner = stance_planner(ref,schema,cfg) if cfg.get('planned_stance') else None
    clearance = support_clearance(schema, dict(urdf=cfg['urdf'],**cfg['left_clearance'])) if 'left_clearance' in cfg else None

    def control(ctx, target):
        nonlocal prior, close
        if planner is not None:
            target = planner(ctx)
        if clearance is not None:
            target = clearance(ctx,target)
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
