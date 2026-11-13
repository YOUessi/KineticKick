"""Whole-body swing and contralateral plant changes layered on the released v161 shot."""
import numpy as np

from v161_strike_search import LEG, make_controller as make_baseline


def layout(names):
    return [names.index(j) for j in ['right_'+n+'_joint' for n in LEG]+
            ['left_hip_pitch_joint','left_knee_joint','left_ankle_pitch_joint','left_hip_roll_joint',
             'left_hip_yaw_joint','waist_yaw_joint']]


def envelope(t):
    # First swing shape during backswing; strike form persists through post-impact follow-through.
    return (np.interp(t,[2.04,2.17,2.30,2.41,2.57],[0,1,1,.9,0]),
            np.interp(t,[2.22,2.34,2.41,2.50,2.65],[0,1,1,.62,0]),
            np.interp(t,[1.83,1.98,2.16,2.34,2.52,2.73],[0,.45,1,1,.65,0]))


def correction(t, params):
    early, strike, plant = envelope(t)
    return np.r_[early*params[:6]+strike*params[6:12], plant*params[12:]]


def make_controller(ref, schema, config):
    baseline=make_baseline(ref,schema,config)
    indices=layout(schema['joint_names'])
    params=np.asarray(config['vertical_parameters'],dtype=float)
    if params.shape != (18,) or not np.all(np.isfinite(params)):
        raise ValueError('Expected eighteen finite swing/support coefficients')
    def control(ctx,target):
        target=baseline(ctx,target)
        target[indices]+=correction(ctx['t'],params)
        return target
    return control
