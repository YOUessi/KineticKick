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
    joint = np.r_[early*params[:6]+strike*params[6:12], plant*params[12:18]]
    if params.size >= 24:
        # A distinct pre-impact wave and its return can alter foot speed while preserving impact pose.
        wave=np.interp(t,[2.23,2.30,2.365,2.43,2.49],[0,1,-.55,-.24,0.])
        joint[:6] += wave*params[18:24]
    if params.size == 30:
        # The stance-toe pivot and arm/trunk counter-rotation start before the final plant.
        support = np.interp(t,[1.83,1.99,2.19,2.36,2.50,2.72],[0,.65,1,1,.52,0])
        return np.r_[joint, support*params[24:30]]
    return joint


def extra_axes(names):
    return [names.index(n) for n in ['left_ankle_roll_joint','waist_roll_joint','waist_pitch_joint',
            'left_shoulder_roll_joint','right_shoulder_roll_joint','right_shoulder_pitch_joint']]


def make_controller(ref, schema, config):
    baseline=make_baseline(ref,schema,config)
    indices=layout(schema['joint_names'])
    params=np.asarray(config['vertical_parameters'],dtype=float)
    if params.shape not in ((18,), (24,), (30,)) or not np.all(np.isfinite(params)):
        raise ValueError('Invalid swing, support or pre-impact coefficients')
    if params.size==30:
        indices+=extra_axes(schema['joint_names'])
    def control(ctx,target):
        target=baseline(ctx,target)
        target[indices]+=correction(ctx['t'],params)
        cap=config.get('impact_ankle_pitch_cap')
        if cap is not None and 2.31 <= ctx['t'] <= 2.51:
            # Cap the commanded ankle extension while the fast impact approaches the original hard stop.
            target[indices[4]]=min(target[indices[4]],cap)
        return target
    return control
