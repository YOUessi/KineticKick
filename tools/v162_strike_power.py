"""Interpolate continuous swing targets between trained speed and goal-plane coordinates."""
import numpy as np
from v161_strike_search import make_controller as make_swing


def bracket(grid, value):
    grid=np.asarray(grid,dtype=float)
    if np.any(np.diff(grid)<=0) or not grid[0]<=value<=grid[-1]:
        raise ValueError('Command outside the registered training interval')
    i=min(len(grid)-2,int(np.searchsorted(grid,value,side='right')-1))
    return i,(value-grid[i])/(grid[i+1]-grid[i])


def coefficients(config,speed,aim):
    if config.get('map_type')=='local_inverse':
        if not (config['domain']['speed'][0]<=speed<=config['domain']['speed'][1]
                and config['domain']['aim'][0]<=aim<=config['domain']['aim'][1]):
            raise ValueError('Request outside local calibration domain')
        error=np.array([speed,aim])-config['observation_mean']
        delta=np.asarray(config['scaled_inverse'])@error
        limit=np.asarray(config['absolute_delta_limit'])
        return np.asarray(config['base'])+np.clip(delta,-limit,limit)
    i,sv=bracket(config['speeds_mps'],speed)
    j,sx=bracket(config['aims_m'],aim)
    grid=np.asarray(config['coefficient_grid'],dtype=float)
    if grid.shape not in [(len(config['speeds_mps']),len(config['aims_m']),n) for n in (12,15)]:
        raise ValueError('Expected two swing phase coefficients for each of six joints')
    return (1-sv)*((1-sx)*grid[i,j]+sx*grid[i,j+1])+sv*((1-sx)*grid[i+1,j]+sx*grid[i+1,j+1])


def make_controller(ref,schema,config):
    ankle=schema['body_names'].index('right_ankle_roll_link')
    hip=schema['joint_names'].index('right_hip_pitch_joint')
    last_position=None
    command=None
    swing=None
    def control(ctx,target):
        nonlocal command,swing,last_position
        requested=ctx['shot_speed_mps'],ctx['aim_x_m']
        if requested!=command:
            value=coefficients(config,*requested)
            params={**config,'coefficients':value[:12]}
            if len(value)==15:
                params.update(swing_lead_s=value[12],swing_rate=value[13],hip_drive=value[14])
            swing=make_swing(ref,schema,params)
            command=requested
        out=swing(ctx,target)
        position=ctx['body_pos'][ankle,:2]
        if last_position is not None and config.get('return_damping') and ctx['first_contact_s'] is not None:
            elapsed=ctx['t']-ctx['first_contact_s']
            if .18<elapsed<.48 and ctx['foot_load'][1]<80.:
                forward=ctx['goal'][:2]-ctx['root_pos'][:2]
                forward=forward/max(np.linalg.norm(forward),1e-9)
                speed=np.dot(position-last_position,forward)/ctx['dt']
                correction=np.clip((-1.25-speed)*config['return_damping'],0.,.18)
                out[hip]-=correction
        last_position=position.copy()
        return out
    return control
