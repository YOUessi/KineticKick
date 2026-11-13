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
    i,sv=bracket(config['speeds_mps'],speed)
    j,sx=bracket(config['aims_m'],aim)
    grid=np.asarray(config['coefficient_grid'],dtype=float)
    if grid.shape!=(len(config['speeds_mps']),len(config['aims_m']),12):
        raise ValueError('Expected two swing phase coefficients for each of six joints')
    return (1-sv)*((1-sx)*grid[i,j]+sx*grid[i,j+1])+sv*((1-sx)*grid[i+1,j]+sx*grid[i+1,j+1])


def make_controller(ref,schema,config):
    command=None
    swing=None
    def control(ctx,target):
        nonlocal command,swing
        requested=ctx['shot_speed_mps'],ctx['aim_x_m']
        if requested!=command:
            swing=make_swing(ref,schema,{**config,'coefficients':coefficients(config,*requested)})
            command=requested
        return swing(ctx,target)
    return control
