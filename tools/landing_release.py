"""Measured capsule approach and one-way smooth release of strike references.

This module changes reference weights only. It never writes physical state,
changes actuator limits, or applies external forces. Import after AppLauncher
inside simulator processes; pure tensor functions also support CPU checks.
"""
import math
import torch


def capsule_landing_state(points,radii,position,rotation,linear_velocity,angular_velocity,
                          ball_position,ball_radius,ground_z,lookahead_s):
    """Original capsule surfaces and first-order endpoint approach, world frame."""
    if points.ndim!=2 or points.shape[1]!=3 or len(points)%2 or radii.shape!=(len(points),):
        raise ValueError('Paired capsule endpoints and radii required')
    lever=torch.einsum('nij,pj->npi',rotation,points)
    world=position[:,None,:]+lever
    velocity=linear_velocity[:,None,:]+torch.linalg.cross(angular_velocity[:,None,:].expand_as(lever),lever,dim=-1)
    height=world[:,:,2]-radii[None,:]-ground_z[:,None]
    predicted=height+lookahead_s*velocity[:,:,2].clamp_max(0.)
    a,b=world[:,::2],world[:,1::2];segment=b-a;delta=ball_position[:,None,:]-a
    fraction=(delta*segment).sum(-1)/segment.square().sum(-1).clamp_min(1e-12)
    closest=a+fraction.clamp(0.,1.)[:,:,None]*segment
    gap=(ball_position[:,None,:]-closest).norm(dim=-1)-radii[None,::2]-ball_radius
    return dict(right_lowest_height_m=height.amin(-1),min_vertical_velocity_mps=velocity[:,:,2].amin(-1),
                predicted_clearance_m=predicted.amin(-1),capsule_ball_gap_m=gap.amin(-1),
                toe_axis_up_dot=rotation[:,2,0])


def retain_operator(raw,projected,retention):
    """Fade an original command operator, including its projection, continuously.

    The frozen raw target may be outside a soft kinematic interval. Abruptly
    removing projection at retention==0 would create a new discontinuity.
    Preserve both endpoints exactly without adding any new joint clamp.
    """
    r=retention.to(dtype=raw.dtype)[:,None]
    mixed=raw+r*(projected-raw)
    return torch.where(r==1.,projected,torch.where(r==0.,raw,mixed))


class LandingRelease:
    def __init__(self,n,device,config):
        self.n=n;self.cfg=dict(config)
        keys=['lookahead_s','maximum_height_m','predicted_margin_m','minimum_descent_mps','separation_s',
              'separation_gap_m','ball_force_max_N','maximum_ground_force_N','duration_s','phase_lag_scale_s']
        if config['mode'] not in ('all','pose') or not all(math.isfinite(config[k]) and config[k]>0 for k in keys):
            raise ValueError('Finite positive landing settings and all/pose mode required')
        if not .04<=config['duration_s']<=.2 or not .04<=config['lookahead_s']<=.25 or config['separation_s']<.01:
            raise ValueError('Landing timing outside declared bounded study')
        self.separated_s=torch.zeros(n,device=device)
        self.previous_gap=torch.full((n,),float('nan'),device=device)
        self.trigger_time_s=torch.full((n,),-1.,device=device)
        self.retention=torch.ones(n,device=device)
        self.release_strength=torch.zeros(n,device=device)

    def reset(self,ids):
        self.separated_s[ids]=0.;self.previous_gap[ids]=float('nan')
        self.trigger_time_s[ids]=-1.;self.retention[ids]=1.;self.release_strength[ids]=0.

    def after_physics(self,contact_latched,ball_normal_force,gap,dt):
        if not math.isfinite(dt) or abs(dt-.005)>1e-8:raise ValueError('Original5ms physics interval required')
        separated=contact_latched&(ball_normal_force<=self.cfg['ball_force_max_N'])
        separated &= (gap>=self.cfg['separation_gap_m'])&torch.isfinite(self.previous_gap)&(gap>self.previous_gap)
        self.separated_s=torch.where(separated,self.separated_s+dt,torch.zeros_like(self.separated_s))
        self.previous_gap=gap.clone()

    def command(self,now,geometry,pulse_finished,right_ground_force,*,phase_lag_s):
        risk=(geometry['right_lowest_height_m']<=self.cfg['maximum_height_m'])
        risk &= geometry['min_vertical_velocity_mps'] < -self.cfg['minimum_descent_mps']
        risk &= geometry['predicted_clearance_m']<=self.cfg['predicted_margin_m']
        ready=(self.separated_s+1e-8>=self.cfg['separation_s'])&pulse_finished
        ready &= right_ground_force<self.cfg['maximum_ground_force_N']
        finite=torch.isfinite(now)&torch.isfinite(right_ground_force)&torch.isfinite(phase_lag_s)
        for key in ('right_lowest_height_m','min_vertical_velocity_mps','predicted_clearance_m'):
            finite &= torch.isfinite(geometry[key])
        trigger=(self.trigger_time_s<0)&ready&risk&finite&(phase_lag_s>1e-6)
        strength=(phase_lag_s/self.cfg['phase_lag_scale_s']).clamp(0.,1.)
        self.release_strength=torch.where(trigger,strength,self.release_strength)
        self.trigger_time_s=torch.where(trigger,now,self.trigger_time_s)
        u=((now-self.trigger_time_s)/self.cfg['duration_s']).clamp(0.,1.)
        smooth=u*u*u*(10.+u*(-15.+6.*u))
        self.retention=torch.where(self.trigger_time_s>=0,1.-self.release_strength*smooth,torch.ones_like(smooth)).clamp(0.,1.)
        return self.retention
