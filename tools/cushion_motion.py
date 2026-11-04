"""Ballistic interception and reachable, velocity-continuous cushioning targets."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from cartesian_strike import LegKinematics


def descending_time(z, vz, height, gravity=9.81):
    discriminant = vz*vz + 2*gravity*(z-height)
    if discriminant < 0:
        return None
    time = (vz + math.sqrt(discriminant))/gravity
    return time if time >= 0 else None


def smooth_step(u):
    u = np.clip(u, 0., 1.)
    return u*u*u*(10+u*(-15+6*u))


def hermite(p0, p1, v0, v1, duration, elapsed):
    u = np.clip(elapsed/duration, 0., 1.)
    p = (2*u**3-3*u*u+1)*p0 + (-2*u**3+3*u*u)*p1 + (u**3-2*u*u+u)*duration*v0 + (u**3-u*u)*duration*v1
    v = (6*u*u-6*u)/duration*p0 + (-6*u*u+6*u)/duration*p1 + (3*u*u-4*u+1)*v0 + (3*u*u-2*u)*v1
    return p, v


class SideLegKinematics(LegKinematics):
    def __init__(self, urdf, side):
        if side == 'right':
            super().__init__(urdf)
            return
        if side != 'left':
            raise ValueError(side)
        self.names = ['left_'+n+'_joint' for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        joints = ET.parse(urdf).getroot().findall('joint')
        self.origins, self.axes, self.limits = [], [], []
        for name in self.names:
            j = next(j for j in joints if j.get('name') == name);o=j.find('origin');t=np.eye(4)
            t[:3,3] = np.fromstring(o.get('xyz','0 0 0'),sep=' ')
            t[:3,:3] = Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
            self.origins.append(t);self.axes.append(np.fromstring(j.find('axis').get('xyz'),sep=' '))
            self.limits.append([float(j.find('limit').get(k)) for k in ('lower','upper')])
        self.limits=np.array(self.limits)


class CushionMotion:
    def __init__(self, plan):
        self.plan=plan;self.cfg=plan['cushion'];self.initial_foot=np.array(plan['feet']['right'],float)
        self.initial_root=np.array(plan['root_pose'],float);self.contact_time=None
        self.hit_position=None;self.hit_velocity=None;self.predictions=[];self.angular_velocity=np.zeros(3)
        self.receive_rotation=self.initial_foot[:3,:3]@Rotation.from_euler('y',self.cfg.get('receive_pitch_rad',0.)).as_matrix()

    def update_prediction(self, now, ball, velocity, touched=False):
        if touched or self.contact_time is not None and now >= self.contact_time-self.cfg['freeze_prediction_lead_s']:
            return
        dt=descending_time(ball[2],velocity[2],self.cfg['contact_ball_height_m'])
        if dt is None:return
        incoming=np.array(velocity,float);incoming[2]-=9.81*dt
        normal=-incoming/np.linalg.norm(incoming)
        impact=np.array(ball,float)+np.array(velocity)*dt;impact[2]-=.5*9.81*dt*dt
        # Original middle forefoot capsule: endpoint x=.132, center z=-.025,
        # radius=.01. Ball radius=.11. No geometry/material changes.
        endpoint=self.receive_rotation@np.array([.132,0,-.025])
        self.hit_position=impact-endpoint-.12*normal
        self.hit_velocity=incoming*self.cfg['velocity_fraction']
        self.contact_time=now+dt
        self.predictions.append(dict(time_s=now,expected_contact_time_s=self.contact_time,
            incoming_velocity=incoming.tolist(),normal=normal.tolist(),foot_hit_position=self.hit_position.tolist(),
            foot_hit_velocity=self.hit_velocity.tolist()))

    def targets(self, now):
        foot=self.initial_foot.copy();root=np.eye(4)
        root[:3,3]=self.initial_root[:3];root[:3,:3]=Rotation.from_quat(self.initial_root[[4,5,6,3]]).as_matrix()
        if self.contact_time is None:return foot,np.zeros(3),root,'receive'
        start=self.contact_time-self.cfg['trajectory_lead_s'];end=self.contact_time
        rotation_end=max(.08,self.contact_time-.12)
        ru=np.clip(now/rotation_end,0.,1.)
        pitch=self.cfg.get('receive_pitch_rad',0.)*smooth_step(ru)
        foot[:3,:3]=self.initial_foot[:3,:3]@Rotation.from_euler('y',pitch).as_matrix()
        pitch_rate=self.cfg.get('receive_pitch_rad',0.)*30*ru*ru*(1-ru)**2/rotation_end
        self.angular_velocity=self.initial_foot[:3,:3][:,1]*pitch_rate
        crouch_start=start-.04
        root[2,3]-=self.cfg['crouch_depth_m']*smooth_step((now-crouch_start)/(end-.02-crouch_start))
        if now<start:return foot,np.zeros(3),root,'prepare'
        if now<=end:
            foot[:3,3],speed=hermite(self.initial_foot[:3,3],self.hit_position,np.zeros(3),self.hit_velocity,end-start,now-start)
            return foot,speed,root,'anticipate'
        duration=self.cfg['follow_seconds'];dt=min(now-end,duration)
        foot[:3,3]=self.hit_position+self.hit_velocity*(dt-.5*dt*dt/duration)
        speed=self.hit_velocity*(1-dt/duration)
        return foot,speed,root,'cushion' if dt<duration else 'settle'
