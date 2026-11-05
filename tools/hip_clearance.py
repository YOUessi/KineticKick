"""URDF capsule clearance and bounded post-contact hip deceleration.

This predicts a stopping angle; it is not a dynamics guarantee. Original
PhysX self-collision, force limits and termination remain the acceptance test.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import torch


class HipClearance:
    def __init__(self, urdf: Path, device):
        xml=ET.parse(urdf).getroot()
        def named(tag,name):return next(e for e in xml.findall(tag) if e.attrib['name']==name)
        def origin(element):
            e=element.find('origin');a={} if e is None else e.attrib
            xyz=np.fromstring(a.get('xyz','0 0 0'),sep=' ')
            rpy=np.fromstring(a.get('rpy','0 0 0'),sep=' ')
            return xyz,Rotation.from_euler('xyz',rpy).as_matrix()
        def tensor(x):return torch.tensor(x,dtype=torch.float32,device=device)
        pitch=named('joint','right_hip_pitch_joint');roll=named('joint','right_hip_roll_joint')
        if pitch.find('axis').attrib['xyz']!='0 1 0' or roll.find('axis').attrib['xyz']!='1 0 0':
            raise ValueError('Expected original G1 hip axes')
        p,r=origin(pitch);self.pitch_p=tensor(p);self.pitch_r=tensor(r)
        p,r=origin(roll);self.roll_p=tensor(p);self.roll_r=tensor(r)
        pelvis=named('link','pelvis').find('collision');thigh=named('link','right_hip_roll_link').find('collision')
        self.center=tensor(origin(pelvis)[0])
        sphere=pelvis.find('geometry/sphere');capsule=thigh.find('geometry/cylinder')
        if sphere is None or capsule is None:raise ValueError('Expected original pelvis sphere and imported thigh capsule')
        self.radius=float(sphere.attrib['radius'])+float(capsule.attrib['radius'])
        p,r=origin(thigh);axis=r[:,2]*float(capsule.attrib['length'])/2.
        self.start=tensor(p-axis);self.end=tensor(p+axis)

    def gap(self,pitch,roll):
        c=pitch.cos();s=pitch.sin();z=torch.zeros_like(c);o=torch.ones_like(c)
        ry=torch.stack([c,z,s,z,o,z,-s,z,c],-1).reshape(-1,3,3)
        c=roll.cos();s=roll.sin()
        rx=torch.stack([o,z,z,z,c,-s,z,s,c],-1).reshape(-1,3,3)
        rp=self.pitch_r@ry
        center=self.pitch_p+(rp@self.roll_p).squeeze(-1)
        rotation=rp@self.roll_r@rx
        a=center+(rotation@self.start).squeeze(-1);b=center+(rotation@self.end).squeeze(-1)
        segment=b-a
        fraction=((self.center-a)*segment).sum(-1)/segment.square().sum(-1).clamp_min(1e-12)
        closest=a+fraction.clamp(0.,1.)[:,None]*segment
        return (self.center-closest).norm(dim=-1)-self.radius

    def pitch_floor(self,roll,clearance):
        lo=torch.full_like(roll,-1.8);hi=torch.full_like(roll,-.5)
        feasible=(self.gap(lo,roll)<=clearance)&(self.gap(hi,roll)>=clearance)
        for _ in range(18):
            mid=(lo+hi)*.5;safe=self.gap(mid,roll)>=clearance
            hi=torch.where(safe,mid,hi);lo=torch.where(safe,lo,mid)
        return hi,feasible


def braking_correction(position,velocity,floor,active,*,deceleration=60.,gain=2.,maximum=.25):
    if deceleration<=0 or gain<=0 or not 0<maximum<=.4:
        raise ValueError('Positive deceleration/gain and bounded correction required')
    closing=velocity.clamp_max(0.)
    predicted=position-closing.square()/(2.*deceleration)
    correction=(gain*(floor-predicted)).clamp(0.,maximum)
    correction=torch.where(active&(velocity<0.),correction,torch.zeros_like(correction))
    return correction,predicted


def pulse_has_finished(command_frame,velocity_target,end_frame):
    return (command_frame>=end_frame)&(velocity_target>=0.)
