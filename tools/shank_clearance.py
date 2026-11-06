"""Original-URDF two-shank clearance and short-horizon swing path correction.

Constant joint-velocity predictions are local approximations. Physical contact,
instep and recovery acceptance remain authoritative.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import torch


def segment_gap(a,b,c,d,radius):
    u=b-a;v=d-c;w=a-c
    aa=(u*u).sum(-1);bb=(u*v).sum(-1);cc=(v*v).sum(-1)
    dd=(u*w).sum(-1);ee=(v*w).sum(-1);den=aa*cc-bb*bb
    s=(bb*ee-cc*dd)/den.clamp_min(1e-12);t=(aa*ee-bb*dd)/den.clamp_min(1e-12)
    interior=(den>1e-12)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    mid=(w+s[...,None]*u-t[...,None]*v).square().sum(-1)
    def point(p,x,y):
        line=y-x;f=((p-x)*line).sum(-1)/line.square().sum(-1).clamp_min(1e-12)
        return (p-x-f.clamp(0,1)[...,None]*line).square().sum(-1)
    distances=torch.stack([point(a,c,d),point(b,c,d),point(c,a,b),point(d,a,b),torch.where(interior,mid,torch.full_like(mid,float('inf')))],-1)
    return distances.min(-1).values.clamp_min(0).sqrt()-radius


class ShankClearance:
    def __init__(self,urdf,joint_names,device):
        xml=ET.parse(urdf).getroot();self.device=device;self.chains=[];self.ends=[];self.radius=0.
        def tensor(x):return torch.tensor(x,dtype=torch.float32,device=device)
        def origin(element):
            el=element.find('origin');a={} if el is None else el.attrib
            return np.fromstring(a.get('xyz','0 0 0'),sep=' '),Rotation.from_euler('xyz',np.fromstring(a.get('rpy','0 0 0'),sep=' ')).as_matrix()
        for side in ['left','right']:
            chain=[]
            for part in ['hip_pitch','hip_roll','hip_yaw','knee']:
                name=f'{side}_{part}_joint';joint=next(x for x in xml.findall('joint') if x.attrib['name']==name)
                p,r=origin(joint);axis=np.fromstring(joint.find('axis').attrib['xyz'],sep=' ');x,y,z=axis
                skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                chain.append((joint_names.index(name),tensor(p),tensor(r),tensor(skew)))
            self.chains.append(chain)
            link=next(x for x in xml.findall('link') if x.attrib['name']==f'{side}_knee_link');collision=link.find('collision');p,r=origin(collision);shape=collision.find('geometry/cylinder')
            if shape is None:raise ValueError('Expected original shank capsule')
            h=float(shape.attrib['length'])/2.;self.radius+=float(shape.attrib['radius']);axis=r[:,2]*h
            self.ends.append((tensor(p-axis),tensor(p+axis)))
        self.roll_id=joint_names.index('right_hip_roll_joint')

    def gap(self,q):
        points=[]
        for chain,(a,b) in zip(self.chains,self.ends):
            r=torch.eye(3,device=q.device).expand(len(q),3,3);p=torch.zeros(len(q),3,device=q.device)
            for index,origin,rotation,skew in chain:
                p=p+(r@origin).squeeze(-1);r=r@rotation
                angle=q[:,index,None,None]
                r=r@(torch.eye(3,device=q.device)+angle.sin()*skew+(1-angle.cos())*(skew@skew))
            points.extend([p+(r@a).squeeze(-1),p+(r@b).squeeze(-1)])
        return segment_gap(*points,self.radius)

    def swept_gap(self,q,v,horizon):
        times=torch.linspace(0.,horizon,7,device=q.device)
        values=self.gap((q[:,None]+v[:,None]*times[None,:,None]).reshape(-1,q.shape[-1])).reshape(len(q),-1)
        value,index=values.min(-1)
        return value,times[index]

    def command(self,q,v,active,previous,*,margin,horizon,maximum,maximum_step):
        desired=torch.zeros_like(previous);predicted=torch.zeros_like(previous);when=torch.zeros_like(previous)
        ids=active.nonzero(as_tuple=False).flatten()
        if len(ids):
            distance,t=self.swept_gap(q[ids],v[ids],horizon);pose=q[ids]+v[ids]*t[:,None]
            perturb=torch.zeros_like(pose);perturb[:,self.roll_id]=.001
            gradient=(self.gap(pose+perturb)-self.gap(pose-perturb))/.002
            change=(margin-distance).clamp_min(0)*gradient/(gradient.square()+1e-6)
            desired[ids]=change.clamp(-maximum,maximum);predicted[ids]=distance;when[ids]=t
        delta=(previous+(desired-previous).clamp(-maximum_step,maximum_step)).clamp(-maximum,maximum)
        return delta,dict(predicted_gap=predicted,closest_time=when)


def ankle_compensation(hip_request,previous,active,maximum_step):
    """Track each applied channel independently after actuator-limit clipping."""
    desired=torch.where(active,-hip_request,torch.zeros_like(previous))
    return previous+(desired-previous).clamp(-maximum_step,maximum_step)
