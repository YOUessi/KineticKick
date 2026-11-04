"""Static gravity/support feedforward through the existing joint actuators."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

class GravitySupport:
    def __init__(self,urdf,joint_names):
        x=ET.parse(urdf).getroot();self.names=joint_names;self.joints=[];self.inertials={}
        for l in x.findall('link'):
            i=l.find('inertial')
            if i is not None:
                o=i.find('origin');self.inertials[l.get('name')]=(float(i.find('mass').get('value')),np.fromstring(o.get('xyz','0 0 0'),sep=' '))
        pending=x.findall('joint');known={'pelvis'}
        while pending:
            before=len(pending)
            for j in list(pending):
                parent=j.find('parent').get('link');child=j.find('child').get('link')
                if parent not in known:continue
                o=j.find('origin');t=np.eye(4)
                if o is not None:t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
                axis=np.fromstring(j.find('axis').get('xyz','0 0 1'),sep=' ') if j.find('axis') is not None else np.zeros(3)
                self.joints.append((j.get('name'),parent,child,t,axis,j.get('type')!='fixed'));known.add(child);pending.remove(j)
            if len(pending)==before:raise ValueError('Disconnected URDF')
    def torques(self,q,root,left_fraction=1.):
        poses={'pelvis':root};ancestors={'pelvis':[]};axes={};centers={};values=dict(zip(self.names,q))
        for name,parent,child,origin,axis,active in self.joints:
            t=poses[parent]@origin;ancestors[child]=list(ancestors[parent])
            if active:
                axes[name]=t[:3,:3]@axis;centers[name]=t[:3,3].copy();ancestors[child].append(name)
                t=t.copy();t[:3,:3]=t[:3,:3]@Rotation.from_rotvec(axis*values[name]).as_matrix()
            poses[child]=t
        total=sum(m for m,_ in self.inertials.values());com=sum(m*(poses[n][:3,3]+poses[n][:3,:3]@c) for n,(m,c) in self.inertials.items())/total
        generalized={n:0. for n in self.names}
        def add(link,point,force):
            for n in ancestors[link]:generalized[n]+=axes[n]@np.cross(point-centers[n],force)
        for n,(m,c) in self.inertials.items():add(n,poses[n][:3,3]+poses[n][:3,:3]@c,np.array([0,0,-9.81*m]))
        for side,fraction in [('left',left_fraction),('right',1-left_fraction)]:
            link=side+'_ankle_roll_link';p=poses[link][:3,3]+poses[link][:3,:3]@np.array([.025,0,-.035])
            if left_fraction>=.99 and side=='left':p[:2]=com[:2]
            add(link,p,np.array([0,0,9.81*total*fraction]))
        return -np.array([generalized[n] for n in self.names])
