"""Local balance feedback in double support with gains from the CPU plant."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def state(p,q,v,angles,speeds):
    e=Rotation.from_quat(q[[1,2,3,0]]).as_euler('xyz')
    return np.r_[p,e,v,angles,speeds]

def make_controller(ref,schema,config):
    d=np.load(Path(config['feedback_file']),allow_pickle=False)
    K=d['gain'];idx=d['indexes'];gain=config['gain'];memory=[]
    xref=command=t0=None
    def control(ctx,target):
        nonlocal xref,command,t0
        if ctx['first_contact_s'] is None or ctx['t']<max(config['start_s'],ctx['first_contact_s']+.80):
            return target
        x=state(ctx['root_pos'],ctx['root_quat'],ctx['root_velocity'],ctx['q'],ctx['dq'])
        if xref is None:
            if min(ctx['foot_load'])<75 or np.linalg.norm(ctx['root_velocity'][:2])>.12:
                memory.clear();return target
            memory.append((x,target.copy()))
            if len(memory)<10:return target
            xref=np.mean([z[0] for z in memory],axis=0);xref[6:12]=0;xref[41:]=0
            command=np.mean([z[1] for z in memory],axis=0);t0=ctx['t']
        dx=x-xref;dx[5]=(dx[5]+np.pi)%(2*np.pi)-np.pi
        fix=np.clip(-K@dx[idx],-config['max_delta'],config['max_delta'])
        alpha=np.clip((ctx['t']-t0)/.24,0.,1.)
        return (1-alpha)*target+alpha*(command+gain*fix)
    return control
