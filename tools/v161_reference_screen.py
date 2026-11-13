"""Compare released motion geometries and qualify all three stylized reference rollouts."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from v160_tracking_cpu import DEFAULT_ASSETS, PROTOCOL, SCHEMA

def rank():
    cfg=json.loads(PROTOCOL.read_text())
    names=json.loads((DEFAULT_ASSETS/SCHEMA).read_text())['body_names']
    rows=[]
    for clip in sorted(cfg['motion_sha256']):
        with np.load(DEFAULT_ASSETS/clip) as ref:
            side=str(ref['kick_leg'].item())
            index=names.index(side+'_ankle_roll_link')
            foot=Rotation.from_quat(ref['body_quat_w'][:,index][:,[1,2,3,0]])
            point=ref['body_pos_w'][:,index]+foot.apply([.075,0,-.015])
            velocity=np.gradient(point,1/50,axis=0)
            peak=int(np.argmax(np.linalg.norm(velocity,axis=1)))
            ball=np.asarray(cfg['ball_initial_xyz_m'])
            rows.append(dict(clip=clip,side=side,speed_peak_mps=float(np.linalg.norm(velocity[peak])),
                t_s=peak/50,normal_world=foot[peak].apply([0,0,1]).tolist(),
                closest_surface_point_to_initial_ball_m=float(np.linalg.norm(point-ball,axis=1).min())))
    return rows
if __name__=='__main__':
    print(json.dumps(rank(),indent=2))
