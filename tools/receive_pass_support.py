"""Keep the measured stable COM goal fixed while repositioning the other foot."""
import numpy as np
from scipy.spatial.transform import Rotation


def passing_support_correction(com,velocity,left,right,left_jacobian,mass,left_fraction,cfg,reference_xy):
    heading=Rotation.from_euler('z',Rotation.from_matrix(left[:3,:3]).as_euler('xyz')[2]).as_matrix()
    lp=left[:3,3]+heading@np.array([.025,0.,-.035])
    rp=right[:3,3]+right[:3,:3]@np.array([.025,0.,-.035])
    baseline_reference=left_fraction*lp+(1-left_fraction)*rp
    reference=np.r_[reference_xy,baseline_reference[2]]
    height=max(.25,float(com[2]-reference[2]))
    acc=-cfg['position_gain']*(com[:2]-reference[:2])-cfg['velocity_gain']*velocity[:2]
    wanted=np.r_[com[:2]-height/9.81*acc,reference[2]]
    lc=(wanted-(1-left_fraction)*rp)/left_fraction
    local=heading.T@(lc-left[:3,3]);unclipped=local.copy()
    local[0]=np.clip(local[0],cfg['foot_x_min'],cfg['foot_x_max'])
    local[1]=np.clip(local[1],-cfg['foot_y_limit'],cfg['foot_y_limit'])
    lc=left[:3,3]+heading@local
    actual=left_fraction*lc+(1-left_fraction)*rp
    # Match GravitySupport's single-left-support baseline special case.
    baseline=left_fraction*np.r_[com[:2],lp[2]]+(1-left_fraction)*rp if left_fraction>=.99 else baseline_reference
    moment=np.cross(np.r_[actual[:2]-baseline[:2],0.],np.array([0.,0.,mass*9.81]))
    torque=-left_jacobian[3:].T@moment
    return torque,dict(com_xy=com[:2].tolist(),velocity_xy=velocity[:2].tolist(),
        reference_xy=reference[:2].tolist(),left_fraction=left_fraction,
        desired_cop_xy=actual[:2].tolist(),local_cop_xy=local[:2].tolist(),
        unclipped_local_cop_xy=unclipped[:2].tolist(),correction_torque=torque.tolist())


def required_right_load(com,velocity,left,right,reference_xy,cfg):
    """Project the requested pressure center onto the two foot centers."""
    lp=left[:3,3]+left[:3,:3]@np.array([.025,0.,-.035])
    rp=right[:3,3]+right[:3,:3]@np.array([.025,0.,-.035])
    h=max(.25,float(com[2]-lp[2]))
    acceleration=-cfg['position_gain']*(com[:2]-reference_xy)-cfg['velocity_gain']*velocity[:2]
    wanted=com[:2]-h/9.81*acceleration;span=rp[:2]-lp[:2]
    return float(np.clip((wanted-lp[:2])@span/max(1e-8,span@span),0.,.45))
