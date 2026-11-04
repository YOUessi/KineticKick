"""Bounded center-of-pressure feedback through the existing stance-leg motors.

The fixed-height inverted-pendulum relation supplies a feedback target only;
actual success is checked in the full free-root, contact-enabled simulation.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def support_cop_correction(com, velocity, foot, jacobian, mass, cfg):
    heading = Rotation.from_euler('z', Rotation.from_matrix(foot[:3,:3]).as_euler('xyz')[2]).as_matrix()
    reference = foot[:3,3]+heading@np.array([.025,0.,-.035])
    height = max(.25,float(com[2]-reference[2]))
    acceleration = -cfg['position_gain']*(com[:2]-reference[:2])-cfg['velocity_gain']*velocity[:2]
    cop = np.r_[com[:2]-height/9.81*acceleration,reference[2]]
    local = heading.T@(cop-foot[:3,3])
    raw = local.copy()
    local[0] = np.clip(local[0],cfg['foot_x_min'],cfg['foot_x_max'])
    local[1] = np.clip(local[1],-cfg['foot_y_limit'],cfg['foot_y_limit'])
    cop = foot[:3,3]+heading@local
    displacement = np.r_[cop[:2]-com[:2],0.]
    moment = np.cross(displacement,np.array([0.,0.,mass*9.81]))
    torque = -jacobian[3:].T@moment
    capture = com[:2]+velocity[:2]/np.sqrt(9.81/height)
    return torque, dict(com_xy=com[:2].tolist(),velocity_xy=velocity[:2].tolist(),capture_xy=capture.tolist(),
        desired_cop_xy=cop[:2].tolist(),local_cop_xy=local[:2].tolist(),unclipped_local_cop_xy=raw[:2].tolist(),
        correction_torque=torque.tolist())
