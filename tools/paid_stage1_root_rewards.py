"""Research ablation: GLOBAL football reference trajectory tracking rewards.

Unlike existing relative-body terms, global planar root tracking does not align
each reference body to the robot's *current* pose and cannot hide opposite travel.
"""
from __future__ import annotations
import torch

def global_torso_planar_tracking_exp(env, command_name: str, std: float = .8):
    cmd=env.command_manager.get_term(command_name)
    dist=torch.linalg.vector_norm(cmd.anchor_pos_w[:,:2]-cmd.robot_anchor_pos_w[:,:2],dim=-1)
    return torch.exp(-dist/std)

def global_torso_planar_huber(env, command_name: str, delta_m: float = .25):
    cmd=env.command_manager.get_term(command_name)
    error=cmd.robot_anchor_pos_w[:,:2]-cmd.anchor_pos_w[:,:2]
    norm=torch.linalg.vector_norm(error,dim=-1)
    return torch.where(norm < delta_m,
       (norm**2)/(2*delta_m),
       norm-delta_m*.5)

def global_torso_linear_velocity_tracking(env, command_name: str, std: float = 1.2):
    cmd=env.command_manager.get_term(command_name)
    err=torch.linalg.vector_norm(cmd.robot_anchor_lin_vel_w[:,:2]-cmd.anchor_lin_vel_w[:,:2],dim=-1)
    return torch.exp(-err/std)

def global_anchor_xy_out_of_bounds(env, command_name: str, threshold: float = 0.7):
    """Terminate if the robot drifts too far from absolute planned soccer root trajectory."""
    cmd=env.command_manager.get_term(command_name)
    dist=torch.linalg.vector_norm(cmd.anchor_pos_w[:,:2]-cmd.robot_anchor_pos_w[:,:2],dim=-1)
    return dist>threshold
