#!/usr/bin/env python3
"""Explicit CPU contract for a future human-to-G1 29DoF temporal policy.

Requires ACTUAL geometry collision clearance evidence, not link-origin distance,
contact net force symmetry or a guessed signed distance. Until a GPU G1 collider
witness is verified, output the unchanged original action (fail closed).
Nothing here certifies hardware safety, learned human biomechanics, CBF or MPC.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from wholebody_safety_interface_v055 import G1_JOINT_ORDER
N=len(G1_JOINT_ORDER)
assert N==29
GROUPS={
    'support_leg':('left_hip_pitch_joint','left_hip_roll_joint','left_knee_joint'),
    'torso':('waist_yaw_joint','waist_pitch_joint','waist_roll_joint'),
    'swing_leg':('right_hip_pitch_joint','right_knee_joint','right_ankle_pitch_joint'),
    'arms':('left_shoulder_pitch_joint','right_shoulder_pitch_joint','left_elbow_joint','right_elbow_joint'),
}
MAX_RAD=.075
MAX_STEP_RAD=.006
MIN_CLEARANCE_M=.035
@dataclass(frozen=True)
class NativeGeometryWitness:
    backend:str
    left_arm_collider_path:str
    hip_collider_path:str
    observed_min_signed_mesh_distance_m:float
    paths_verified_as_G1_original_colliders:bool
    matches_current_physics_frame:bool
    # This cannot be substituted with v54 net force or link-origin distance.

@dataclass(frozen=True)
class RealG1State:
    joint_names:tuple[str,...]
    physical_frame_time_s:float
    joint_pos_rad:np.ndarray
    measured_joint_vel_rad_s:np.ndarray
    actual_joint_vel_limits_rad_s:np.ndarray
    actual_joint_pos_limits_rad:np.ndarray
    measured_left_support_normal_N:float
    measured_right_support_normal_N:float
    measured_nonfoot_contact_N:float
    measured_body_tilt_deg:float
    measured_support_margin_m:Optional[float]
    actual_collider_witness:Optional[NativeGeometryWitness]
    ball_contact_occurred:bool

@dataclass(frozen=True)
class CommandDecision:
    action:np.ndarray
    reason:str
    actual_joint_delta_rad:np.ndarray
    eligible_for_physx_deployment:bool
    certified_hardware_collision_safety:bool=False

def _array(x,shape,name):
    a=np.asarray(x,dtype=float)
    if a.shape!=shape or not np.isfinite(a).all():
        raise ValueError('Invalid G1 '+name+' / shape / finite status')
    return a

def stage_29dof_action(original_action, learned_intent_rad, prev_applied_rad,
                       original_motor_action_scale, state:RealG1State)->CommandDecision:
    """Coupled 29DoF phase goal -> bounded original motor action, missing proof -> baseline.

    Inputs are current original-G1 measured values. Geometry witness is an explicit
    newly verified runtime collider query, which the project presently LACKS.
    Intended to be applied only prior to impact with original 14-second recovery.
    """
    orig=_array(original_action,(N,),'original_action')
    intent=_array(learned_intent_rad,(N,),'intent')
    prev=_array(prev_applied_rad,(N,),'previous_delta')
    scale=_array(original_motor_action_scale,(N,),'original_scale')
    if np.any(scale==0):raise ValueError('Zero motor scale')
    if state.joint_names!=G1_JOINT_ORDER:raise ValueError('Actual original G1 joint ordering unverified')
    q=_array(state.joint_pos_rad,(N,),'q')
    dq=_array(state.measured_joint_vel_rad_s,(N,),'dq')
    qlim=_array(state.actual_joint_pos_limits_rad,(N,2),'q limits')
    vlim=_array(state.actual_joint_vel_limits_rad_s,(N,),'velocity limits')
    if np.any(vlim<=0) or np.any(qlim[:,0]>=qlim[:,1]):
        raise ValueError('Original G1 limits are not valid')
    def no(reason):
        return CommandDecision(orig.copy(),reason,np.zeros_like(orig),False)
    measurements=(state.physical_frame_time_s,state.measured_left_support_normal_N,
                  state.measured_right_support_normal_N,state.measured_nonfoot_contact_N,
                  state.measured_body_tilt_deg)
    if not np.isfinite(measurements).all():return no('nonfinite_real_physics')
    if not (.86<=state.physical_frame_time_s<=1.56) or state.ball_contact_occurred:
        return no('outside_original_preimpact_window_or_already_touched_ball')
    # No unverified data, zero-dynamics/geometry fixture, or post-hoc clearance.
    w=state.actual_collider_witness
    if w is None:return no('no_verified_original_GPU_G1_mesh_collider_witness')
    if (w.backend!='cuda:0' or not w.paths_verified_as_G1_original_colliders or
        not w.matches_current_physics_frame or
        '/Robot/left_' not in w.left_arm_collider_path or
        '/Robot/left_hip_' not in w.hip_collider_path or
        not np.isfinite(w.observed_min_signed_mesh_distance_m) or
        w.observed_min_signed_mesh_distance_m<MIN_CLEARANCE_M):
        return no('uncertified_or_insufficient_original_mesh_clearance')
    margin=state.measured_support_margin_m
    if margin is None or not np.isfinite(margin) or margin<.025:
        return no('support_polygon_margin_missing_or_too_small')
    if (state.measured_left_support_normal_N<80 or state.measured_right_support_normal_N>65 or
        state.measured_nonfoot_contact_N>=5 or state.measured_body_tilt_deg>20):
        return no('real_support_tilt_or_nonfoot_guard')
    if (np.abs(dq)/vlim).max()>.94 or np.any(q<=qlim[:,0]+.025) or np.any(q>=qlim[:,1]-.025):
        return no('actual_original_G1_qdot_or_position_margin_guard')
    # Reject a disguised one-foot correction: a real wholebody intent across groups.
    if np.count_nonzero(np.abs(intent)>1e-4)<12:
        return no('not_a_wholebody_multijoint_intent')
    if any(max(abs(intent[G1_JOINT_ORDER.index(x)]) for x in names)<1e-4
           for names in GROUPS.values()):
        return no('one_or_more_wholebody_kinetic_chain_groups_missing')
    target=np.clip(intent,-MAX_RAD,MAX_RAD)
    applied=prev+np.clip(target-prev,-MAX_STEP_RAD,MAX_STEP_RAD)
    # One-step predicted motor reference only, not claimed barrier-certificate.
    if np.any(q+applied<=qlim[:,0]+.025) or np.any(q+applied>=qlim[:,1]-.025):
        return no('predicted_joint_reference_margin_guard')
    action=orig+applied/scale
    if not np.isfinite(action).all():raise ValueError('invalid normalized motor action')
    return CommandDecision(action,'accepted_under_specified_runtime_witness',applied,True)
