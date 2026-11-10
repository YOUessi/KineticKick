#!/usr/bin/env python3
"""Executable 29DoF coupled intent adapter, NOT a learned football policy.

Intended next-stage controller boundary for phase-conditioned whole-body
actions in original G1 joint order. No ball/rigid-body state writes. Actual
contact pair and support constraints must be checked again in live PhysX.
"""
from dataclasses import dataclass
import numpy as np

DOF=29
# Ordered verbatim from the original G1 real-PhysX 29DoF telemetry, not guessed
# from arbitrary retargeting arrays. This is a safety-critical mapping.
G1_JOINT_ORDER=(
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
assert len(G1_JOINT_ORDER)==DOF and len(set(G1_JOINT_ORDER))==DOF
@dataclass(frozen=True)
class SafetyEnvelope:
    max_deviation_rad:float=.075
    max_change_per_20ms_rad:float=.006
    max_qdot_ratio:float=.85
    min_left_ground_N:float=80.
    max_right_ground_N:float=45.
    min_wrist_hip_origin_distance_m:float=.18
    max_com_to_left_ankle_xy_m:float=.20
    min_time_s:float=.85
    max_time_s:float=1.56
    # These are deliberate default safety budgets, not claims of true clearance.

def as29(name,value):
    v=np.asarray(value,dtype=float)
    if v.shape!=(DOF,) or not np.isfinite(v).all():raise ValueError(name+" must be finite 29D in G1 order")
    return v

def phase_coupled_intent(phase,amplitude_rad=.075):
    """Phase basis couples hips/torso/arm and swing knee, not a 1mrad foot patch.

    Phase in [0,1] is an external human/physics phase estimate; this fixed
    reference is only a synthetic test stimulus and is NOT CMU truth or trained
    motion. The pattern requires future human-body identification and fitting.
    """
    if not np.isfinite(phase) or not 0<=phase<=1:raise ValueError("Invalid phase")
    if not np.isfinite(amplitude_rad) or not 0<=amplitude_rad<=.075:raise ValueError("Unsafe test amplitude")
    out=np.zeros(DOF)
    # G1 measured joint-order contract fixed by v52/v53: body-wide test intent.
    # Coupled arm/torso counter-rotation, plant hip, swing knee/ankle.
    gate=np.sin(np.pi*phase)**2
    swing=np.sin(2*np.pi*phase)
    idx={0:.45,1:-.40,2:.5,3:-.35,4:.25,5:-.42,8:.38,9:-.3,
         10:.42,11:-.72,12:.67,13:.22,14:-.18,15:.54,16:-.51,
         19:.5,20:-.52,21:-.34,22:.34,23:.25,24:-.25,25:.28,
         26:-.23,27:.27,28:-.24}
    for joint,coef in idx.items():out[joint]=amplitude_rad*coef*gate
    out[1]+=amplitude_rad*.2*swing
    out[11]+=-amplitude_rad*.15*swing
    return out

def adapt(base_action,desired_delta_rad,previous_delta_rad,actual_q_rad,actual_dq_rad_s,
          joint_q_lower_rad,joint_q_upper_rad,joint_qdot_limit_rad_s,action_scale,
          left_ground_force_N,right_ground_force_N,com_to_left_ankle_xy_m,
          left_wrist_to_hip_origin_m,physical_pair_contact,time_s,
          already_hit_ball=False,env=SafetyEnvelope(),
          joint_names=None,pair_evidence_available=False):
    """Return normalized original-action update + full decision evidence.

    Gate is deliberately conservative. Return baseline unchanged if unsafe.
    This guard is NOT a proof of collision avoidance or a certified barrier:
    link-origin distance is a veto proxy only, not a minimum mesh gap. No
    contact threshold, actuator capacity or impulse can be relaxed here.
    """
    base=as29("base_action",base_action)
    target=as29("desired_delta",desired_delta_rad)
    last=as29("previous_delta",previous_delta_rad)
    q=as29("actual_joint_position",actual_q_rad)
    dq=as29("actual_joint_velocity",actual_dq_rad_s)
    lo=as29("lower_limit",joint_q_lower_rad);hi=as29("upper_limit",joint_q_upper_rad)
    vel=as29("joint_velocity_limit",joint_qdot_limit_rad_s)
    scale=as29("original_action_scale",action_scale)
    if (np.any(lo>=hi) or np.any(vel<=0) or np.any(scale==0)
        or env.max_deviation_rad<=0 or env.max_change_per_20ms_rad<=0):
        raise ValueError("Invalid original G1 joint limits / envelope")
    observed=np.asarray([left_ground_force_N,right_ground_force_N,
        com_to_left_ankle_xy_m,left_wrist_to_hip_origin_m,time_s])
    if not np.isfinite(observed).all():raise ValueError("Missing physical observation")
    # A zero-event or missing collider report cannot be interpreted as proof
    # that the collision pair is absent. Without independently certified live
    # PhysX pair evidence this proposed interface must be inactive.
    known_order = tuple(joint_names) == G1_JOINT_ORDER if joint_names is not None else False
    measured_pair = pair_evidence_available is True and isinstance(physical_pair_contact, (bool, np.bool_))
    qdot_ratio=float(np.max(np.abs(dq)/vel))
    factors=dict(
        stance=(left_ground_force_N>=env.min_left_ground_N
                and right_ground_force_N<=env.max_right_ground_N),
        qdot=(qdot_ratio<=env.max_qdot_ratio),
        center_of_mass=(com_to_left_ankle_xy_m<=env.max_com_to_left_ankle_xy_m),
        collision_margin_proxy=(left_wrist_to_hip_origin_m>=env.min_wrist_hip_origin_distance_m),
        pair_evidence_available=measured_pair,
        joint_order_verified=known_order,
        no_measured_pair_contact=measured_pair and not physical_pair_contact,
        pre_impact=not already_hit_ball,
        phase=(env.min_time_s<=time_s<=env.max_time_s),
        joint_limits=(bool(np.all(q>=lo)&np.all(q<=hi))))
    eligible=all(factors.values())
    if not eligible:
        return base.copy(),np.zeros(DOF),dict(eligible=False,gates=factors,
            qdot_ratio=qdot_ratio,change_was_zero=True,unverified_hardware_safety=True)
    # Coupled vector trust region, absolute limits and per-step slew.
    bounded=np.clip(target,-env.max_deviation_rad,env.max_deviation_rad)
    bounded=np.clip(bounded,last-env.max_change_per_20ms_rad,
                    last+env.max_change_per_20ms_rad)
    # The requested absolute actuator setpoint cannot command out of the
    # original physical joint interval; margins are 0.02 rad conservative.
    bounded=np.maximum(bounded,lo+.02-q)
    bounded=np.minimum(bounded,hi-.02-q)
    if np.any(np.abs(bounded)>env.max_deviation_rad+1e-9):
        raise ValueError("Original pose too close to limit to issue safe intent")
    cmd=base+bounded/scale
    if not np.isfinite(cmd).all():raise ValueError("Invalid normalized original action")
    return cmd,bounded,dict(eligible=True,gates=factors,qdot_ratio=qdot_ratio,
          max_commanded_delta_rad=float(np.max(np.abs(bounded))),
          all_29_joint_order_preserved=True,unverified_hardware_safety=True)
