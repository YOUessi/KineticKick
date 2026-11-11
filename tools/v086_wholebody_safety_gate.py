"""Fail-closed readiness checks for proposed 29DoF support/contact-safe controller.

No actuator commands are emitted by this module. A PROTOTYPE_READY decision
is not a physical safety certification until valid dynamics & QP are integrated.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

@dataclass(frozen=True)
class CollisionWitness:
    collider0: str
    collider1: str
    signed_distance_m: float
    dq_distance_gradient: Sequence[float]
    geometry_asset_sha_verified: bool
    backend_contact_stream_positive_calibrated: bool
    age_s: float

@dataclass(frozen=True)
class WholebodyState:
    q: Sequence[float]
    dq: Sequence[float]
    supporting_foot_normal_forces_N: Sequence[float]
    signed_contact: CollisionWitness | None
    max_joint_speed_rad_s: Sequence[float]
    physics_dt_s: float = .005

def check_prototype_readiness(obs:WholebodyState, safety_clearance_m=.06)->dict:
    issues=[]
    if not (len(obs.q)==len(obs.dq)==len(obs.max_joint_speed_rad_s)==29):
        issues.append("INVALID_29_DOF_SCHEMA")
    if any(not isfinite(float(x)) for values in
           (obs.q,obs.dq,obs.max_joint_speed_rad_s) for x in values):
        issues.append("NONFINITE_JOINT_DATA")
    if (len(obs.supporting_foot_normal_forces_N)!=2 or
        any(not isfinite(float(x)) or x<0 for x in obs.supporting_foot_normal_forces_N)):
        issues.append("INVALID_SUPPORT_SENSOR")
    elif max(obs.supporting_foot_normal_forces_N)<20:
        issues.append("NO_VERIFIED_SUPPORT")
    if not isfinite(obs.physics_dt_s) or abs(obs.physics_dt_s-.005)>1e-8:
        issues.append("PHYSICS_RATE_UNVERIFIED")
    if len(obs.dq)==len(obs.max_joint_speed_rad_s)==29 and any(
        abs(qdot)>0.9*lim for qdot,lim in zip(obs.dq,obs.max_joint_speed_rad_s)):
        issues.append("JOINT_QDOT_HARD_MARGIN")
    if obs.signed_contact is None:
        issues.append("MISSING_SHAPE_PAIR_WITNESS")
    else:
        w=obs.signed_contact
        if not w.geometry_asset_sha_verified:issues.append("UNVERIFIED_COLLIDER_GEOMETRY")
        if not w.backend_contact_stream_positive_calibrated:issues.append("UNCALIBRATED_NATIVE_CONTACT_BACKEND")
        if not w.collider0 or not w.collider1 or w.collider0==w.collider1:
            issues.append("INVALID_ACTOR_COLLIDER_PAIR")
        if len(w.dq_distance_gradient)!=29 or not all(
            isfinite(float(x)) for x in w.dq_distance_gradient):
            issues.append("MISSING_COLLISION_JACOBIAN")
        if not isfinite(w.age_s) or w.age_s>.0051 or w.age_s<0:
            issues.append("STALE_COLLISION_GEOMETRY")
        if not isfinite(w.signed_distance_m):
            issues.append("NONFINITE_CLEARANCE")
        elif len(obs.dq)==len(w.dq_distance_gradient)==29:
            estimated=w.signed_distance_m+sum(
                float(a)*float(b) for a,b in zip(obs.dq,w.dq_distance_gradient))*obs.physics_dt_s
            if estimated<safety_clearance_m:
                issues.append("PREDICTED_WRIST_HIP_CLEARANCE_VIOLATION")
    return {"prototype_ready":not issues,"reject_reasons":issues,
            "physical_safety_certified":False,
            "allowed_use":"OFFLINE_QP_CONSTRAINT_DESIGN_ONLY",
            "original_G1_motor_command_authorized":False}
