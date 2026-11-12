#!/usr/bin/env python3
"""Fail-closed 29DoF whole-body motor-command admission contract, no actuation.

A future human-derived planner may propose full-body q, dq, τ, but the physical
controller must not consume it until independent native contact geometry,
support, torque, and kinematic witnesses all exist at the measured timestamp.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class Limits29:
    q_lo: list[float]
    q_hi: list[float]
    max_qdot: list[float]
    max_qddot: list[float]
    max_abs_torque: list[float]
@dataclass(frozen=True)
class Witness:
    age_seconds: float
    native_pair_certified: bool
    signed_collision_clearance_m: float | None
    support_certified: bool
    stance_foot_normal_force_N: float | None
    stance_foot_slip_mps: float | None
    body_geometry_certified: bool
    provenance_sha256: str | None
def _array(v):
    a=np.asarray(v,dtype=float)
    if a.shape!=(29,) or not np.isfinite(a).all():
        raise ValueError("INVALID_ORIGINAL29_VECTOR")
    return a
def admit(q_now,dq_now,q_next,dq_next,tau_next,limits,witness,
          dt_s:float=.005,clearance_margin_m:float=.015,
          slip_max_mps:float=.035,stance_force_min_N:float=30.):
    """Returns a witnessed decision, never executes action or changes old policy."""
    problems=[]
    now=_array(q_now);vel=_array(dq_now)
    target=_array(q_next);dvel=_array(dq_next);torque=_array(tau_next)
    lo=_array(limits.q_lo);hi=_array(limits.q_hi)
    vmax=_array(limits.max_qdot);amax=_array(limits.max_qddot)
    tmax=_array(limits.max_abs_torque)
    if dt_s<=0 or not np.isfinite(dt_s):raise ValueError("INVALID_DT")
    if np.any(lo>=hi) or np.any(vmax<=0) or np.any(amax<=0) or np.any(tmax<=0):
        raise ValueError("INVALID_ACTUATOR_LIMITS")
    if np.any(target<lo) or np.any(target>hi):problems.append("JOINT_POSITION")
    if np.any(np.abs(dvel)>vmax):problems.append("JOINT_VELOCITY")
    if np.any(np.abs((dvel-vel)/dt_s)>amax):problems.append("JOINT_ACCELERATION")
    if np.any(np.abs(torque)>tmax):problems.append("JOINT_TORQUE")
    if np.any(np.abs(target-now-dvel*dt_s)>.02):
        problems.append("INCONSISTENT_INTEGRATION")
    # Missing witnesses are NOT treated as safe absence of collision.
    if not witness.native_pair_certified or not witness.body_geometry_certified:
        problems.append("UNCERTIFIED_ACTOR_COLLIDER_GEOMETRY")
    if witness.signed_collision_clearance_m is None or (
      not np.isfinite(witness.signed_collision_clearance_m) or witness.signed_collision_clearance_m<clearance_margin_m):
        problems.append("COLLISION_CLEARANCE")
    if not witness.support_certified:problems.append("UNCERTIFIED_SUPPORT")
    if witness.stance_foot_normal_force_N is None or (
      not np.isfinite(witness.stance_foot_normal_force_N) or witness.stance_foot_normal_force_N<stance_force_min_N):
        problems.append("INSUFFICIENT_STANCE_FORCE")
    if witness.stance_foot_slip_mps is None or (
      not np.isfinite(witness.stance_foot_slip_mps) or witness.stance_foot_slip_mps>slip_max_mps):
        problems.append("STANCE_SLIP")
    if not np.isfinite(witness.age_seconds) or not 0<=witness.age_seconds<=dt_s:
        problems.append("STALE_SENSOR_WITNESS")
    if not isinstance(witness.provenance_sha256,str) or len(witness.provenance_sha256)!=64:
        problems.append("MISSING_SOURCE_PROVENANCE")
    return {"admitted":not problems,"refusal_reasons":problems,
            "original_joints_checked":29,
            "motor_command_issued":False,
            "native_pair_not_assumed_from_tensor":True}
