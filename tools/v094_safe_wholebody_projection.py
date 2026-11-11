#!/usr/bin/env python3
"""Offline-only, hard-constrained 29DoF candidate velocity filter for future learned motion.

No dispatch to original G1 motors, no live action: a 'feasible' QP here is not
an independently certified inverse-dynamics/contact-safe physical controller.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

N=29
@dataclass(frozen=True)
class CollisionShapeCertificate:
 collider0: str
 collider1: str
 signed_distance_m: float
 distance_gradient_dq: Sequence[float]
 geometry_sha_verified: bool
 native_contact_channel_positive_control_calibrated: bool
 age_s: float

@dataclass(frozen=True)
class CoordinatedVelocityRequest:
 desired_all29_joint_velocity_rad_s: Sequence[float]
 measured_all29_joint_velocity_rad_s: Sequence[float]
 all29_joint_speed_limits_rad_s: Sequence[float]
 all29_accel_limits_rad_s2: Sequence[float]
 measured_foot_normal_loads_N: Sequence[float]
 # 3x29 support-sole contact Jacobian at the *current* measured q, per foot.
 left_support_linear_J: Sequence[Sequence[float]]
 right_support_linear_J: Sequence[Sequence[float]]
 shape: CollisionShapeCertificate | None
 dt_s: float=.005

def _check_finite(a,shape,positive=False):
 try: arr=np.asarray(a,dtype=np.float64)
 except (ValueError,TypeError):return None
 if arr.shape!=shape or not np.isfinite(arr).all() or (positive and np.any(arr<=0)):
  return None
 return arr

def filter_offline_candidate(r:CoordinatedVelocityRequest,
   min_sole_normal_N=80.0, max_sole_slip_mps=.03,
   clearance_m=.06) -> dict:
 """Minimize ||v-desired||² over true 29 joint speed/accel and stance/collision
 halfspaces. Never emits motors: independent torque, native pair and GRF checks
 are required *again* on the original G1 before future physical deployment.
 """
 reject=[]
 target=_check_finite(r.desired_all29_joint_velocity_rad_s,(N,))
 old=_check_finite(r.measured_all29_joint_velocity_rad_s,(N,))
 lim=_check_finite(r.all29_joint_speed_limits_rad_s,(N,),True)
 accel=_check_finite(r.all29_accel_limits_rad_s2,(N,),True)
 loads=_check_finite(r.measured_foot_normal_loads_N,(2,))
 JL=_check_finite(r.left_support_linear_J,(3,N))
 JR=_check_finite(r.right_support_linear_J,(3,N))
 if any(x is None for x in (target,old,lim,accel,loads,JL,JR)):
  reject.append("INVALID_MEASURED_29DOF_OR_STANCE_JACOBIAN")
 if abs(r.dt_s-.005)>1e-9 or not np.isfinite(r.dt_s):
  reject.append("REQUIRES_200HZ_MEASURED_STATE")
 if loads is not None and (np.any(loads<0) or max(loads)<min_sole_normal_N):
  reject.append("NO_CONFIRMED_PLANTED_SUPPORT")
 shape=r.shape
 grad=None
 if shape is None:
  reject.append("MISSING_NATIVE_SHAPE_DISTANCE_CERTIFICATE")
 else:
  grad=_check_finite(shape.distance_gradient_dq,(N,))
  if not shape.geometry_sha_verified:reject.append("GEOMETRY_SOURCE_SHA_UNVERIFIED")
  if not shape.native_contact_channel_positive_control_calibrated:
   reject.append("NATIVE_PAIR_STREAM_UNCALIBRATED")
  if not shape.collider0 or not shape.collider1 or shape.collider0==shape.collider1:
   reject.append("INVALID_COLLIDER_PAIR")
  if grad is None or not np.isfinite(shape.signed_distance_m):
   reject.append("INVALID_SIGNED_COLLISION_DISTANCE_JACOBIAN")
  if not np.isfinite(shape.age_s) or shape.age_s<0 or shape.age_s>r.dt_s+1e-9:
   reject.append("STALE_COLLIDER_GEOMETRY")
 if reject:
  return dict(offline_projection_feasible=False,
   physical_actuation_authorized=False,
   candidate_all29_rad_s=None,reasons=reject,
   status="FAIL_CLOSED_UNQUALIFIED_SENSORS_OR_NATIVE_CONTACTS")
 from scipy.optimize import minimize
 low=np.maximum(-lim,old-accel*r.dt_s)
 high=np.minimum(lim,old+accel*r.dt_s)
 if np.any(low>high):
  return dict(offline_projection_feasible=False,physical_actuation_authorized=False,
   candidate_all29_rad_s=None,reasons=["EMPTY_SPEED_ACCEL_INTERSECTION"])
 constraints=[]
 # Linear inequalities are relative to the measured, contemporaneous state.
 for load,J in zip(loads,(JL,JR)):
  if load>=min_sole_normal_N:
   for axis in range(3):
    g=J[axis].copy()
    constraints.append({"type":"ineq","fun":lambda v,g=g:float(max_sole_slip_mps-g@v),
                       "jac":lambda v,g=g:-g})
    constraints.append({"type":"ineq","fun":lambda v,g=g:float(max_sole_slip_mps+g@v),
                       "jac":lambda v,g=g:g})
 constraints.append({"type":"ineq","fun":lambda v:float(shape.signed_distance_m+
             r.dt_s*(grad@v)-clearance_m),"jac":lambda v:r.dt_s*grad})
 init=np.clip(target,low,high)
 res=minimize(lambda v:float(np.sum((v-target)**2)),
     init,jac=lambda v:2*(v-target),bounds=list(zip(low,high)),
     constraints=constraints,method="SLSQP",options={"ftol":1e-10,"maxiter":100})
 if not res.success or not np.isfinite(res.x).all():
  return dict(offline_projection_feasible=False,physical_actuation_authorized=False,
   candidate_all29_rad_s=None,reasons=["QP_INFEASIBLE_OR_SOLVER_DID_NOT_CONVERGE"])
 v=np.asarray(res.x)
 if (np.any(v<low-1e-7) or np.any(v>high+1e-7)
  or shape.signed_distance_m+r.dt_s*(grad@v)<clearance_m-1e-7
  or any(np.max(np.abs(J@v))>max_sole_slip_mps+1e-7
         for load,J in zip(loads,(JL,JR)) if load>=min_sole_normal_N)):
  return dict(offline_projection_feasible=False,physical_actuation_authorized=False,
   candidate_all29_rad_s=None,reasons=["POST_QP_HARD_CONSTRAINT_RECHECK_FAILED"])
 return dict(offline_projection_feasible=True,physical_actuation_authorized=False,
  candidate_all29_rad_s=v.tolist(),
  max_measured_support_slip_proxy_mps=float(max(np.max(np.abs(J@v))
    for load,J in zip(loads,(JL,JR)) if load>=min_sole_normal_N)),
  next_step_clearance_linearized_m=float(shape.signed_distance_m+r.dt_s*grad@v),
  reasons=["NO_VERIFIED_INVERSE_DYNAMICS_TORQUE_OR_REAL_HUMAN_MOTION_CERTIFICATION"],
  status="OFFLINE_FEASIBLE_NOT_AUTHORIZED_FOR_ROBOT_ACTUATION")
