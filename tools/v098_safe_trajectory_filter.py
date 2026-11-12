#!/usr/bin/env python3
"""200Hz full-29 action horizon projection: offline feasibility only, NEVER motor output.

The 50Hz learned future is interpreted as a proposed velocity HOLD over four
real 200Hz increments. Actual measured q(0), dq(0), URDF angle bounds,
source-specific 29 speed bounds, and frozen acceleration caps constrain every
increment. This is an actuator-authority *denial layer*, not a complete
contact/torque/GRF safety controller or human expert imitation system.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
N=29
K=20
SUB=4
DT=.005
@dataclass(frozen=True)
class MeasuredG1:
    q0: object
    dq0: object
    urdf_low: object
    urdf_high: object
    original_speed_limit: object
    conservative_accel_limit: object
    urdf_source_sha_verified: bool

def arr(x,shape):
    try:a=np.asarray(x,dtype=np.float64)
    except (TypeError,ValueError):return None
    return a if a.shape==shape and np.isfinite(a).all() else None

def project_full29(candidate_50hz_normalized, measured:MeasuredG1,margin_rad=.003):
    """Project 20×29 candidates into 80×29 200Hz joint trajectories.

    Fail closed on unsafe input or empty feasible intersections. Outputs
    intentionally exclude actuation functions, torque approval, and normal-force
    estimates; they CANNOT be passed to an original G1 controller as approved.
    """
    denied=["MISSING_NATIVE_AUTHENTICATED_COLLIDER_PAIR_AND_CONTACT_NORMAL",
            "MISSING_CONCURRENT_STANCE_JACOBIAN_FRICTION_AND_COP",
            "MISSING_INVERSE_DYNAMICS_TORQUE_EFFORT_LIMIT_CERTIFICATION",
            "MISSING_QUALIFIED_ELITE_HUMAN_MOTION_AND_REAL_IMPULSE_LABEL"]
    def fail(reason):
        return dict(offline_joint_kinematics_feasible=False,
                    physical_actuation_authorized=False,rejection=reason,
                    required_qualifications=denied, q_200hz=None,dq_200hz=None)
    if not measured.urdf_source_sha_verified:return fail("UNQUALIFIED_SOURCE_URDF_SHA")
    want=arr(candidate_50hz_normalized,(K,N))
    q=arr(measured.q0,(N,))
    v=arr(measured.dq0,(N,))
    lo=arr(measured.urdf_low,(N,))
    hi=arr(measured.urdf_high,(N,))
    speed=arr(measured.original_speed_limit,(N,))
    acc=arr(measured.conservative_accel_limit,(N,))
    if any(a is None for a in (want,q,v,lo,hi,speed,acc)):
        return fail("INVALID_FULL29_MEASUREMENT_OR_CANDIDATE")
    if (not np.isfinite(margin_rad) or margin_rad<0 or
        np.any(speed<=0) or np.any(acc<=0) or
        np.any(lo+2*margin_rad>=hi)):
        return fail("INVALID_URDF_ANGLE_SPEED_OR_ACCEL_BOUNDS")
    if np.any(q<lo+margin_rad-1e-9) or np.any(q>hi-margin_rad+1e-9):
        return fail("ACTUAL_Q0_OUTSIDE_CONSERVATIVE_URDF_BOUNDS")
    if np.any(np.abs(v)>speed+1e-7):
        return fail("ACTUAL_DQ0_EXCEEDS_SOURCE_SPEED_BOUNDS")
    desired=want*speed[None,:]
    qout=[];vout=[];changed=0;total=K*SUB*N
    for k in range(K):
        for _ in range(SUB):
            lower=np.maximum.reduce([-speed,v-acc*DT,(lo+margin_rad-q)/DT])
            upper=np.minimum.reduce([speed,v+acc*DT,(hi-margin_rad-q)/DT])
            if np.any(lower>upper+1e-8):
                return fail("UNSTOPPABLE_CURRENT_MOTION_OR_EMPTY_JOINT_INTERSECTION")
            # numerical tolerance from a saturated boundary
            lower=np.minimum(lower,upper)
            next_v=np.clip(desired[k],lower,upper)
            changed+=int(np.count_nonzero(np.abs(next_v-desired[k])>1e-7))
            q=q+DT*next_v
            if np.any(q<lo+margin_rad-1e-8) or np.any(q>hi-margin_rad+1e-8):
                return fail("POST_INTEGRATION_URDF_BOUNDS_FAILED")
            if np.any(np.abs(next_v-v)>DT*acc+1e-7):
                return fail("POST_INTEGRATION_ACCEL_FAILED")
            v=next_v
            qout.append(q.copy());vout.append(v.copy())
    result=dict(offline_joint_kinematics_feasible=True,
        physical_actuation_authorized=False,
        q_200hz=np.asarray(qout).tolist(),dq_200hz=np.asarray(vout).tolist(),
        initial_q0_sha_check_only=True,
        real_g1_29_joint_speed_limit_source=True,
        projected_200hz_frames=K*SUB,dt_s=DT,
        interventions=int(changed),total_joint_increments=int(total),
        intervention_fraction=float(changed/total),
        maximum_urdf_margin_excess_rad=float(max(
            np.max(lo[None,:]+margin_rad-np.asarray(qout)),
            np.max(np.asarray(qout)-hi[None,:]+margin_rad),0)),
        required_qualifications=denied,
        conclusion="OFFLINE_JOINT_KINEMATICS_ONLY_NOT_CERTIFIED_FULLBODY_POLICY")
    return result
