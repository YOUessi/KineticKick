#!/usr/bin/env python3
"""KineticChain-29 offline seven-phase finite-horizon safety contract.

Pure read-only admission check; no motor output, no claims of native collision
certification without measured actor/collider and support/slip witnesses.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from v107_motor_admission import Limits29, Witness, admit

PHASES=("approach","stance","backswing","hip_trunk_accel",
        "instep_contact","follow_through","landing_recovery")


@dataclass(frozen=True)
class TemporalPlan:
    q_targets: np.ndarray
    dq_targets: np.ndarray
    torque_targets: np.ndarray
    phases: tuple[str,...]
    dt_s: float = 0.005
    source_kind: str = "UNQUALIFIED_CANDIDATE"


def _matrix(a, n: int) -> np.ndarray:
    x=np.asarray(a,dtype=float)
    if x.shape!=(n,29) or not np.isfinite(x).all():
        raise ValueError("INVALID_FINITE_TRAJECTORY_Nx29")
    return x


def validate_phase_schedule(phases: tuple[str,...]) -> dict:
    if not phases or any(x not in PHASES for x in phases):
        raise ValueError("UNKNOWN_OR_EMPTY_KICK_PHASE")
    indices=[PHASES.index(x) for x in phases]
    if indices != sorted(indices):
        raise ValueError("NON_MONOTONIC_COORDINATION_PHASE")
    missing=[x for x in PHASES if x not in phases]
    return dict(monotonic=True,observed_phases=tuple(dict.fromkeys(phases)),
                whole_kick_sequence_complete=not missing, missing_phases=missing)


def examine_plan(plan: TemporalPlan, *,
                 initial_q: np.ndarray,initial_dq: np.ndarray,
                 limits: Limits29, witnesses: list[Witness]) -> dict:
    """Refuse entire plan on missing witness, unsafe timestep or phase omission."""
    n=len(plan.phases)
    if n<7 or not np.isfinite(plan.dt_s) or plan.dt_s != .005:
        raise ValueError("BAD_REAL_200HZ_PLAN_HORIZON")
    phase=validate_phase_schedule(plan.phases)
    q=_matrix(plan.q_targets,n)
    dq=_matrix(plan.dq_targets,n)
    tau=_matrix(plan.torque_targets,n)
    q0=np.asarray(initial_q,dtype=float)
    dq0=np.asarray(initial_dq,dtype=float)
    if q0.shape!=(29,) or dq0.shape!=(29,) or not (
        np.isfinite(q0).all() and np.isfinite(dq0).all()):
        raise ValueError("INVALID_START_29DOF")
    if len(witnesses)!=n:
        raise ValueError("MISSING_FRAMEWISE_WITNESSES")
    reasons=[]
    flagged_frames=[]
    for i in range(n):
        prior_q=q[i-1] if i else q0
        prior_vel=dq[i-1] if i else dq0
        result=admit(prior_q,prior_vel,q[i],dq[i],tau[i],
                     limits,witnesses[i],dt_s=plan.dt_s)
        if not result["admitted"]:
            flagged_frames.append(dict(
                frame=i,phase=plan.phases[i],
                reasons=result["refusal_reasons"]))
            reasons.extend(result["refusal_reasons"])
    if not phase["whole_kick_sequence_complete"]:
        reasons.append("INCOMPLETE_FULLBODY_COORDINATION_PHASE_SEQUENCE")
    # Even a mathematically valid vector is not an authorization to actuate.
    if plan.source_kind != "QUALIFIED_REAL_HUMAN_WHOLEBODY_WITH_IMPACT":
        reasons.append("UNQUALIFIED_HUMAN_ACTION_SOURCE")
    return dict(
        nominal_safety_screen_pass=not reasons,
        refusal_reasons=sorted(set(reasons)),
        frames_with_any_refusal=len(flagged_frames),
        framewise_refusals=flagged_frames,
        phase_evidence=phase,
        checked_original_joint_dimensions=29,
        physical_motor_command_issued=False,
        true_physx_contact_pair_witnesses_provided=bool(witnesses and all(
            w.native_pair_certified for w in witnesses)),
        human_grf_cop_or_true_instep_impulse_inferred=False,
    )
