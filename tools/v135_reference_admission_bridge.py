#!/usr/bin/env python3
"""Bridge synthetic 29DoF plans into existing fullbody safety contracts.

This module intentionally has no actuator, training or PhysX execution entry.
No candidate can be considered a genuine motor-safe kick without measurements.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from v107_motor_admission import Limits29, Witness
from v117_witnessed_horizon_admission import TemporalPlan
from v127_lookahead_safety import screen_offline_horizon
import v127_collider_proximity as geom
from v135_chain29_offline import Proposal, frozen


REQUIRED_RUNTIME_MEASUREMENTS = (
    "all29_calibrated_acceleration_limits",
    "all29_measured_inverse_dynamics_or_motor_torque",
    "all_2800_fresh_verified_native_physx_collider_witnesses",
    "all_2800_support_slip_and_stance_force_witnesses",
    "independent_current_29dof_state",
    "fully_synced_root_COM_and_ball_kinematics",
    "verified_high_skill_human_kick_source",
)


def inspect_reference_interface(
    proposal: Proposal, *,
    measured_torque: np.ndarray | None = None,
    calibrated_limits: Limits29 | None = None,
    physical_witnesses: list[Witness] | None = None,
    original_state: tuple[np.ndarray, np.ndarray] | None = None,
    urdf_path: Path | None = None,
    geometry_cfg: dict | None = None,
) -> dict:
    """Fail-closed unless all prerequisites supplied; never actuates.

    Supplying synthetic versions of these arguments in tests cannot certify
    physical provenance, actor/collider pairs or human force supervision.
    """
    if proposal.q.shape != (2800, 29) or proposal.dq.shape != (2800, 29):
        raise ValueError("MISSING_ORIGINAL29_COMPLETE_TRAJECTORY")
    if proposal.source != "ENGINEERING_SYNTHETIC_NOT_HUMAN":
        raise ValueError("UNVERIFIED_OR_MUTATED_REFERENCE_ORIGIN")
    missing = []
    if measured_torque is None:
        missing.append("MEASURED_TORQUE")
    if calibrated_limits is None:
        missing.append("CALIBRATED_ORIGINAL29_DYNAMIC_LIMITS")
    if physical_witnesses is None:
        missing.append("PHYSX_SUPPORT_COLLISION_WITNESSES")
    if original_state is None:
        missing.append("INDEPENDENT_PHYSICAL_CURRENT_STATE")
    if urdf_path is None or geometry_cfg is None:
        missing.append("SHA_LOCKED_ORIGINAL_RUNTIME_URDF_GEOMETRY")
    if missing:
        return {
            "status": "BLOCKED_MISSING_PHYSICAL_EVIDENCE",
            "missing_sources": sorted(missing),
            "required_runtime_measurements": list(REQUIRED_RUNTIME_MEASUREMENTS),
            "whole_horizon_steps": 2800,
            "contact_and_support_not_inferred_from_q": True,
            "command_torque_not_inferred_from_ddq": True,
            "motor_execution_authorized": False,
            "native_contact_pair_certified": False,
            "strict_kick_pass": False,
        }
    tau = np.asarray(measured_torque, dtype=float)
    if tau.shape != (2800, 29) or not np.isfinite(tau).all():
        raise ValueError("INVALID_29DOF_MEASURED_TORQUE")
    if len(physical_witnesses) != 2800 or len(original_state) != 2:
        raise ValueError("INCOMPLETE_200HZ_PHYSICAL_WITNESS")
    if not all(isinstance(x, Witness) for x in physical_witnesses):
        raise ValueError("UNTRUSTED_PHYSX_WITNESS_TYPE")
    q0, dq0 = (np.asarray(x, dtype=float) for x in original_state)
    if (q0.shape != (29,) or dq0.shape != (29,)
            or not np.isfinite(q0).all() or not np.isfinite(dq0).all()):
        raise ValueError("INVALID_INDEPENDENT_ORIGINAL29_STATE")
    frozen_cfg = frozen()
    if set(geometry_cfg["focus_collision_pairs"][0]) != set(
            frozen_cfg["geometry_pairs"][0]):
        raise ValueError("GEOMETRY_PAIR_MODEL_DRIFT")
    geom.require_source(Path(urdf_path), frozen_cfg["original_urdf_sha256"])
    plan = TemporalPlan(
        q_targets=proposal.q, dq_targets=proposal.dq,
        torque_targets=tau, phases=proposal.phases, dt_s=.005,
        source_kind=proposal.source,
    )
    result = screen_offline_horizon(
        plan, initial_q=q0, initial_dq=dq0,
        limits=calibrated_limits, witnesses=physical_witnesses,
        urdf_path=Path(urdf_path), cfg=geometry_cfg)
    return dict(
        status="OFFLINE_REVIEW_ONLY",
        screen=result,
        source_is_not_verified_human_high_performance_trajectory=True,
        trusted_native_runtime_source_attestation_missing=True,
        original_29dof_motor_execution_authorized=False,
        motor_execution_authorized=False,
        strict_kick_pass=False,
    )
