#!/usr/bin/env python3
"""v127 offline 29DoF horizon collision-veto interface, NEVER a motor controller.

Complements v117 independently witnessed time-horizon admission: checks exact
original URDF model's conservative cylinder capsule separation each step.
A zero PhysX callback, nonpositive capsule gap, or fake witness never proves
a physical collision pair. No actuator call exists in this module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import v117_witnessed_horizon_admission as temporal
import v127_collider_proximity as geometry


def screen_offline_horizon(
    plan: temporal.TemporalPlan,
    *, initial_q, initial_dq, limits, witnesses,
    urdf_path: Path, cfg: dict,
):
    """Check all original 29DoF joints and all four frozen collision pairs.

    Returns diagnostics only; motor_execution_authorized is always False,
    irrespective of apparent witness quality. Native PhysX actor/collider
    ownership remains unverified by this offline geometry model.
    """
    geometry.require_source(Path(urdf_path),cfg["source_urdf_sha256"])
    base=temporal.examine_plan(
        plan,initial_q=np.asarray(initial_q),initial_dq=np.asarray(initial_dq),
        limits=limits,witnesses=witnesses)
    tree=geometry.parse_tree(Path(urdf_path))
    clearance=geometry.clearance_for_episode(
        tree,plan.q_targets,cfg["focus_collision_pairs"])
    threshold=cfg["capsule_lower_bound_thresholds_m"][1]
    if threshold!=.02 or cfg["native_physx_pair_certification_permitted"]:
        raise ValueError("FROZEN_GEOMETRY_GATE_DRIFT")
    risks={}
    for pair,values in clearance.items():
        first=np.flatnonzero(values<=threshold)
        risks[pair]={
            "min_capsule_lower_bound_m":float(values.min()),
            "risky_count":int(len(first)),
            "first_risk_index":int(first[0]) if len(first) else None,
            "risk_is_actuation_permission":False,
        }
    reasons=set(base["refusal_reasons"])
    if any(x["risky_count"] for x in risks.values()):
        reasons.add("URDF_CAPSULE_PROXIMITY_REQUIRES_NATIVE_PHYSX_VERIFICATION")
    # A static URDF collision geometry cannot certify a dynamic PhysX pair.
    reasons.add("NATIVE_ACTOR_COLLIDER_PAIR_UNCERTIFIED_BY_OFFLINE_MODEL")
    return {
        "nominal_original29_temporal_precheck_pass":
            bool(base["nominal_safety_screen_pass"]),
        "horizon_geometry_outside_frozen_2cm_zone":
            all(x["risky_count"]==0 for x in risks.values()),
        "frames_screened":int(len(plan.phases)),
        "collision_pair_count":len(risks),
        "phase_evidence":base["phase_evidence"],
        "phase_and_original_29_joint_refusals":base["framewise_refusals"],
        "URDF_capsule_proximity":risks,
        "total_refusal_reasons":sorted(reasons),
        "native_physx_pair_certified":False,
        "motor_execution_authorized":False,
        "physical_motor_command_issued":False,
        "human_high_performance_force_truth":False,
    }
