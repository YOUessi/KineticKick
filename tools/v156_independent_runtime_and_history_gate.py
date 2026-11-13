#!/usr/bin/env python3
"""v156 original v54 real 200Hz SHA history admission; no invented base state.

No old logger records root quaternion or full 6D base velocity, so historical
q29-only trajectories cannot supply true 35D M/h/J6 at the observed pose.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from v146_real_200hz_coupled_audit import (
    verified_sources, split_real_5ms_rows, verify_g1_metadata,
)
from v154_cpu_feasibility_campaign import qualified_history
from v156_real_urdf_pin_model import frozen

REQUIRED_RUNTIME = (
    "root_quat_wxyz",
    "root_lin_vel_world",
    "root_ang_vel_world",
    "same_step_source_sha256",
)
ACTUAL_EARLY_TERMINAL = 5.140
HISTORY_NOMINAL_WINDOW = (1.170, 2.820)

def validate_state_admission(row):
    if not isinstance(row, dict):
        raise ValueError("V156_NOT_RUNTIME_STATE")
    for field in REQUIRED_RUNTIME:
        if field not in row:
            raise ValueError("V156_MISSING_REAL_6D_ROOT:" + field)
    if "joint_pos" not in row or "joint_vel" not in row:
        raise ValueError("V156_MISSING_ORIGINAL_29")
    if len(row["joint_pos"])!=29 or len(row["joint_vel"])!=29:
        raise ValueError("V156_INCOMPLETE_ORIGINAL_29")
    q=np.asarray(row["root_quat_wxyz"],dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-8:
        raise ValueError("V156_BAD_RUNTIME_QUATERNION")
    for name in ("root_lin_vel_world","root_ang_vel_world"):
        x=np.asarray(row[name],dtype=float)
        if x.shape!=(3,) or not np.isfinite(x).all():
            raise ValueError("V156_BAD_WORLD_BASE_VELOCITY")
    # A caller must authenticate same-step tensors and model/scene/physx provenance.
    if not isinstance(row["same_step_source_sha256"], str) or len(row["same_step_source_sha256"])!=64:
        raise ValueError("V156_INVALID_SAME_STEP_SOURCE_SHA")
    # Presence of a 64-char field is not a verified signed PhysX provenance chain.
    # This gate has no trusted live scene callback or contact-report verifier.
    raise ValueError("V156_SCHEMA_ONLY_NO_NATIVE_RUNTIME_VERIFICATION")

def independent_real_history_audit():
    cfg=frozen()
    sources=verified_sources()
    expected=qualified_history()
    results={}
    verified_total=0
    window_total=0
    early_total=0
    for name,path in sources.items():
        raw=path.read_bytes()
        data=json.loads(raw)
        vel_lim,tau_lim=verify_g1_metadata(data)
        rows,terminal=split_real_5ms_rows(data["traces"][0],expected[name]["true_historical_200hz_rows"])
        missed={}
        for row in rows:
            try:
                validate_state_admission(row)
            except ValueError as exc:
                missed[str(exc)]=missed.get(str(exc),0)+1
            else:
                raise AssertionError("V156_HISTORICAL_SUDDENLY_RUNTIME_COMPLETE")
        sample=[row for row in rows if HISTORY_NOMINAL_WINDOW[0]-1e-9<=row["time_s"]<=HISTORY_NOMINAL_WINDOW[1]+1e-9]
        if len(sample)!=331:
            raise ValueError("V156_HISTORICAL_WINDOW_COUNT_CHANGED")
        if any("root_quat_wxyz" in row or "root_ang_vel_world" in row for row in sample):
            raise ValueError("V156_HISTORICAL_SCHEMA_CHANGED_REASSESS")
        q=np.asarray([r["joint_pos"] for r in sample])
        v=np.asarray([r["joint_vel"] for r in sample])
        if q.shape!=(331,29) or v.shape!=(331,29):
            raise ValueError("V156_ACTUAL_29_STATES_INCOMPLETE")
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("V156_NONFINITE_HISTORICAL_JOINT")
        early=0
        if name=="cmu_10_05":
            early=sum(4.740-1e-9 <= r["time_s"] <= ACTUAL_EARLY_TERMINAL+1e-9 for r in rows)
            if early!=81 or len(terminal)!=1:
                raise ValueError("V156_ABORTED_TRACE_INVALID")
        elif len(terminal):
            raise ValueError("V156_UNEXPECTED_TERMINAL")
        results[name]={
            "source_sha256":hashlib.sha256(raw).hexdigest(),
            "real_200hz_rows":len(rows),
            "real_29dof_nominal_window_rows":len(sample),
            "real_29dof_joint_velocity_ratio_max":float(np.max(np.abs(v)/np.asarray(vel_lim))),
            "real_q29_window_span_max_rad":float(np.ptp(q,axis=0).max()),
            "reported_foot_contact_not_ground_only_grf":True,
            "admission_rejection_counts":missed,
            "admissible_full_35d_observed_runtime_steps":0,
            "early_terminal_400ms_rows":early,
            "native_collider_manifold_truth":"UNKNOWN",
            "no_unobserved_terminal_recovery_filled":True,
        }
        verified_total+=len(rows)
        window_total+=len(sample)
        early_total+=early
    if (verified_total,window_total,early_total)!=(cfg["history_states"],
          cfg["history_nominal_window_states"],cfg["early_terminal_states"]):
        raise ValueError("V156_ORIGINAL_REAL_HISTORY_DENOMINATOR_DRIFT")
    return {
        "actual_historical_physx_episodes":len(results),
        "actual_original_200hz_states":verified_total,
        "original_29dof_event_window_states":window_total,
        "early_terminal_real_steps":early_total,
        "new_physx_episodes":0,
        "valid_35d_historical_pin_reconstruction_steps":0,
        "why_35d_runtime_reconstruction_rejected":"root quaternion and six-dimensional base velocity absent",
        "real_foot_ball_contact_manifold":"UNKNOWN",
        "model_is_not_engine_dynamics":True,
        "elite_human_motion_training_qualified":False,
        "runs":results,
    }
