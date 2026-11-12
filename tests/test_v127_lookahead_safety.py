#!/usr/bin/env python3
"""v127 full-horizon motor veto: geometry never acts as PhysX pair certification."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))

import v107_motor_admission as motor
import v117_witnessed_horizon_admission as temporal
import v127_collider_proximity as geom
import v127_lookahead_safety as guard


def artifacts(*,source="UNQUALIFIED_CANDIDATE",native=False):
    cfg=geom.frozen()
    n=7
    phases=temporal.PHASES
    plan=temporal.TemporalPlan(
        q_targets=np.zeros((n,29)),dq_targets=np.zeros((n,29)),
        torque_targets=np.zeros((n,29)),phases=phases,
        source_kind=source)
    limits=motor.Limits29([-10.]*29,[10.]*29,[40.]*29,
                           [300.]*29,[150.]*29)
    w=motor.Witness(age_seconds=0,native_pair_certified=native,
                    signed_collision_clearance_m=.2 if native else None,
                    support_certified=native,stance_foot_normal_force_N=150. if native else None,
                    stance_foot_slip_mps=.001 if native else None,
                    body_geometry_certified=native,provenance_sha256="a"*64)
    path=Path("/home/you/projects/kinetickick-wt-v116/configs/paid_original/paid_main_resolved.urdf")
    return dict(plan=plan,initial_q=np.zeros(29),initial_dq=np.zeros(29),
                limits=limits,witnesses=[w]*n,urdf_path=path,cfg=cfg)


def test_missing_real_native_witness_refuses_all_seven_phases():
    r=guard.screen_offline_horizon(**artifacts())
    assert r["frames_screened"]==7
    assert r["collision_pair_count"]==4
    assert not r["nominal_original29_temporal_precheck_pass"]
    assert "NATIVE_ACTOR_COLLIDER_PAIR_UNCERTIFIED_BY_OFFLINE_MODEL" in r["total_refusal_reasons"]
    assert not r["motor_execution_authorized"]


def test_even_synthetic_forged_valid_witness_cannot_authorize_motor():
    r=guard.screen_offline_horizon(**artifacts(
        source="QUALIFIED_REAL_HUMAN_WHOLEBODY_WITH_IMPACT",native=True))
    assert r["nominal_original29_temporal_precheck_pass"]
    assert not r["native_physx_pair_certified"]
    assert not r["motor_execution_authorized"]
    assert not r["physical_motor_command_issued"]


def test_joint_qdot_violation_never_dropped():
    kw=artifacts(source="QUALIFIED_REAL_HUMAN_WHOLEBODY_WITH_IMPACT",native=True)
    qdot=kw["plan"].dq_targets.copy()
    qdot[3,0]=200.
    kw["plan"]=replace(kw["plan"],dq_targets=qdot)
    r=guard.screen_offline_horizon(**kw)
    assert not r["nominal_original29_temporal_precheck_pass"]
    assert any("JOINT_VELOCITY" in d["reasons"]
               for d in r["phase_and_original_29_joint_refusals"])


def test_phase_missing_rejected_but_no_motor():
    kw=artifacts(source="QUALIFIED_REAL_HUMAN_WHOLEBODY_WITH_IMPACT",native=True)
    kw["plan"]=replace(kw["plan"],phases=tuple(["approach"]*7))
    r=guard.screen_offline_horizon(**kw)
    assert "INCOMPLETE_FULLBODY_COORDINATION_PHASE_SEQUENCE" in r["total_refusal_reasons"]
    assert not r["motor_execution_authorized"]


def test_mutated_urdf_source_hash_rejected(tmp_path):
    kw=artifacts()
    fake=tmp_path/"fake.urdf"
    fake.write_text("<robot name='fake'/>")
    kw["urdf_path"]=fake
    with pytest.raises(ValueError,match="SHA256_DRIFT"):
        guard.screen_offline_horizon(**kw)


def test_frozen_2cm_threshold_must_remain_untuned():
    kw=artifacts()
    kw["cfg"]["capsule_lower_bound_thresholds_m"]=[0.,.05]
    with pytest.raises(ValueError,match="FROZEN_GEOMETRY_GATE_DRIFT"):
        guard.screen_offline_horizon(**kw)


def test_invalid_29dof_plan_fails_closed():
    kw=artifacts()
    kw["plan"]=replace(kw["plan"],q_targets=np.zeros((7,28)))
    with pytest.raises(ValueError,match="INVALID_FINITE_TRAJECTORY_Nx29"):
        guard.screen_offline_horizon(**kw)
