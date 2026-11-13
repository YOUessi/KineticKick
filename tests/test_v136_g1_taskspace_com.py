"""v136 preregistered CPU-only original G1 fullbody support/taskspace audit."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import v136_g1_taskspace_mass_com as m


@pytest.fixture(scope="module")
def state():
    cfg = m.frozen()
    m.require_original(m.URDF, cfg)
    tree = m.fk.parse_tree(m.URDF)
    lim = m.chain.source_limits(m.URDF, cfg["urdf_sha256"])
    q0 = (lim["lo"] + lim["hi"])/2
    return cfg, tree, lim, q0


def test_protocol_lock_is_a_prior_commit_with_exact_parent():
    cfg=m.frozen()
    assert subprocess.check_output([
        "git","-C",str(ROOT),"rev-parse",m.FROZEN_SHA+"^"
    ],text=True).strip()==cfg["parent_sha"]
    assert cfg["no_kick_success_claim"] and cfg["no_human_expert_force_truth"]
    assert cfg["strict_kick_gates"]["instep_impulse_fraction_min"] == .95
    assert cfg["swing_link_origin_velocity_targets_mps"] == [0,.5,1,3]


def test_frozen_protocol_file_drift_is_rejected(tmp_path,monkeypatch):
    bad=tmp_path/"protocol.json"
    obj=json.loads(m.CONFIG.read_text())
    obj["swing_link_origin_velocity_targets_mps"]=[0,1,2,10]
    bad.write_text(json.dumps(obj))
    monkeypatch.setattr(m,"CONFIG",bad)
    with pytest.raises(ValueError,match="PROTOCOL_DRIFT"):
        m.frozen()


def test_original_urdf_sha_mutation_fails_before_any_kinematics(tmp_path,state):
    cfg,*_=state
    fake=tmp_path/"tampered.urdf"
    fake.write_bytes(m.URDF.read_bytes()+b"\n<!--changed-->\n")
    with pytest.raises(ValueError,match="SOURCE_SHA256_DRIFT"):
        m.require_original(fake,cfg)


def test_all_mass_inertials_accounted_without_synthetic_grf(state):
    _,tree,_,_=state
    masses=m.mass_inertials(m.URDF,tree)
    assert len(masses)==37
    assert sum(x[1] for x in masses)==pytest.approx(33.341142022)
    assert len(set(x[0] for x in masses))==len(masses)
    assert all(row[1]>0 for row in masses)


def test_mass_weighted_com_translation_equivariance(state):
    _,tree,_,q0=state
    masses=m.mass_inertials(m.URDF,tree)
    names={row[0] for row in masses}
    poses=m.fk.fk_targets(tree,q0[None,:],names)
    baseline=m.mass_weighted_com(poses,masses)
    v=np.array([1.25,-.4,.3])
    translated={name:(R,p+v) for name,(R,p) in poses.items()}
    shifted=m.mass_weighted_com(translated,masses)
    assert np.allclose(shifted-baseline,v[None,:],atol=1e-11)


@pytest.mark.parametrize("link",["left_ankle_roll_link",
                                 "right_ankle_roll_link",
                                 "left_wrist_yaw_link"])
def test_independent_fd_jacobian_of_original_29dof(state,link):
    c,tree,_,q0=state
    point,J=m.point_jacobian(tree,q0,link)
    assert point.shape==(3,) and J.shape==(3,29)
    finite=m.fd_jacobian(tree,q0,link,c["jacobian_fd_step_rad"])
    assert np.max(abs(finite-J)) < c["jacobian_error_tolerance"]
    # Genuine kinematic locality: a shoulder cannot translate the ankle.
    if "ankle" in link:
        arms=m._arm_indices()
        assert np.max(abs(J[:,arms]))<1e-12


def test_zero_and_invalid_jacobian_input_rejected(state):
    _,tree,_,q0=state
    with pytest.raises(ValueError,match="INVALID_Q_ORIGINAL29"):
        m.point_jacobian(tree,q0[:-1],"left_ankle_roll_link")
    with pytest.raises(ValueError,match="UNKNOWN_KINEMATIC_LINK"):
        m.point_jacobian(tree,q0,"made_up_foot")
    with pytest.raises(ValueError,match="INVALID_FINITE_DIFFERENCE_STEP"):
        m.fd_jacobian(tree,q0,"right_ankle_roll_link",0)


@pytest.mark.parametrize("scale",[0,1,2,4,-1])
def test_all_five_frozen_14s_mass_and_support_hypotheses(state,scale):
    cfg,tree,lim,q0=state
    p=m.chain.generate(m.chain.frozen(),q0,scale)
    result=m.coupled_support_audit(
        p,tree,m.mass_inertials(m.URDF,tree),cfg)
    assert result["frame_count"]==2800
    assert result["original_urdf_inertial_link_count"]==37
    assert result["original_urdf_total_inertial_mass_kg"]==pytest.approx(33.341142022)
    assert result["only_kinematic_root_hypotheses"]
    assert result["actual_base_support_grf_cop_and_contact_unknown"]
    assert not result["original29_motor_execution_authorized"]
    if scale==0:
        assert result["fixed_root_left_ankle_origin_travel_max_m"]==pytest.approx(0,abs=1e-12)
        assert result["at_2p95s_right_origin_relative_speed_mps"]==pytest.approx(0,abs=1e-12)
    else:
        assert result["fixed_root_left_ankle_origin_travel_max_m"]>0
        assert result["left_ankle_translation_locked_inertial_com_travel_max_m"]>0


@pytest.mark.parametrize("target",[0,.5,1,3])
def test_all_predeclared_targets_still_refuse_motor(state,target):
    cfg,tree,lim,q0=state
    p=m.chain.generate(m.chain.frozen(),q0,4)
    x=m.bounded_task_feasibility(p,lim,tree,cfg,target)
    assert x["original29_qdot_limit_violation_count"]==0
    assert x["max_analytic_vs_fd_jacobian_abs_error"]<cfg["jacobian_error_tolerance"]
    assert np.isfinite(x["swing_vector_residual_mps"])
    assert np.isfinite(x["instantaneous_qddot_if_applied_in_5ms_max_rad_s2"])
    assert not x["physics_success_claim"]
    assert not x["native_collision_contact_certified"]
    assert not x["motor_execution_authorized"]


def test_not_frozen_target_and_poisoned_limits_fail(state):
    cfg,tree,lim,q0=state
    p=m.chain.generate(m.chain.frozen(),q0,1)
    with pytest.raises(ValueError,match="UNREGISTERED_TASK_TARGET"):
        m.bounded_task_feasibility(p,lim,tree,cfg,10)
    bad=dict(lim)
    bad["vmax"]=np.zeros(29)
    with pytest.raises(ValueError,match="BAD_ORIGINAL29_VELOCITY_LIMITS"):
        m.bounded_task_feasibility(p,bad,tree,cfg,1)


def test_proposal_lookahead_missing_samples_rejected(state):
    cfg,tree,lim,q0=state
    p=m.chain.generate(m.chain.frozen(),q0,1)
    bad=replace(p,q=p.q[:-5])
    with pytest.raises(ValueError,match="NOT_14S_ORIGINAL29"):
        m.coupled_support_audit(bad,tree,m.mass_inertials(m.URDF,tree),cfg)


@pytest.fixture(scope="module")
def batch():
    return m.run(m.URDF)


def test_entire_grid_is_5x4_and_fully_unauthorized(batch):
    assert batch["original29_14s_200hz_reference_frames"]==14000
    assert len(batch["support_kinematics_rows"])==5
    assert len(batch["frozen_taskspace_rows"])==20
    assert batch["new_true_physx_rollouts"]==0
    assert batch["actual_motor_commands_issued"]==0
    assert batch["foot_link_origin_is_not_true_instep"]
    assert batch["base_translation_is_not_real_stance_support"]
    assert not batch["human_kicking_grf_training_data_claim"]
    assert all(not r["motor_execution_authorized"]
               for r in batch["frozen_taskspace_rows"])


def test_all_nonzero_target_footspeed_warning_is_negative_not_success(batch):
    rows=batch["frozen_taskspace_rows"]
    for r in rows:
        if r["target_ankle_origin_root_plus_x_mps"]>0:
            # Negative result, frozen warning threshold. Don't tune retroactively.
            assert r["speed_task_warning"]
            assert r["swing_vector_residual_mps"] > .2
    assert all(r["original29_qdot_limit_violation_count"]==0 for r in rows)


def test_synthetic_zero_motion_is_positive_control_for_mass_invariance(batch):
    r=next(r for r in batch["support_kinematics_rows"] if r["multiplier"]==0)
    assert r["fixed_root_inertial_com_travel_max_m"]==pytest.approx(0,abs=1e-12)
    assert r["right_ankle_origin_relative_to_left_travel_max_m"]==pytest.approx(0,abs=1e-12)
