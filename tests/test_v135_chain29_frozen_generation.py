"""v135 preregistered offline 29DoF seven-phase engineering proposal negatives."""
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
import v135_chain29_offline as m

URDF = Path("/home/you/projects/kinetickick/configs/paid_original/paid_main_resolved.urdf")


@pytest.fixture(scope="module")
def setup():
    cfg = m.frozen()
    return cfg, m.source_limits(URDF, cfg["original_urdf_sha256"])


def test_protocol_already_committed_before_analysis(setup):
    cfg, _ = setup
    parent = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", m.FROZEN_SHA+"^"],
        text=True).strip()
    assert parent == cfg["parent_sha"]
    assert len(cfg["all_phases"]) == len(cfg["phase_durations_s"]) == 7
    assert sum(cfg["phase_durations_s"]) == 14
    assert cfg["multipliers"] == [0, 1, 2, 4, -1]


def test_all_29_joints_are_coupled_and_basis_rank_four():
    assert m.weight_matrix().shape == (29, 4)
    assert np.linalg.matrix_rank(m.weight_matrix()) == 4
    seen = [j for names in m.GROUPS.values()
            for j in m.G1_JOINT_ORDER if j.startswith(names)]
    assert len(seen) == len(set(seen)) == 29


@pytest.mark.parametrize("mult", [0.,1.,2.,4.,-1.])
def test_all_controls_2800_frames_zero_end_and_continuous(setup, mult):
    cfg, limits = setup
    p = m.generate(cfg, .5*(limits["lo"]+limits["hi"]), mult)
    assert p.q.shape == p.dq.shape == p.ddq.shape == (2800,29)
    assert p.latent.shape == (2800,4)
    assert np.isfinite(p.q).all()
    assert np.isclose(p.t[-1], 13.995)
    assert tuple(dict.fromkeys(p.phases)) == m.PHASES
    assert np.allclose(p.q[0], p.q[-1], atol=1e-7)
    assert np.max(abs(p.dq[[0,-1]])) < 1e-5
    assert not p.motor_execution_authorized
    assert not p.human_instep_contact_truth
    assert not p.native_pair_certified


def test_zero_motion_null_is_not_coupled_motion(setup):
    cfg, limits = setup
    q0 = .5*(limits["lo"]+limits["hi"])
    p = m.generate(cfg, q0, 0)
    assert np.all(p.q == q0)
    assert not np.any(p.dq)
    assert not np.any(p.ddq)


@pytest.mark.parametrize("mult", [1.,2.,4.,-1.])
def test_nonzero_controls_activate_both_arms_legs_and_waist(setup, mult):
    cfg, limits = setup
    p = m.generate(cfg, .5*(limits["lo"]+limits["hi"]), mult)
    for names in m.GROUPS.values():
        idx = [i for i, j in enumerate(m.G1_JOINT_ORDER)
               if j.startswith(names)]
        assert np.max(abs(p.q[:, idx]-p.q[0, idx])) > 1e-3


@pytest.mark.parametrize("bad", [3.,5.,-2.,float("nan"),True])
def test_unregistered_multiplier_is_never_accepted(setup,bad):
    cfg, limits = setup
    with pytest.raises(ValueError,match="UNFROZEN_MULTIPLIER"):
        m.generate(cfg, .5*(limits["lo"]+limits["hi"]),bad)


def test_invalid_nonfinite_and_bad_length_start_states(setup):
    cfg, _ = setup
    for q in (np.zeros(28), np.full(29,float("nan")),
              np.full(29,float("inf"))):
        with pytest.raises(ValueError,match="INVALID_SYNTHETIC_ORIGINAL_Q0"):
            m.generate(cfg,q,1.)


def test_wrong_urdf_sha_rejected_before_generation(setup,tmp_path):
    cfg,_=setup
    b=URDF.read_bytes()
    fake=tmp_path/"source.urdf"
    fake.write_bytes(b+b"\n")
    with pytest.raises(ValueError,match="SOURCE_SHA256_DRIFT"):
        m.source_limits(fake,cfg["original_urdf_sha256"])


def test_genuine_v134_physx_source_not_claimed_by_synthetic(setup):
    cfg,limits=setup
    p=m.generate(cfg,.5*(limits["lo"]+limits["hi"]),1.)
    assert p.source=="ENGINEERING_SYNTHETIC_NOT_HUMAN"
    assert p.torque_or_grf_available is False
    assert p.human_instep_contact_truth is False


def test_phase_junctions_are_stationary_in_velocity(setup):
    cfg,limits=setup
    p=m.generate(cfg,.5*(limits["lo"]+limits["hi"]),4.)
    i=0
    for duration in cfg["phase_durations_s"]:
        start=i
        i+=round(duration/.005)
        # Sample immediately before a boundary is not at u=1; its derivative
        # need only decay relative to that phase's peak, not equal zero.
        assert np.max(abs(p.dq[i-1])) < .5*np.max(abs(p.dq[start:i]))
        if i<2800: assert np.max(abs(p.dq[i])) < 1e-8


def test_actual_kinematic_corruption_rejected_before_geometry(setup):
    cfg,limits=setup
    p=m.generate(cfg,.5*(limits["lo"]+limits["hi"]),1.)
    tree=m.geom.parse_tree(URDF)
    pbad=replace(p,q=np.concatenate([p.q[:-1],np.zeros((1,29))],axis=0))
    # Still a finite 2800x29 candidate: only kinematic limits, not a real collision.
    r=m.inspect(pbad,limits,tree,cfg)
    assert not r["original29_motor_execution_authorized"]


def test_forged_physical_witness_flags_do_not_override_unauthorized(setup):
    cfg,limits=setup
    tree=m.geom.parse_tree(URDF)
    p=m.generate(cfg,.5*(limits["lo"]+limits["hi"]),1.)
    forged=replace(p, native_pair_certified=True,
                   human_instep_contact_truth=True,
                   torque_or_grf_available=True,
                   motor_execution_authorized=True)
    r=m.inspect(forged,limits,tree,cfg)
    assert r["original29_motor_execution_authorized"] is False
    assert r["full_kick_strict_physics_pass"] is False
    assert r["true_physx_rollouts"] == 0


def test_frozen_config_drift_rejected(monkeypatch,tmp_path):
    modified=tmp_path/"changed.json"
    c=json.loads(m.CFG.read_text())
    c["multipliers"]=[0,1,2,8,-1]
    modified.write_text(json.dumps(c))
    monkeypatch.setattr(m,"CFG",modified)
    with pytest.raises(ValueError,match="V135_PROTOCOL_DRIFT"):
        m.frozen()
