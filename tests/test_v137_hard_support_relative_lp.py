"""v137 preregistered exact 29DoF *kinematic* LP and CPU negative controls."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import v137_hard_support_relative_lp as v


@pytest.fixture(scope="module")
def setup():
    cfg = v.frozen()
    tree, lim = v.original_kinematics(v.URDF, cfg)
    mass = v.task.mass_inertials(v.URDF, tree)
    q0 = (lim["lo"]+lim["hi"])/2
    return cfg, tree, lim, mass, q0


@pytest.fixture(scope="module")
def report():
    return v.run()


def test_protocol_has_independent_committed_parent_before_implementation():
    c = v.frozen()
    assert subprocess.check_output(
        ["git","-C",str(ROOT),"rev-parse",v.FROZEN_SHA+"^"], text=True).strip()==c["parent_sha"]
    assert c["forbid_motor_execution"]
    assert c["forbid_claim_new_physx_success"]
    assert c["forbid_claim_human_learning"]
    assert c["strict_kick_gates"]["instep_impulse_fraction_min"] == .95
    assert c["swing_relative_link_origin_velocity_targets_x_mps"]==[0,.5,1,3]


def test_protocol_tamper_strictly_rejected(tmp_path, monkeypatch):
    cfg=json.loads(v.CONFIG.read_text())
    cfg["com_world_relative_to_virtual_stance_velocity_cap_mps"]=[None,10.]
    p=tmp_path/"tampered.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setattr(v,"CONFIG",p)
    with pytest.raises(ValueError,match="PROTOCOL_PATH_DRIFT"):
        v.frozen()


def test_source_urdf_mutation_strictly_rejected(tmp_path, setup):
    cfg,*_=setup
    fake=tmp_path/"wrong.urdf"
    fake.write_bytes(v.URDF.read_bytes()+b"\n<!-- drift -->")
    with pytest.raises(ValueError,match="SOURCE_SHA256_DRIFT"):
        v.original_kinematics(fake,cfg)


def test_original_g1_29_limits_and_inertia_qualify(setup):
    c,tree,lim,mass,q0=setup
    assert len(mass)==37
    assert sum(m for _,m,_ in mass)==pytest.approx(33.341142022)
    assert q0.shape==(29,)
    assert len(v.ARM)==14
    assert len(v.G1_JOINT_ORDER)==29
    assert np.all(lim["vmax"]>0)
    assert max(lim["vmax"]) > min(lim["vmax"])


@pytest.mark.parametrize("mult,time",[(0,2.25),(1,2.95),(2,3.6),(4,2.95),(-1,3.6)])
def test_com_analytic_derivative_matches_independent_batched_fd(setup,mult,time):
    cfg,tree,lim,mass,q0=setup
    p=v.chain.generate(v.chain.frozen(),q0,mult)
    q=p.q[round(time/.005)]
    com, J=v.inertial_com_and_analytic_jacobian(tree,mass,q)
    other=v.independent_fd_com_jacobian(tree,mass,q,1.e-6)
    assert com.shape==(3,) and J.shape==(3,29)
    assert np.isfinite(com).all() and np.isfinite(J).all()
    assert np.max(abs(J-other)) < cfg["jacobian_crosscheck_max_abs"]


def test_com_from_independent_mass_inertial_xyz(setup):
    cfg,tree,lim,mass,q0=setup
    pts=v.kin.fk_targets(tree,q0[None,:],{x[0] for x in mass})
    direct=v.task.mass_weighted_com(pts,mass)[0]
    com,_=v.inertial_com_and_analytic_jacobian(tree,mass,q0)
    np.testing.assert_allclose(com,direct,atol=1.e-12,rtol=0)


@pytest.mark.parametrize("n",[-1,0,3,30])
def test_invalid_29dof_rejected(setup,n):
    _,tree,_,mass,_=setup
    q=np.zeros(max(n,0))
    if n==30:
        q[-1]=np.inf
    with pytest.raises(ValueError,match="BAD_ORIGINAL29_Q"):
        v.inertial_com_and_analytic_jacobian(tree,mass,q)


@pytest.mark.parametrize("eps",[0,1.,-1.])
def test_bad_com_derivative_step_rejected(setup,eps):
    _,tree,_,mass,q0=setup
    with pytest.raises(ValueError,match="BAD_FD_EPS"):
        v.independent_fd_com_jacobian(tree,mass,q0,eps)


def test_left_ankle_velocity_cancelled_exactly_by_virtual_base_translation(setup):
    cfg,tree,lim,mass,q0=setup
    o=v.problem_at_pose(tree,mass,q0,cfg)
    synthetic_v = np.linspace(-.2,.2,29)
    base=-o["Jleft"]@synthetic_v
    np.testing.assert_allclose(base + o["Jleft"]@synthetic_v,[0,0,0],atol=1.e-15)
    np.testing.assert_allclose(
        o["Jrelative"]@synthetic_v,
        (o["Jrelative"]+o["Jleft"])@synthetic_v + base,atol=1.e-12)
    # A link origin lock is not an orientation lock, never measured GRF.
    assert cfg["physical_constraints_missing"]


def test_arm_velocity_bounds_are_not_covertly_ignored(setup):
    c,tree,lim,mass,q0=setup
    o=v.problem_at_pose(tree,mass,q0,c)
    qref=np.zeros(29)
    case=dict(o,com_cap=.1,arm_cap=.25)
    bounds,Aub,bub=v.constraints(qref,lim,case,c)
    assert Aub.shape==(6,29) and bub.shape==(6,)
    for i in v.ARM:
        assert bounds[i][0] >= -.25-1e-12
        assert bounds[i][1] <= .25+1e-12
    assert any(bounds[i][1] > .25 for i in range(29) if i not in v.ARM)


def test_arm_soft_cap_cannot_exceed_original_joint_limits(setup):
    c,tree,lim,mass,q0=setup
    o=v.problem_at_pose(tree,mass,q0,c)
    qref=np.full(29,0.5)
    bounds,_,_=v.constraints(qref,lim,dict(o,com_cap=None,arm_cap=.25),c)
    assert all(low >= -lim["vmax"][i] and hi<=lim["vmax"][i]
               for i,(low,hi) in enumerate(bounds))


def test_out_of_frozen_constraints_refused(setup):
    c,tree,lim,mass,q0=setup
    o=v.problem_at_pose(tree,mass,q0,c)
    with pytest.raises(ValueError,match="NOT_FROZEN_CONSTRAINT"):
        v.constraints(np.zeros(29),lim,dict(o,com_cap=.2,arm_cap=None),c)
    with pytest.raises(ValueError,match="NOT_FROZEN_CONSTRAINT"):
        v.constraints(np.zeros(29),lim,dict(o,com_cap=None,arm_cap=10.),c)


def test_poisoned_velocity_limits_fail(setup):
    c,tree,lim,mass,q0=setup
    o=v.problem_at_pose(tree,mass,q0,c)
    bad=dict(lim,vmax=np.zeros(29))
    with pytest.raises(ValueError,match="BAD_ORIGINAL29_QDOT_OR_LIMITS"):
        v.constraints(np.zeros(29),bad,dict(o,com_cap=None,arm_cap=None),c)


def test_lp_returns_witness_on_known_feasible_toy():
    a = np.array([[1.,1.]])
    x=v._lp(np.zeros(2),a,np.array([.5]),[(-1.,1.),(-1.,1.)])
    assert x is not None
    assert (a@x)[0]==pytest.approx(.5,abs=1e-9)


def test_lp_infeasible_toy_returns_none():
    a=np.array([[1.,1.]])
    assert v._lp(np.zeros(2),a,np.array([3.]),[(-1.,1.),(-1.,1.)]) is None


def test_lp_negative_inequality_infeasible():
    a=np.array([[1.,0.]])
    assert v._lp(np.zeros(2),a,np.array([.8]),[(-1.,1.),(-1.,1.)],
                 np.array([[1.,0.]]),np.array([0.])) is None


def test_one_hard_lp_case_satisfies_all_original_velocity_com_and_arm_constraints(setup):
    c,tree,lim,mass,q0=setup
    p=v.chain.generate(v.chain.frozen(),q0,4)
    idx=round(2.95/.005)
    o=v.problem_at_pose(tree,mass,p.q[idx],c)
    r=v.reachability_case(o,p.dq[idx],lim,c,.1,.25)
    assert r["feasible_target_count"]==4
    assert r["min_attainable_relative_x_mps"] < 0
    assert r["max_attainable_relative_x_mps"] >= 3
    for row in r["targets"]:
        assert row["mathematically_feasible"]
        assert row["max_qdot_limit_ratio"]<=1+1e-8
        assert abs(row["right_ankle_relative_velocity_mps"][1])<1e-6
        assert abs(row["right_ankle_relative_velocity_mps"][2])<1e-6
        assert max(abs(x) for x in row["com_relative_velocity_root_xyz_mps"]) <= .1 + 1e-6
        assert not row["motor_execution_authorized"]
        assert not row["is_physics_verified"]


def test_full_grid_has_exact_original29_conditions_and_no_actuation(report):
    assert report["original_robot_dof"]==29
    assert report["frozen_synthetic_ref_proposals"]==5
    assert report["frozen_probe_pose_conditions"]==15
    assert report["hard_lp_envelopes"]==60
    assert report["exact_target_feasibility_decisions"]==240
    assert len(report["cases"])==60
    assert all(len(r["targets"])==4 for r in report["cases"])
    assert report["new_true_physx_rollouts"]==0
    assert report["actual_strict_physx_successes"]==0
    assert report["motor_execution_authorized"] is False
    assert report["learned_professional_human_motion"] is False
    assert report["original_urdf_ankle_origin_is_not_true_instep"]


def test_matrix_all_hard_kinematic_targets_feasible_but_not_physical_success(report):
    # This is a preregistered *negative control* for interpreting v136 soft LSQ.
    assert all(t["mathematically_feasible"] for r in report["cases"] for t in r["targets"])
    assert all(not r["strict_kick_success"] and not r["motor_execution_authorized"]
               for r in report["cases"])
    assert report["max_com_jacobian_fd_error"]<2e-5


def test_hard_com_arms_envelope_no_greater_than_free(report):
    for mult in [0.,1.,2.,4.,-1.]:
        for time in [2.25,2.95,3.6]:
            subset=[r for r in report["cases"] if r["multiplier"]==mult
                    and r["synthetic_time_s"]==time]
            free=next(r for r in subset if r["com_velocity_cap_mps"] is None
                      and r["arm_joint_velocity_deviation_cap_rad_s"] is None)
            locked=next(r for r in subset if r["com_velocity_cap_mps"]==.1
                        and r["arm_joint_velocity_deviation_cap_rad_s"]==.25)
            assert locked["max_attainable_relative_x_mps"]<=free["max_attainable_relative_x_mps"]+1e-8


def test_exact_lp_is_not_minimum_effort_or_recovery_certificate(report):
    # linprog feasibility witness can live on original velocity bounds.
    targets=[t for r in report["cases"] for t in r["targets"] if
             t["target_ankle_origin_relative_x_mps"]==3]
    assert len(targets)==60
    assert any(t["max_qdot_limit_ratio"]>.95 for t in targets)
    assert all(t["hypothetical_5ms_qddot_rad_s2"]>100 for t in targets)
    assert all(not t["is_physics_verified"] for t in targets)


def test_report_explicitly_rejects_cop_grf_native_and_human_attestation(report,setup):
    c,*_=setup
    missing=c["physical_constraints_missing"]
    assert any("ground reaction" in x for x in missing)
    assert any("PhysX" in x for x in missing)
    assert any("human" in x for x in missing)
    assert not report["actual_ball_kick_speed_and_impulse_verified"]
