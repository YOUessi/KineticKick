"""v139 CPU preregistration, OSQP full-window C2 and physical fail-closed tests."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v139_joint_window_qp as v

@pytest.fixture(scope="module")
def setup():
    c=v.frozen()
    pc=v.lp.frozen()
    tree,lim=v.lp.original_kinematics(v.task.URDF,pc)
    masses=v.task.mass_inertials(v.task.URDF,tree)
    return c,pc,tree,lim,masses

@pytest.fixture(scope="module")
def static(setup):
    c,pc,tree,lim,masses=setup
    p=v.chain.generate(v.chain.frozen(),(lim["lo"]+lim["hi"])/2,0.)
    mod=v.lp.problem_at_pose(tree,masses,p.q[590],pc)
    return p,mod

def test_protocol_commit_parent_and_bytes(setup):
    c,*_=setup
    assert subprocess.check_output(["git","-C",str(ROOT),"rev-parse",v.FROZEN_SHA+"^"],text=True).strip()==c["parent_sha"]
    assert subprocess.check_output(["git","-C",str(ROOT),"show",v.FROZEN_SHA+":configs/v139_frozen_joint_window_qp.json"])==v.CONFIG.read_bytes()
    assert c["required_grid_cells"]==45
    assert c["motor_execution_authorized"] is False and c["actual_new_physx_rollouts"]==0

def test_not_a_human_or_native_contact_witness(setup):
    c,*_=setup
    assert c["real_grf_or_cop_available"] is False
    assert c["human_professional_instep_force_truth_available"] is False
    assert c["native_contact_pair_certified"] is False
    assert c["strict_physics_gates"]["instep_impulse_fraction_min"]==.95

def test_config_path_tamper(tmp_path,monkeypatch):
    fake=tmp_path/"config.json"
    fake.write_bytes(v.CONFIG.read_bytes())
    monkeypatch.setattr(v,"CONFIG",fake)
    with pytest.raises(ValueError,match="PATH_DRIFT"):
        v.frozen()

def test_config_content_drift(monkeypatch):
    monkeypatch.setattr(Path,"read_bytes",lambda self:b"{}")
    with pytest.raises(ValueError,match="BYTE_DRIFT"):
        v.frozen()

def test_original_urdf_tamper(setup,tmp_path):
    c,pc,*_=setup
    path=tmp_path/"spoofed.urdf"
    path.write_bytes(v.task.URDF.read_bytes()+b"\n<!-- tampered -->")
    with pytest.raises(ValueError,match="SOURCE_SHA256_DRIFT"):
        v.lp.original_kinematics(path,pc)

@pytest.mark.parametrize("power",[1,3,5,7])
@pytest.mark.parametrize("s",[-.83,-.42,-.11,.0,.37,.78])
def test_independent_centered_finite_differences(power,s):
    width=.8; eps=1e-6; center=2.95
    t=center+width*s
    val,d,dd=v.basis(np.array([t]),center,width,(power,))
    vp,dp,_=v.basis(np.array([t+eps]),center,width,(power,))
    vm,dm,_=v.basis(np.array([t-eps]),center,width,(power,))
    np.testing.assert_allclose(d,(vp-vm)/(2*eps),atol=1e-8,rtol=1e-7)
    np.testing.assert_allclose(dd,(dp-dm)/(2*eps),atol=1e-7,rtol=1e-6)
    assert np.isfinite(val).all()

@pytest.mark.parametrize("s",[-2.,-1.,1.,2.])
def test_compact_c2_boundary_exact(s):
    b,d,dd=v.basis(np.array([2.95+s*.8]),2.95,.8)
    assert not b.any() and not d.any() and not dd.any()

def test_center_basis_only_first_derivative_nonzero():
    b,d,dd=v.basis(np.array([2.95]),2.95,1.2)
    np.testing.assert_array_equal(b,np.zeros((1,4)))
    np.testing.assert_array_equal(d,np.array([[1,0,0,0.]]))
    np.testing.assert_array_equal(dd,np.zeros((1,4)))

@pytest.mark.parametrize("bad",[float("nan"),float("inf"),float("-inf")])
def test_invalid_time_nan_inf_rejected(bad):
    with pytest.raises(ValueError,match="BAD_BASIS_TIME"):
        v.basis(np.array([bad]),2.95,.8)

@pytest.mark.parametrize("bad",[-1,2,4,False])
def test_unfrozen_target_rejected(setup,static,bad):
    c,pc,tree,lim,masses=setup
    p,prob=static
    with pytest.raises(ValueError,match="UNFROZEN_V139_TARGET"):
        v.constrained_problem(p,prob,lim,c,bad,.8)

@pytest.mark.parametrize("bad",[0,.2,.6,False])
def test_unfrozen_width_rejected(setup,static,bad):
    c,pc,tree,lim,masses=setup
    p,prob=static
    with pytest.raises(ValueError,match="UNFROZEN_V139_WIDTH"):
        v.constrained_problem(p,prob,lim,c,1.,bad)

def test_full_29_joint_and_14_second_reference(setup,static):
    c,pc,tree,lim,masses=setup
    p,prob=static
    assert p.q.shape==(2800,29)
    assert len(masses)==37
    assert len(v.lp.ARM)==14
    assert abs(sum(m for _,m,_ in masses)-33.341142022)<1e-6
    qp=v.constrained_problem(p,prob,lim,c,0.,.8)
    assert qp["P"].shape==(116,116)
    assert qp["A"].shape[1]==116
    assert qp["lo"].shape==qp["hi"].shape==(qp["A"].shape[0],)
    assert np.max(qp["lo"]-qp["hi"])<=1e-8

def test_qp_objective_psd_and_task_jacobian_exact(setup,static):
    c,pc,tree,lim,masses=setup
    p,prob=static
    qp=v.constrained_problem(p,prob,lim,c,1.,.8)
    np.testing.assert_allclose(qp["P"].toarray(),qp["P"].toarray().T,atol=1e-12)
    assert np.linalg.eigvalsh(qp["P"].toarray())[0]>0
    assert qp["task_rows"].shape==(3,116)

def test_zero_velocity_negative_control_has_no_motion(setup,static):
    c,pc,tree,lim,masses=setup
    p,prob=static
    z=v.solve(p,prob,lim,c,0.,.8)
    assert z["solved"] and z["reason"]=="PASS_SAMPLED_KINEMATICS_ONLY"
    np.testing.assert_array_equal(z["_q"],p.q)
    np.testing.assert_array_equal(z["_dq"],p.dq)
    np.testing.assert_array_equal(z["_ddq"],p.ddq)
    assert z["strict_physics_success"] is False

def test_nonzero_target_uses_multiple_bases_and_never_authorizes_motor(setup,static):
    c,pc,tree,lim,masses=setup
    p,prob=static
    z=v.solve(p,prob,lim,c,1.,.8)
    if z["solved"]:
        assert z["coefficients_nonzero"]>29
        assert z["center_task_residual_mps"]<=c["postsolve_max_center_task_velocity_residual_mps"]
        assert z["max_analytic_qddot_rad_s2"]<=c["research_only_qddot_screen_hard_rad_s2"]+c["postsolve_max_absolute_acceleration_violation_rad_s2"]
        assert z["max_absolute_perturbation_rad"]>0
        assert np.array_equal(z["_q"][0],p.q[0])
        assert np.array_equal(z["_q"][-1],p.q[-1])
        assert not z["motor_execution_authorized"] and not z["real_foot_contact_or_grf_certified"]
    else:
        assert z["reason"].startswith("QP_") or z["reason"]=="POSTSOLVE_HARD_CONSTRAINT_VIOLATION"

def test_unreachable_target_is_not_misreported_as_shot(setup,static):
    c,pc,tree,lim,masses=setup
    p,prob=static
    z=v.solve(p,prob,lim,c,3.,.8)
    if not z["solved"]:
        assert "reason" in z and z["solver_iterations"]>0
    else:
        assert z["strict_physics_success"] is False

def test_out_of_shape_model_rejected(setup,static):
    c,pc,tree,lim,masses=setup
    p,prob=static
    bad=dict(prob,Jrelative=np.zeros((3,28)))
    with pytest.raises(ValueError,match="MALFORMED_ORIGINAL29"):
        v.constrained_problem(p,bad,lim,c,1.,.8)

def test_invalid_dof_reference_rejected(setup,static):
    import dataclasses
    c,pc,tree,lim,masses=setup
    p,prob=static
    p=dataclasses.replace(p,q=p.q[:200])
    with pytest.raises(ValueError,match="BAD_14S_FULL29_REFERENCE"):
        v.constrained_problem(p,prob,lim,c,1.,.8)

def test_full_grid_reports_no_hallucinated_rollouts(setup):
    c,*_=setup
    # Reconstruct the entire 45-cell denominator directly from the freeze, not its outcomes.
    combos=[(a,vv,w) for a in c["multipliers"] for vv in c["relative_right_minus_left_ankle_origin_x_targets_mps"] for w in c["compact_halfwidths_s"]]
    assert len(combos)==len(set(combos))==c["required_grid_cells"]
    assert c["actual_new_physx_rollouts"]==0

def fake_complete_audit():
    import v139_independent_grid_audit as audit
    rows=[dict(multiplier=float(a),target_mps=float(b),halfwidth_s=float(w),
               solved=False,solver_status="primal infeasible",
               solver_iterations=1,reason="QP_PRIMAL_INFEASIBLE")
          for a,b,w in audit.EXPECTED]
    return {"version":"v0.1.139","origin":"SYNTHETIC_CPU_QP_NOT_PHYSICS",
            "original29_dof":29,"new_physx_rollouts":0,
            "human_professional_instep_learned":False,
            "genuine_physical_shooting_success_rate":None,
            "attempts":rows,"summary":{"attempts":45,"qp_solved":0,"qp_rejected":45}}

def test_independent_auditor_accepts_complete_negative_grid():
    import v139_independent_grid_audit as audit
    x=audit.validate(fake_complete_audit())
    assert x["complete_cells"]==45
    assert x["not_admitted"]==45
    assert x["kinematic_screen_admitted"]==0
    assert x["strict_physics_success_rate"] is None

def test_independent_auditor_catches_missing_cell():
    import v139_independent_grid_audit as audit
    x=fake_complete_audit()
    x["attempts"].pop()
    with pytest.raises(ValueError,match="MISSING_FROZEN_ATTEMPTS"):
        audit.validate(x)

def test_independent_auditor_catches_duplicate_cell():
    import v139_independent_grid_audit as audit
    x=fake_complete_audit()
    x["attempts"][2]=x["attempts"][1].copy()
    with pytest.raises(ValueError,match="DUPLICATE_CELL"):
        audit.validate(x)

def test_independent_auditor_never_confuses_reported_success():
    import v139_independent_grid_audit as audit
    x=fake_complete_audit()
    x["genuine_physical_shooting_success_rate"]=1
    with pytest.raises(ValueError,match="FALSE_PHYSICAL_PROVENANCE"):
        audit.validate(x)

def test_independent_auditor_catches_misleading_summary():
    import v139_independent_grid_audit as audit
    x=fake_complete_audit()
    x["summary"]["qp_solved"]=1
    with pytest.raises(ValueError,match="SUMMARY_DENOMINATOR_MISMATCH"):
        audit.validate(x)

def test_independent_auditor_does_not_accept_fake_kinematic_pass():
    import v139_independent_grid_audit as audit
    x=fake_complete_audit()
    x["attempts"][0]["solved"]=True
    x["attempts"][0]["reason"]="PASS_SAMPLED_KINEMATICS_ONLY"
    with pytest.raises(ValueError,match="FAKE_KINEMATIC"):
        audit.validate(x)

def test_independent_auditor_rejects_forged_unexecuted_rollout():
    import v139_independent_grid_audit as audit
    x=fake_complete_audit()
    x["new_physx_rollouts"]=1
    with pytest.raises(ValueError,match="FALSE_PHYSICAL_PROVENANCE"):
        audit.validate(x)
