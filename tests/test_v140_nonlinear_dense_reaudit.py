"""Source-locked, independent dense kinematics and false-evidence negative tests."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v140_nonlinear_dense_reaudit as v
import v140_independent_nonlinear_audit as audit


@pytest.fixture(scope="module")
def sources():
    c=v.frozen(); base=v.chain.frozen(); pc=v.lp.frozen()
    tree,lim=v.lp.original_kinematics(v.task.URDF,pc)
    masses=v.task.mass_inertials(v.task.URDF,tree)
    return c,base,pc,tree,lim,masses


@pytest.fixture(scope="module")
def static_qp(sources):
    c,base,pc,tree,lim,masses=sources
    p=v.chain.generate(base,(lim["lo"]+lim["hi"])/2,0.)
    problem=v.lp.problem_at_pose(tree,masses,p.q[590],pc)
    row=v.parent.solve(p,problem,lim,v.parent.frozen(),0.,.8)
    assert row["solved"]
    return p,row


def test_frozen_protocol_and_parent_sha(sources):
    c,*_=sources
    assert subprocess.check_output(["git","-C",str(ROOT),"show",v.FROZEN_SHA+":configs/v140_frozen_nonlinear_dense_audit.json"])==v.CONFIG.read_bytes()
    assert subprocess.check_output(["git","-C",str(ROOT),"rev-parse",v.FROZEN_SHA+"^"],text=True).strip()==c["parent_sha"]
    assert v.parent.FROZEN_SHA==c["parent_protocol_sha"]
    assert c["required_cells"]==45
    assert c["new_physx_rollouts"]==0


def test_source_urdf_remains_original29(sources):
    c,base,pc,tree,lim,masses=sources
    assert len(v.chain.G1_JOINT_ORDER)==29
    assert len(masses)==37
    assert all(len(lim[k])==29 for k in ("lo","hi","vmax","taumax"))
    assert sum(m for _,m,_ in masses)>33


def test_config_mutation_rejected(monkeypatch,tmp_path):
    fake=tmp_path/"config.json"
    fake.write_bytes(v.CONFIG.read_bytes())
    monkeypatch.setattr(v,"CONFIG",fake)
    with pytest.raises(ValueError,match="PATH_DRIFT"):
        v.frozen()


def test_config_content_mutation_rejected(monkeypatch):
    monkeypatch.setattr(Path,"read_bytes",lambda _:b'{}')
    with pytest.raises(ValueError,match="BYTE_DRIFT"):
        v.frozen()


@pytest.mark.parametrize("mult",[0,1,2,4,-1])
def test_independent_analytic_reference_reproduces_old_200hz(sources,mult):
    c,base,pc,tree,lim,masses=sources
    original=v.chain.generate(base,(lim["lo"]+lim["hi"])/2,mult)
    got=v.analytic_reference(original.t,(lim["lo"]+lim["hi"])/2,mult,base)
    for a,b in zip(got,(original.q,original.dq,original.ddq)):
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize("time",[.213,1.203,2.307,2.955,3.417,4.211,10.331])
def test_dense_reference_derivatives_by_independent_fd(sources,time):
    c,base,pc,tree,lim,masses=sources
    q0=(lim["lo"]+lim["hi"])/2
    eps=1e-5
    t=np.array([time-eps,time,time+eps])
    q,dq,ddq=v.analytic_reference(t,q0,1.,base)
    np.testing.assert_allclose(dq[1],(q[2]-q[0])/(2*eps),atol=1e-8,rtol=1e-5)
    np.testing.assert_allclose(ddq[1],(dq[2]-dq[0])/(2*eps),atol=1e-6,rtol=1e-4)


@pytest.mark.parametrize("bad",[np.nan,np.inf,-1,14.,15.])
def test_invalid_time_is_rejected(sources,bad):
    c,base,pc,tree,lim,masses=sources
    with pytest.raises(ValueError,match="INVALID_DENSE_ANALYTIC_REFERENCE"):
        v.analytic_reference(np.array([bad]),(lim["lo"]+lim["hi"])/2,1.,base)


def test_parent_coefficient_recovery_and_zero_negative_control(sources,static_qp):
    c,base,pc,tree,lim,masses=sources
    p,sol=static_qp
    coef,errs=v.recover_coefficients(p,sol,2.95,.8)
    assert coef.shape==(4,29)
    assert abs(coef).max()<1e-10
    assert max(errs)<1e-8
    z=v.dense_evaluate(p,sol,lim,tree,masses,pc,c,.8,0.)
    assert z["dense_samples"]==14000
    assert z["new_kinematic_admitted"] is True
    assert z["veto_reasons"]==[]
    assert z["native_physx_pair_certified"] is False
    assert z["motor_execution_authorized"] is False


def test_tampered_parent_witness_cannot_be_forged(sources,static_qp):
    c,base,pc,tree,lim,masses=sources
    p,sol=static_qp
    fake=dict(sol)
    fake["_q"]=sol["_q"].copy()
    fake["_q"][100,0]+=.01
    with pytest.raises(ValueError,match="RECONSTRUCTION_CANNOT_ATTEST"):
        v.recover_coefficients(p,fake,2.95,.8)


def test_nonzero_qp_center_basis_displacement_is_zero(sources):
    c,base,pc,tree,lim,masses=sources
    p=v.chain.generate(base,(lim["lo"]+lim["hi"])/2,0.)
    problem=v.lp.problem_at_pose(tree,masses,p.q[590],pc)
    x=v.parent.solve(p,problem,lim,v.parent.frozen(),1.,.8)
    if not x["solved"]:pytest.skip("parent QP failed; negative status must remain")
    assert np.max(np.abs(x["_q"][590]-p.q[590]))<1e-10
    o=v.dense_evaluate(p,x,lim,tree,masses,pc,c,.8,1.)
    assert o["center_joint_pose_change_rad"]<1e-10
    assert o["exact_pose_center_task_residual_mps"]<c["nonlinear_ankle_center_velocity_residual_tolerance_mps"]
    assert not o["strict_physics_success"]


def test_source_com_exact_pose_rejects_bad_shape(sources):
    c,base,pc,tree,lim,masses=sources
    with pytest.raises(ValueError,match="BAD_NONLINEAR_CENTER"):
        v.exact_center_kinematics(tree,masses,np.zeros(28),np.zeros(29),pc)


def fake_audit():
    rows=[]
    for m,t,w in sorted(audit.GRID):
        rows.append({"multiplier":m,"target_mps":t,"halfwidth_s":w,
                     "parent_qp_status":"primal infeasible",
                     "parent_solved":False,"parent_reason":"QP_PRIMAL_INFEASIBLE",
                     "new_kinematic_admitted":False,"veto_reasons":["PARENT_QP_NOT_ADMITTED"]})
    return {"version":"v0.1.140",
        "origin":"CPU_ONLY_SYNTHETIC_29DOF_KINEMATICS_NOT_PHYSICS",
        "new_physx_rollouts":0,"strict_physics_success_rate":None,
        "native_contact_pair_certified":False,
        "human_elite_instep_kinetics_truth":False,
        "motor_execution_authorized":False,
        "attempts":rows,
        "summary":{"total_attempts":45,"parent_rejected":45,"parent_admitted":0,
                   "new_kinematic_admitted":0,"new_kinematic_rejected_or_unresolved":45}}


def test_complete_independent_negative_grid():
    x=audit.audit(fake_audit())
    assert x["all_frozen_cells"]==45 and x["original_QP_rejected"]==45
    assert x["strict_dense_nonlinear_pass"]==0
    assert x["new_real_physx_shots"]==0


def test_independent_auditor_rejects_missing_cell():
    x=fake_audit();x["attempts"].pop()
    with pytest.raises(ValueError,match="MISSING_GRID_CELLS"):audit.audit(x)


def test_independent_auditor_rejects_duplicate_cell():
    x=fake_audit();x["attempts"][3]=x["attempts"][2].copy()
    with pytest.raises(ValueError,match="DUPLICATE_OR_UNREGISTERED"):audit.audit(x)


@pytest.mark.parametrize("attr,value",[("new_physx_rollouts",1),
  ("strict_physics_success_rate",1.0),("native_contact_pair_certified",True),
  ("human_elite_instep_kinetics_truth",True),("motor_execution_authorized",True)])
def test_independent_auditor_refuses_fake_physical_provenance(attr,value):
    x=fake_audit();x[attr]=value
    with pytest.raises(ValueError,match="FORGED_PHYSICAL_PROVENANCE"):audit.audit(x)


def test_independent_auditor_rejects_fake_parent_upgrade():
    x=fake_audit();x["attempts"][0]["new_kinematic_admitted"]=True
    with pytest.raises(ValueError,match="FALSE_UPGRADE"):audit.audit(x)


def test_independent_auditor_rejects_fake_summary():
    x=fake_audit();x["summary"]["parent_rejected"]=44
    with pytest.raises(ValueError,match="BAD_SUMMARY"):audit.audit(x)


def test_independent_auditor_rejects_true_parent_with_wrong_veto():
    x=fake_audit()
    row=x["attempts"][0]
    row.update(parent_qp_status="solved",parent_solved=True,
        parent_reason="PASS_SAMPLED_KINEMATICS_ONLY",
        dense_samples=14000,dense_dt_s=.001,
        dense_max_position_bound_violation_rad=0.,
        dense_max_original29_velocity_violation_rad_s=0.,
        dense_max_research_accel_violation_rad_s2=1.,
        exact_pose_center_task_residual_mps=0.,
        exact_pose_center_com_relative_speed_max_abs_mps=0.,
        center_joint_pose_change_rad=0.,
        new_kinematic_admitted=True,veto_reasons=[],
        native_physx_pair_certified=False,motor_execution_authorized=False,
        strict_physics_success=False)
    with pytest.raises(ValueError,match="MISCLASSIFIED_FROZEN_PHYSICS_VETO"):
        audit.audit(x)


def test_independent_auditor_never_claims_continuous_time_certificate():
    x=audit.audit(fake_audit())
    assert "continuous_time_guaranteed" not in x
    assert x["strict_real_physics_success_rate"] is None
