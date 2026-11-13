"""v141 polynomial identity, hidden-peak negative controls and independent auditor."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from numpy.polynomial import Polynomial as Poly

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v141_continuous_envelope as v
import v141_independent_envelope_audit as independent

@pytest.fixture(scope="module")
def fixture():
    c=v.frozen();b=v.chain.frozen();pc=v.lp.frozen()
    tree,limits=v.lp.original_kinematics(v.task.URDF,pc)
    q0=(limits["lo"]+limits["hi"])/2
    return c,b,tree,limits,q0

def test_parent_commit_and_source_protocol(fixture):
    c,*_=fixture
    assert subprocess.check_output(["git","-C",str(ROOT),"show",v.FROZEN_SHA+":configs/v141_frozen_continuous_polynomial_envelope.json"])==v.CONFIG.read_bytes()
    assert subprocess.check_output(["git","-C",str(ROOT),"rev-parse",v.FROZEN_SHA+"^"],text=True).strip()==c["parent_sha"]
    assert c["frozen_grid_cells"]==45 and c["motor_execution_authorized"] is False

def test_original29_limit_shape(fixture):
    c,b,t,limits,q0=fixture
    assert q0.shape==(29,)
    assert len(v.chain.G1_JOINT_ORDER)==29
    assert all(limits[k].shape==(29,) for k in ("lo","hi","vmax","taumax"))

@pytest.mark.parametrize("width",[.4,.8,1.2])
def test_complete_phase_plus_support_partition(width):
    s=v.segments(width)
    assert s[0][0]==0 and s[-1][1]==14
    assert all(abs(s[k][1]-s[k+1][0])<1e-12 for k in range(len(s)-1))
    assert any(abs(a-(2.95-width))<1e-10 for a,b in s)
    assert any(abs(b-(2.95+width))<1e-10 for a,b in s)

@pytest.mark.parametrize("width",[.4,.8,1.2])
@pytest.mark.parametrize("mult",[0,1,2,4,-1])
def test_analytic_29_joint_reference_match_arbitrary_subframes(fixture,width,mult):
    c,b,t,limits,q0=fixture
    coeff=np.zeros((4,29))
    errors=[]
    for a,z in v.segments(width):
        pieces=v.polynomial_piece(a,z,mult,width,coeff,q0,b)
        errors.append(v.verify_piece(pieces,a,z,b,coeff,mult,width,q0,c))
    assert max(errors)<1e-8

@pytest.mark.parametrize("width",[.4,.8,1.2])
def test_nontrivial_correction_all29_matches_independent_analytic(fixture,width):
    c,b,t,limits,q0=fixture
    coeff=np.arange(4*29,dtype=float).reshape(4,29)/193.
    for a,z in v.segments(width):
        p=v.polynomial_piece(a,z,1,width,coeff,q0,b)
        assert v.verify_piece(p,a,z,b,coeff,1,width,q0,c)<2e-7

@pytest.mark.parametrize("coeff",[[2.],[-1.,3.],[1.,-2.,1.],[0.,0.,1.,0.,1.]])
def test_power_to_bernstein_matches_exact_polynomial_samples(coeff):
    p=Poly(coeff);h=v.power_to_bernstein(p)
    from math import comb
    degree=len(h)-1
    for x in [0,.123,.5,.91,1]:
        val=sum(h[i]*comb(degree,i)*x**i*(1-x)**(degree-i) for i in range(degree+1))
        assert abs(val-p(x))<1e-12

def test_de_casteljau_preserves_polynomial_halves():
    p=Poly([.3,-2,8,-1]);b=v.power_to_bernstein(p)
    l,r=v.bisect_bernstein(b)
    from math import comb
    n=len(l)-1
    for x in [.0,.1,.33,.5,.81,1.]:
        evalbern=lambda z:sum(z[i]*comb(n,i)*x**i*(1-x)**(n-i) for i in range(n+1))
        assert abs(evalbern(l)-p(x*.5))<1e-12
        assert abs(evalbern(r)-p(.5+x*.5))<1e-12

def test_interior_peak_bypasses_sparse_sample_and_is_rejected(fixture):
    c,*_=fixture
    # p(x)=0.0100 - (x-0.5)**2; p(0),p(1) pass 0.005;
    # p(.5) fails. A two-endpoint sampled screen would miss it.
    p=Poly([-.24,1.,-1.])
    x=v.bound_one(p,-1.,.005,0.,1.,c)
    assert x["state"]=="FAIL"
    assert 0<x["time_s"]<1
    assert x["value"]>.005

def test_known_safe_polynomial_admits(fixture):
    c,*_=fixture
    x=v.bound_one(Poly([0.,.05,-.05]),-.01,.1,0,1,c)
    assert x["state"]=="PASS"

def test_negative_constant_at_interval_boundary_rejected(fixture):
    c,*_=fixture
    x=v.bound_one(Poly([.125]),-.1,.1,2.,3.,c)
    assert x["state"]=="FAIL" and x["time_s"]==2.

def test_depth_limited_near_safety_boundary_is_unresolved(fixture):
    c,*_=fixture
    c=copy.copy(c);c["bernstein_max_depth"]=0
    # Bernstein hull has extraneous overshoot but endpoints are admissible.
    x=v.bound_one(Poly([0.,1.,-1.]),-.01,.28,0,1,c)
    assert x["state"]=="UNRESOLVED"

@pytest.mark.parametrize("bad",[[float('nan')],[float('inf')]])
def test_nonfinite_polynomial_rejected(bad):
    with pytest.raises(ValueError,match="NONFINITE_POLYNOMIAL"):
        v.power_to_bernstein(Poly(bad))

def test_high_degree_is_rejected():
    with pytest.raises(ValueError,match="DEGREE"):
        v.power_to_bernstein(Poly([1.]*17))

@pytest.mark.parametrize("badwidth",[.2,0.,1.4])
def test_unregistered_width_refused(badwidth):
    with pytest.raises(ValueError,match="UNREGISTERED"):
        v.segments(badwidth)

def test_tampered_config_location_refused(monkeypatch,tmp_path):
    fake=tmp_path/"v141.json";fake.write_bytes(v.CONFIG.read_bytes())
    monkeypatch.setattr(v,"CONFIG",fake)
    with pytest.raises(ValueError,match="PATH_DRIFT"):
        v.frozen()

def test_tampered_config_bytes_refused(monkeypatch):
    monkeypatch.setattr(Path,"read_bytes",lambda _:b'{}')
    with pytest.raises(ValueError,match="BYTE_DRIFT"):
        v.frozen()

def test_static_real_original29_zero_correction_has_envelope_pass(fixture):
    c,b,t,lim,q0=fixture
    x=v.inspect(np.zeros((4,29)),lim,0.,.8,b,q0,c)
    assert x["numeric_envelope_state"]=="PASS"
    assert x["polynomial_bounds_checked"]>=29*3*7
    assert x["violations_count"]==0 and x["unresolved_count"]==0
    assert not x["first_violation"]
    assert x["floating_point_conservative_not_formally_verified"] is True

def fake_grid():
    rows=[{"multiplier":m,"target_mps":t,"halfwidth_s":w,
           "parent_solved":False,"parent_qp_status":"primal infeasible",
           "parent_reason":"QP_PRIMAL_INFEASIBLE",
           "v140_pass":False,"numeric_envelope_state":"PARENT_QP_REJECTED",
           "v141_admitted":False,"native_physx_contact_pair_certified":False,
           "physical_success":False} for m,t,w in sorted(independent.GRID)]
    return {"version":"v0.1.141","provenance":independent.SOURCE,
         "protocol_sha":v.FROZEN_SHA,
         "source_parent_sha":"df46cf40857f4cde177aa14cae6e93092dc3adff",
         "rows":rows,"summary":{"total":45,"parent_rejected_or_unresolved":45,"v141_pass":0},
         "new_physx_rollouts":0,"physical_success_rate":None,
         "native_physx_contact_pair_certified":False,
         "professional_human_instep_motion_learned":False,"motor_execution_authorized":False}

def test_independent_negative_grid_roundtrip():
    x=independent.audit(fake_grid())
    assert x["total"]==45 and x["envelope_pass"]==0

def test_independent_detects_missing_cell():
    x=fake_grid();x["rows"].pop()
    with pytest.raises(ValueError,match="INCOMPLETE_FROZEN_GRID"):
        independent.audit(x)

def test_independent_detects_duplicate_cell():
    x=fake_grid();x["rows"][2]=x["rows"][1].copy()
    with pytest.raises(ValueError,match="DUPLICATE"):
        independent.audit(x)

@pytest.mark.parametrize("field,value",[("new_physx_rollouts",1),
  ("physical_success_rate",.5),("native_physx_contact_pair_certified",True),
  ("professional_human_instep_motion_learned",True),("motor_execution_authorized",True)])
def test_independent_denies_fabricated_physics(field,value):
    x=fake_grid();x[field]=value
    with pytest.raises(ValueError,match="FORGED_PHYSICAL_PROVENANCE"):
        independent.audit(x)

def test_independent_denies_qp_rejection_promotion():
    x=fake_grid();x["rows"][0]["v141_admitted"]=True
    with pytest.raises(ValueError,match="FAKE_ENVELOPE_ADMISSION"):
        independent.audit(x)

def test_independent_denies_summary_tamper():
    x=fake_grid();x["summary"]["total"]=44
    with pytest.raises(ValueError,match="SUMMARY_COUNT_DRIFT"):
        independent.audit(x)

def test_bounded_intersample_violation_has_independent_analytic_witness(fixture):
    c,b,tree,lim,q0=fixture
    masses=v.task.mass_inertials(v.task.URDF,tree)
    p=v.chain.generate(b,q0,4)
    problem=v.lp.problem_at_pose(tree,masses,p.q[590],v.lp.frozen())
    solution=v.qp.solve(p,problem,lim,v.qp.frozen(),0,.4)
    assert solution["solved"]  # original v139 sampled QP positive
    coeff,_=v.dense.recover_coefficients(p,solution,2.95,.4)
    scan=v.inspect(coeff,lim,4,.4,b,q0,c)
    assert scan["numeric_envelope_state"]=="FAIL"
    assert len(scan["violation_witnesses"])==scan["violations_count"]
    assert scan["violation_witnesses"][0]==scan["first_violation"]
    hit=scan["first_violation"]
    assert hit["kind"]=="RESEARCH_ACCEL"
    t=np.asarray([hit["time_s"]])
    _,_,base_ddq=v.dense.analytic_reference(t,q0,4,b)
    _,_,basis_ddq=v.qp.basis(t,2.95,.4)
    j=v.chain.G1_JOINT_ORDER.index(hit["joint"])
    measured=(base_ddq+basis_ddq@coeff)[0,j]
    assert abs(measured-hit["value"])<1e-7
    assert abs(measured)>40.005+1e-8
    assert 2.9<=hit["time_s"]<=3.
    assert scan["floating_point_conservative_not_formally_verified"]
