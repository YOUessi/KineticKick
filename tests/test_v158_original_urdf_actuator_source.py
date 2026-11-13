"""v158 frozen original G1 URDF actuation (never runtime physics) tests."""
from __future__ import annotations
import copy
import hashlib
import itertools
import numpy as np
import pytest

from v156_real_urdf_pin_model import OriginalUrdfModel,independent_invariants
from v158_source_actuation_wrench import (
    frozen,make_case,urdf_joint_limits,solve,validate_case,
    independently_audit_solution,phase_reference)
from v158_frozen_actuation_campaign import damage

@pytest.fixture(scope="module")
def model():
    return OriginalUrdfModel()

def test_frozen_protocol_exact_denominators():
    c=frozen()
    assert (len(c["phases"])*len(c["support_modes"])*len(c["seeds"])==48)
    assert len(c["effort_conditions"])==2
    assert c["expected_qp_solutions"]==96
    assert c["expected_negative_rejections"]==16
    assert c["no_runtime_control_authority"]
    assert c["no_elite_human_training_claim"]

def test_original29_limits_source_and_order(model):
    lim=urdf_joint_limits(model)
    assert lim.shape==(29,4)
    assert lim[:,2].min()==5
    assert lim[:,2].max()==139
    assert lim[:,3].min()==20
    assert lim[:,3].max()==37
    assert model.sha==hashlib.sha256(model.source.read_bytes()).hexdigest()
    assert list(model.idxv)!=list(range(6,35))

@pytest.mark.parametrize("phase,mode,seed,condition",
    list(itertools.product(
        frozen()["phases"],frozen()["support_modes"],frozen()["seeds"],frozen()["effort_conditions"]
    )))
def test_every_frozen_full35d_qp_source_model_and_independent_audit(model,phase,mode,seed,condition):
    p=make_case(model,phase,mode,seed,condition)
    result=solve(p,model)
    assert result["source_urdf_sha256"]==model.sha
    assert result["cpu_optimization_not_PhysX"] is True
    assert result["can_issue_control"] is False
    assert result["human_elite_learned"] is False
    assert result["ground_reaction_or_CoP_certified"] is False
    assert result["actual_contact_manifold"] is None
    assert result["strict_instep_success"] is None
    audit=independently_audit_solution(p,result,model)
    if result["solved_numerically"]:
        assert audit["audited"]
        assert audit["max_dynamics_residual"]<.008
        assert audit["max_hypothetical_6d_stance_residual"]<.008
        assert audit["max_inequality_violation"]<.008
    else:
        assert result["status"]
    if condition=="urdf_effort":
        assert np.array_equal(p["torque_caps"],np.asarray(p["source_limits"])[:,2])
    else:
        assert np.array_equal(p["torque_caps"],np.asarray(p["source_limits"])[:,2]*100)

@pytest.mark.parametrize("phase",frozen()["phases"])
def test_joint_acceleration_is_full_body_and_not_old_one_mrad(model,phase):
    r=phase_reference(frozen(),phase,2)
    assert r.shape==(29,)
    assert np.any(abs(r[:12])>1.)
    assert np.any(abs(r[12:15])>.1)
    assert np.any(abs(r[15:])>.1)
    assert np.isfinite(r).all()

@pytest.mark.parametrize("kind",frozen()["negative_kinds"])
@pytest.mark.parametrize("seed",[0,1])
def test_bad_source_inputs_cannot_be_relabelled_real(model,kind,seed):
    valid=make_case(model,"approach","left",seed,"urdf_effort")
    with pytest.raises(ValueError):
        validate_case(damage(valid,kind),model)

@pytest.mark.parametrize("phase,mode",itertools.product(
    frozen()["phases"],frozen()["support_modes"]))
def test_freefloating_dynamics_source_not_plausibility_only(model,phase,mode):
    p=make_case(model,phase,mode,0,"urdf_effort")
    q,v=np.asarray(p["q36"]),np.asarray(p["v35"])
    m=model.evaluate(q,v)
    result=independent_invariants(model,q,v,m)
    assert result["max_aba_error"]<1e-7
    assert result["max_rnea_consistency_error"]<1e-7
    assert m["can_certify_physx"] is False

def test_fake_grf_pair_or_runtime_permission_rejected(model):
    original=make_case(model,"approach","double",0,"urdf_effort")
    for key in ("native_contact_pair_certified","runtime_ground_force_certified","can_issue_control"):
        p=copy.deepcopy(original)
        p[key]=True
        with pytest.raises(ValueError,match="UNCERTIFIED_RUNTIME_CLAIM"):
            validate_case(p,model)

def test_unapproved_phase_and_impossible_effort_denied(model):
    with pytest.raises(ValueError,match="UNDECLARED_CASE"):
        make_case(model,"celebrate","left",0,"urdf_effort")
    with pytest.raises(ValueError,match="UNDECLARED_CASE"):
        make_case(model,"swing","left",0,"no_effort")

@pytest.mark.parametrize("kind",[
    "counterfeit_mass", "counterfeit_bias", "counterfeit_jacobian",
    "counterfeit_jdotv", "counterfeit_sole_envelope", "changed_phase_reference"
])
def test_strict_urdf_source_contract_rejects_forged_matrices_and_references(model,kind):
    p=make_case(model,"swing","left",2,"urdf_effort")
    if kind=="counterfeit_mass":p["M"][2][2]+=1
    if kind=="counterfeit_bias":p["h"][3]+=1
    if kind=="counterfeit_jacobian":p["contacts"][0]["J6"][2][4]+=.1
    if kind=="counterfeit_jdotv":p["contacts"][0]["Jdotv6"][1]+=.1
    if kind=="counterfeit_sole_envelope":p["contacts"][0]["envelope_xyz"][0]+=.1
    if kind=="changed_phase_reference":p["a_ref29"][3]+=.1
    with pytest.raises(ValueError,match="V158_"):
        validate_case(p,model)
