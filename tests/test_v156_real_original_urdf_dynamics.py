"""Original 29D URDF-source 35D CPU matrices, kinematics and adversarial gates."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pytest
from v156_real_urdf_pin_model import (
    OriginalUrdfModel, frozen, independent_invariants, independent_time_derivative, urdf_path,
)
from v156_frozen_urdf_campaign import run_frozen

@pytest.fixture(scope="module")
def model():
    return OriginalUrdfModel()

def cases():
    cfg=frozen()
    return [(p,m,s) for p in cfg["phases"] for m in cfg["modes"] for s in cfg["seeds"]]

@pytest.mark.parametrize("phase,mode,seed",cases())
def test_each_frozen_original_urdf_model_dynamics_identity(model,phase,mode,seed):
    q,v=model.fixture(phase,mode,seed)
    out=model.evaluate(q,v)
    result=independent_invariants(model,q,v,out)
    assert result["max_rnea_consistency_error"] < 1e-7
    assert result["max_aba_error"] < 1e-7
    assert result["min_M_eigenvalue"] > 1e-7
    assert result["wrist_hip_link_origin_distance_m"] > 0
    assert out["source_kind"] == "original_g1_urdf_offline"
    assert out["can_issue_control"] is False
    assert out["can_certify_physx"] is False
    assert out["can_certify_grf_cop"] is False
    assert out["can_certify_wrist_hip_manifold"] is False

@pytest.mark.parametrize("phase,mode,seed",cases()[:12])
def test_independent_numerical_jdotv_at_original_full35(model,phase,mode,seed):
    q,v=model.fixture(phase,mode,seed)
    obs=model.evaluate(q,v)
    error=independent_time_derivative(model,q,v,obs)
    assert max(error.values()) < 2e-5
    assert len(error)==4

def test_source_joint_order_actual_not_pin_internal(model):
    q29=np.linspace(-.05,.05,29);v29=np.linspace(-.03,.03,29)
    q=model.pose(q29);v=model.speed(v29)
    assert np.array_equal(q[model.idxq],q29)
    assert np.array_equal(v[model.idxv],v29)
    assert set(model.idxv)==set(range(6,35))
    assert list(model.idxv)!=list(range(6,35))
    assert model.total_mass_urdf_kg == pytest.approx(33.341142022,abs=1e-9)

@pytest.mark.parametrize("vector_kind",["bad_q29","bad_v29","bad_basepos","bad_quaternion","nonunit_quaternion"])
def test_source_pose_speed_rejects_invalid(model,vector_kind):
    if vector_kind=="bad_q29":
        with pytest.raises(ValueError,match="Q29"):model.pose(np.zeros(28))
    elif vector_kind=="bad_v29":
        with pytest.raises(ValueError,match="QDOT29"):model.speed(np.zeros(31))
    elif vector_kind=="bad_basepos":
        with pytest.raises(ValueError,match="BASE_POS"):model.pose(np.zeros(29),xyz=[0,0,np.nan])
    elif vector_kind=="bad_quaternion":
        with pytest.raises(ValueError,match="BASE_QUAT"):model.pose(np.zeros(29),xyzw=[0,0,0])
    else:
        with pytest.raises(ValueError,match="NONUNIT"):model.pose(np.zeros(29),xyzw=[0,0,0,.6])

@pytest.mark.parametrize("corruption",["mass","bias","jdotv"])
def test_independent_check_detects_numerical_forgery(model,corruption):
    q,v=model.fixture("swing","left",2)
    o=model.evaluate(q,v)
    if corruption=="mass":
        o["M"][0,0]+=10
        with pytest.raises(AssertionError,match="RNEA"): independent_invariants(model,q,v,o)
    elif corruption=="bias":
        o["h"][0]+=2
        with pytest.raises(AssertionError,match="RNEA"): independent_invariants(model,q,v,o)
    else:
        o["Jdotv6"]["left_ankle_roll_link"][0]+=1
        with pytest.raises(AssertionError,match="FD_FAIL"): independent_time_derivative(model,q,v,o)

def test_urdf_sha_is_locked_and_rejects_substitution(tmp_path):
    original=urdf_path()
    altered=tmp_path/"not_original.urdf"
    altered.write_bytes(original.read_bytes()+b"\n<!-- tampered -->\n")
    with pytest.raises(ValueError,match="URDF_SHA_MISMATCH"):OriginalUrdfModel(source=altered)

def test_frozen_campaign_all_counts_and_no_false_succcess():
    out=run_frozen()
    assert out["full_35d_model_checks"]==48
    assert out["finite_difference_6d_kinematic_checks"]==12
    assert out["max_identity_error"]<1e-7
    assert out["max_6d_Jdotv_fd_error"]<2e-5
    assert out["new_real_physx_episode_count"]==0
    assert out["legal_high_speed_instep_episodes"]==0
    assert out["native_left_wrist_hip_contact_manifold"]=="UNKNOWN"
    assert out["actual_original_physx_historical_audit"]["actual_original_200hz_states"]==6628
    assert all(x["real_physx_success"] is None for x in out["cases"])

def test_undocumented_fixture_denied(model):
    with pytest.raises(ValueError,match="NOT_PREDECLARED"): model.fixture("strike", "left",0)

def test_pin_model_does_not_mutate_original_urdf(model):
    assert hashlib.sha256(Path(model.source).read_bytes()).hexdigest()==frozen()["source_urdf_sha256"]
