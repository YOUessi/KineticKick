"""v135 fullbody 200Hz contract refuses unmeasured dynamics by construction."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v135_chain29_offline as motion
import v135_reference_admission_bridge as admission
from v107_motor_admission import Limits29, Witness
import v127_collider_proximity as geom

U=Path("/home/you/projects/kinetickick/configs/paid_original/paid_main_resolved.urdf")


@pytest.fixture(scope="module")
def generated():
    cfg=motion.frozen()
    l=motion.source_limits(U,cfg["original_urdf_sha256"])
    p=motion.generate(cfg,(l["lo"]+l["hi"])/2,2)
    return p,l


def test_missing_all_physical_sources_rejects_not_even_a_torque_guess(generated):
    p,l=generated
    r=admission.inspect_reference_interface(p)
    assert r["status"]=="BLOCKED_MISSING_PHYSICAL_EVIDENCE"
    assert len(r["missing_sources"])==5
    assert "MEASURED_TORQUE" in r["missing_sources"]
    assert "PHYSX_SUPPORT_COLLISION_WITNESSES" in r["missing_sources"]
    assert not r["motor_execution_authorized"]


def test_faked_source_label_or_invalid_29_shape_never_admitted(generated):
    p,l=generated
    with pytest.raises(ValueError,match="UNVERIFIED_OR_MUTATED_REFERENCE_ORIGIN"):
        admission.inspect_reference_interface(replace(p,source="QUALIFIED_REAL_HUMAN_WHOLEBODY_WITH_IMPACT"))
    with pytest.raises(ValueError,match="MISSING_ORIGINAL29_COMPLETE_TRAJECTORY"):
        admission.inspect_reference_interface(replace(p,q=p.q[:5]))


def test_partial_inputs_still_fail_closed(generated):
    p,l=generated
    r=admission.inspect_reference_interface(p,measured_torque=np.zeros_like(p.q))
    assert not r["motor_execution_authorized"]
    assert "MEASURED_TORQUE" not in r["missing_sources"]
    assert "PHYSX_SUPPORT_COLLISION_WITNESSES" in r["missing_sources"]


def test_fake_complete_witnesses_do_not_pass_source_gate(generated):
    p,l=generated
    lim=Limits29(q_lo=l["lo"].tolist(),q_hi=l["hi"].tolist(),
                 max_qdot=l["vmax"].tolist(),max_qddot=[1000.]*29,
                 max_abs_torque=l["taumax"].tolist())
    fake=Witness(age_seconds=0.,native_pair_certified=True,
                 signed_collision_clearance_m=.1,
                 support_certified=True,stance_foot_normal_force_N=100.,
                 stance_foot_slip_mps=0.,body_geometry_certified=True,
                 provenance_sha256="f"*64)
    c=geom.frozen()
    r=admission.inspect_reference_interface(
        p,measured_torque=np.zeros_like(p.q),
        calibrated_limits=lim,physical_witnesses=[fake]*2800,
        original_state=(p.q[0],p.dq[0]),
        urdf_path=U,geometry_cfg=c)
    assert r["status"]=="OFFLINE_REVIEW_ONLY"
    assert r["screen"]["motor_execution_authorized"] is False
    assert r["screen"]["physical_motor_command_issued"] is False
    assert r["source_is_not_verified_human_high_performance_trajectory"]
    assert "UNQUALIFIED_HUMAN_ACTION_SOURCE" in r["screen"]["total_refusal_reasons"]


def test_forged_torque_wrong_dimension_rejected(generated):
    p,l=generated
    with pytest.raises(ValueError,match="INVALID_29DOF_MEASURED_TORQUE"):
        admission.inspect_reference_interface(
            p,measured_torque=np.zeros((2800,28)),
            calibrated_limits=object(),physical_witnesses=[None]*2800,
            original_state=(p.q[0],p.dq[0]),urdf_path=U,
            geometry_cfg=geom.frozen())
