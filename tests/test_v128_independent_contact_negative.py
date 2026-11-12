#!/usr/bin/env python3
"""Independent geometry cross-validation and net-force false-positive controls."""
import copy
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v128_v54_event_geometry as audit
import v128_independent_scipy_geometry as independent


def example_force():
    cfg=audit.frozen_protocol()
    f=np.array([25.89015328949694,0,0])
    raw={"nonfoot_contact_diagnostics":[
        {"body":"left_wrist_yaw_link","force_w_N":f.tolist(),
         "time_s":cfg["true_nonfoot_event_t_s"],"position_w":[0,0,0]},
        {"body":"left_hip_roll_link","force_w_N":(-f).tolist(),
         "time_s":cfg["true_nonfoot_event_t_s"],"position_w":[.1,0,0]},
    ]}
    witness={"left_wrist_hip_symmetric_force_pair":{
        "body_pair_directly_measured_in_contact_manifold":False,
        "force_magnitude_each_N":cfg["true_nonfoot_force_N"]}}
    return cfg,raw,witness


def test_equal_opposite_net_forces_never_native_contact_certification():
    cfg,raw,witness=example_force()
    a=audit.certify_force_pair(raw,witness,cfg)
    assert a["net_force_pair_equal_opposite"]
    assert a["native_pair_owner_unknown"]
    assert a["force_is_link_net_not_direct_contact_manifold"]


@pytest.mark.parametrize("field,value",[
    ("body","some_other_link"),
    ("force_w_N",[0,0,0]),
    ("time_s",5.13),
    ("position_w",[float("nan"),0,0]),
])
def test_forged_or_misplaced_sensor_evidence_is_rejected(field,value):
    cfg,raw,witness=example_force()
    raw["nonfoot_contact_diagnostics"][0][field]=value
    with pytest.raises((ValueError,TypeError)):
        audit.certify_force_pair(raw,witness,cfg)


def test_forged_direct_contact_certification_refused():
    cfg,raw,witness=example_force()
    witness["left_wrist_hip_symmetric_force_pair"]["body_pair_directly_measured_in_contact_manifold"]=True
    with pytest.raises(ValueError,match="FALSELY_CERTIFIED"):
        audit.certify_force_pair(raw,witness,cfg)


def test_missing_sensor_body_refused():
    cfg,raw,witness=example_force()
    raw["nonfoot_contact_diagnostics"]=[]
    with pytest.raises(ValueError,match="MISSING"):
        audit.certify_force_pair(raw,witness,cfg)


def test_independent_scipy_segment_minimizer_synthetic():
    p0,p1=np.array([0.,0.,0.]),np.array([0.,0.,1.])
    q0,q1=np.array([.11,0,0]),np.array([.11,0,1.])
    assert independent.independent_segment_gap(p0,p1,q0,q1,(.05,.05))==pytest.approx(.01,abs=1e-8)


def test_9_real_v54_original_g1_geometry_states_independently_agree():
    out=independent.run(audit.frozen_protocol())
    assert out["original_G1_samples_crosschecked"]==9
    assert out["maximum_difference_m"]<2e-7
    assert not any(x["native_contact_pair_certified"] for x in out["results"])
    failed=[x for x in out["results"] if x["case"]=="cmu_10_05"]
    assert min(x["original_URDF_capsule_lower_bound_m"] for x in failed)<.0001
