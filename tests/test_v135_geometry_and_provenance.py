"""v135 independent geometry, pre-registration, and physics-truth refusal."""
from pathlib import Path
import json
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v135_chain29_offline as m

U=Path("/home/you/projects/kinetickick/configs/paid_original/paid_main_resolved.urdf")

@pytest.fixture(scope="module")
def audit():
    cfg=m.frozen()
    limits=m.source_limits(U,cfg["original_urdf_sha256"])
    tree=m.geom.parse_tree(U)
    q0=(limits["lo"]+limits["hi"])/2
    return [m.inspect(m.generate(cfg,q0,f),limits,tree,cfg) for f in cfg["multipliers"]]

def test_full_grid_five_frozen_cases_and_not_fake_rollouts(audit):
    assert [x["multiplier"] for x in audit]==[0.,1.,2.,4.,-1.]
    assert sum(x["proposal_200hz_frames"] for x in audit)==14000
    assert sum(x["true_physx_rollouts"] for x in audit)==0
    assert all(not x["full_kick_strict_physics_pass"] for x in audit)

def test_five_groups_nonzero_only_when_multiplier_nonzero(audit):
    assert audit[0]["all_groups_move_above_1mrad"] is False
    assert all(x["all_groups_move_above_1mrad"] for x in audit[1:])

def test_all_four_predefined_original_collider_pairs_scanned(audit):
    cfg=m.frozen()
    expected={"__".join(p) for p in cfg["geometry_pairs"]}
    for x in audit:
        assert set(x["original_urdf_four_pair_geometry"])==expected
        assert all(np.isfinite(y["min_lower_bound_m"]) for y
                   in x["original_urdf_four_pair_geometry"].values())

def test_no_geometry_based_contact_authentication(audit):
    for x in audit:
        assert "UNKNOWN_NATIVE_PHYSX_ACTOR_COLLIDER_PAIR" in x["unique_refusal_reasons"]
        assert "NO_REAL_STANCE_GRF_COP_OR_SLIP" in x["unique_refusal_reasons"]
        assert "NO_MEASURED_FLOATING_BASE_COM" in x["unique_refusal_reasons"]
        assert "NO_TRUE_BALL_SPEED_INSTEP_IMPULSE_OR_GOAL" in x["unique_refusal_reasons"]
        assert not x["native_contact_pair_certified"]
        assert not x["original29_motor_execution_authorized"]

def test_sign_reverse_is_kinematically_symmetric_but_not_geometrically_equal(audit):
    plus, minus = audit[1],audit[4]
    assert plus["max_original_qdot_limit_ratio"]==pytest.approx(minus["max_original_qdot_limit_ratio"])
    assert plus["peak_analytic_qddot_rad_s2"]==pytest.approx(minus["peak_analytic_qddot_rad_s2"])
    a=plus["original_urdf_four_pair_geometry"]
    b=minus["original_urdf_four_pair_geometry"]
    assert any(abs(a[k]["min_lower_bound_m"]-b[k]["min_lower_bound_m"])>1e-6 for k in a)

def test_four_x_requires_more_kinematics_than_nominal(audit):
    assert audit[3]["max_original_qdot_limit_ratio"] > audit[1]["max_original_qdot_limit_ratio"]
    for name in m.GROUPS:
        assert audit[3]["group_excursion_rad"][name]>audit[1]["group_excursion_rad"][name]

def test_local_record_consistent_with_frozen_actual_audit(audit):
    source=ROOT/"reports/v135_frozen_5case_chain29_audit_20261113.json"
    if not source.exists():
        pytest.skip("original CPU audit not mounted; this test does not invent evidence")
    saved=json.loads(source.read_text())
    assert saved["frozen_protocol_sha"]==m.FROZEN_SHA
    for a,b in zip(saved["trials"],audit):
        assert a["max_original_qdot_limit_ratio"]==b["max_original_qdot_limit_ratio"]
        assert a["original_urdf_four_pair_geometry"]==b["original_urdf_four_pair_geometry"]
    assert saved["actual_new_physx_episodes"]==0
    assert saved["actual_new_policy_strict_successes"]==0
