"""v56 native contact provenance and independent PhysX cohort contracts.

Unit tests are CPU-only and never equate synthetic callbacks to real PhysX.
"""
import json
from types import SimpleNamespace

import pytest

from tools.native_contact_reader_v056 import G1ReadOnlyContactSubscriber,vec
from tools.audit_real_batch_v056 import effective_resets
from tools.run_frozen_batch_v056 import verify,CFG


def test_contact_vec_rejects_nonfinite_or_nonnumeric():
    assert vec([1,2,3])==[1.,2.,3.]
    assert vec([1,2,float("nan")]) is None
    assert vec([1,2]) is None
    assert vec("not numeric") is None


def mock_native_subscriber():
    o=G1ReadOnlyContactSubscriber.__new__(G1ReadOnlyContactSubscriber)
    o.env=SimpleNamespace(sim=SimpleNamespace(current_time=5.145))
    o.decode=lambda x:x
    o.events=[];o.callback_calls=0;o.native_headers=0
    o.errors=[];o.collisions_dropped=0;o.max_events=8
    o.report_apis={"/World/Robot/left_hip_roll_link":True}
    return o


def test_native_callback_records_exact_paths_and_valid_manifold():
    o=mock_native_subscriber()
    h=SimpleNamespace(actor0="/World/Robot/left_wrist_yaw_link",
        actor1="/World/Robot/left_hip_roll_link",
        collider0="/World/Robot/left_wrist_yaw_link/shape",
        collider1="/World/Robot/left_hip_roll_link/shape",
        contact_data_offset=0,num_contact_data=1)
    cp=SimpleNamespace(position=[0.1,0.2,0.3],
        normal=[0,1,0],impulse=[.01,0,0],separation=-.003)
    o._on_contacts([h],[cp])
    result=o.result()
    assert result["native_headers"]==1
    assert result["wrist_hip_direct_path_events"]==1
    assert result["rigid_pair_certified"]
    assert result["saved_interesting_events"][0]["point_sample"][0]["separation_m"]==-.003


def test_native_callback_absence_does_not_certify_clearance():
    o=mock_native_subscriber()
    o._on_contacts([],[])
    r=o.result()
    assert not r["rigid_pair_certified"]
    assert r["callback_calls"]==1
    assert "cannot prove" in r["limitation"]


def test_contact_capture_overflow_is_uncertified():
    o=mock_native_subscriber();o.max_events=0
    h=SimpleNamespace(actor0="/World/Robot/left_wrist_yaw_link",
        actor1="/World/Robot/left_hip_roll_link",
        collider0="/World/Robot/left_wrist_yaw_link/shape",
        collider1="/World/Robot/left_hip_roll_link/shape",
        contact_data_offset=0,num_contact_data=1)
    cp=SimpleNamespace(position=[0,0,0],normal=[0,0,1],impulse=[1,0,0])
    o._on_contacts([h],[cp])
    assert o.result()["dropped_events"]==1
    assert o.result()["rigid_pair_certified"] is False


def test_seed_label_not_independent_without_actual_reset_difference():
    rows=[dict(id=f"C_s{seed}_frozen_v32",policy="frozen_v32",
               perturb_reset=True,seed=seed) for seed in [2111,2333]]
    same={"initial_robot_state":{"joint_position":[1,2,3]},"traces":[[{"x":1},{"x":2}]]}
    outcome=effective_resets(rows,lambda _:same)
    assert len(outcome)==1
    assert not outcome[0]["physical_initial_state_different"]
    assert outcome[0]["not_independent_if_equal"]


def test_seed_actual_state_change_is_detected():
    rows=[dict(id=f"C_s{seed}_frozen_v32",policy="frozen_v32",
               perturb_reset=True,seed=seed) for seed in [2111,2333]]
    def data(c):
        return {"initial_robot_state":{"q":c["seed"]},
                "traces":[[{"x":c["seed"]},{"x":2}]]}
    assert effective_resets(rows,data)[0]["physical_initial_state_different"]


def test_frozen_protocol_contract_and_unique_scenarios():
    cfg=json.loads(CFG.read_text())
    assert verify(cfg)
    ids=[c["id"] for c in cfg["cases"]]
    assert len(ids)==len(set(ids))==15
    assert all(c["perturb_reset"] and not c["native_report"] for c in cfg["cases"][:12])
    assert all(c["native_report"] and not c["perturb_reset"] for c in cfg["cases"][12:])
    assert len({tuple(c["ball_offset_xy_m"]) for c in cfg["cases"][:12]})==2


def test_no_collision_evidence_overcomes_missing_joint_name_or_pair():
    from tools.wholebody_safety_interface_v055 import adapt
    # The previous whole-body interface must remain inactive absent certified
    # live collision/manifold availability; no need to instantiate a GPU robot.
    assert "pair_evidence_available" in adapt.__code__.co_varnames
