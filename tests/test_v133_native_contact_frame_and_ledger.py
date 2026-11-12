"""v133 CPU isolation + authentic 200Hz provenance, never GPU/actuator tests."""
from __future__ import annotations

from pathlib import Path
import copy
import sys
from types import SimpleNamespace as N
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v133_native_contact_frame as n
import v133_physical_event_ledger as h


def fixture():
    header = N(
        actor0="A", actor1="B", collider0="C", collider1="D",
        contact_data_offset=0, num_contact_data=1, stage_id=9,
        type="CONTACT_FOUND")
    point = N(position=[.1,.2,.3], normal=[0,0,1],
              impulse=1.5, separation=-.0004)
    decoder = {
        "A": "/World/Robot/left_wrist_yaw_link",
        "B": "/World/Robot/left_hip_roll_link",
        "C": "/World/Robot/left_wrist_yaw_link/collision",
        "D": "/World/Robot/left_hip_roll_link/collision"}
    return header, point, decoder


def native_packet():
    a,b,decode = fixture()
    return n.copy_native_callback([a], [b], decode_path=decode.__getitem__,
                                  step=1028, stage_id=9)


def test_protocol_frozen_sha_and_hard_gates():
    cfg=n.frozen()
    assert cfg["frozen_success_gates"]["instep_impulse_fraction_min"]==.95
    assert cfg["frozen_success_gates"]["ball_speed_mps_min"]==3.5
    assert not cfg["no_edit_existing_provenance"] is False


def test_copies_native_borrowed_buffer_and_no_false_cert():
    hdr, point, dec = fixture()
    p = n.copy_native_callback([hdr], [point], decode_path=dec.__getitem__,
                               stage_id=9, step=1028)
    point.position[0]=1000
    hdr.collider0="z"
    assert p["headers"][0]["points"][0]["position"]==[.1,.2,.3]
    assert p["headers"][0]["colliders"][0].startswith("/World")
    assert p["all_native_contacts_covered"]
    assert p["native_header_count"]==1
    assert p["headers"][0]["points"][0]["impulse"]==1.5
    assert p["native_actor_collider_pair_certified"] is False
    assert p["motor_execution_authorized"] is False
    assert p["same_step_original29_state_certified"] is False


@pytest.mark.parametrize("bad", [0,-1,1.0,True,None])
def test_bad_callback_step_rejected(bad):
    a,b,d=fixture()
    with pytest.raises(ValueError,match="STEP"):
        n.copy_native_callback([a],[b],decode_path=d.__getitem__,step=bad)


@pytest.mark.parametrize("bad", [-.01,0.,1.,float("nan")])
def test_non200hz_clock_rejected(bad):
    a,b,d=fixture()
    with pytest.raises(ValueError):
        n.copy_native_callback([a],[b],decode_path=d.__getitem__,
                               step=5,dt_s=bad)


def test_wrong_stage_refused():
    a,b,d=fixture()
    with pytest.raises(ValueError,match="STAGE"):
        n.copy_native_callback([a],[b],decode_path=d.__getitem__,
                               step=2,stage_id=8)


@pytest.mark.parametrize("offset,count", [(1,1),(0,2),(-1,1)])
def test_contact_slice_overflow_refused(offset,count):
    a,b,d=fixture()
    a.contact_data_offset=offset
    a.num_contact_data=count
    with pytest.raises(ValueError,match="INDEX|BOUNDS"):
        n.copy_native_callback([a],[b],decode_path=d.__getitem__,step=2)


def test_header_slices_cannot_overlap():
    a,b,d=fixture()
    a2=copy.deepcopy(a)
    with pytest.raises(ValueError,match="OVERLAPPING"):
        n.copy_native_callback([a,a2],[b],decode_path=d.__getitem__,step=2)


@pytest.mark.parametrize("field,val,err", [
    ("position",[1,2],"VECTOR"),("normal",[0,0,0],"NORMAL"),
    ("impulse",float("nan"),"NONFINITE"),("impulse",-1,"NEGATIVE"),
    ("separation",float("inf"),"NONFINITE")])
def test_bad_contact_manifold_rejected(field,val,err):
    a,b,d=fixture()
    setattr(b,field,val)
    with pytest.raises(ValueError,match=err):
        n.copy_native_callback([a],[b],decode_path=d.__getitem__,step=2)


@pytest.mark.parametrize("field,token", [("actor1","A"),("collider1","C")])
def test_same_actor_or_collider_not_valid(field,token):
    a,b,d=fixture()
    setattr(a,field,token)
    with pytest.raises(ValueError,match="DUPLICATE"):
        n.copy_native_callback([a],[b],decode_path=d.__getitem__,step=2)


@pytest.mark.parametrize("path", ["", "relative", "/bad//path", "/bad/../path", "/bad/","/space path"])
def test_reject_unqualified_absolute_usd_path(path):
    with pytest.raises(ValueError,match="PATH"):
        n.canonical_path(path)


def test_no_substring_owner_inference():
    p=native_packet()
    assert n.candidate_links(p,allowed_owners={})==[]
    colliders=p["headers"][0]["colliders"]
    owners={colliders[0]:"left_wrist_yaw_link",colliders[1]:"left_hip_roll_link"}
    row=n.candidate_links(p,allowed_owners=owners)[0]
    assert row["suspected_only"] and not row["native_pair_certified"]
    assert row["points"]==1


def test_forged_calibration_is_not_accepted_as_native_proof():
    p=native_packet()
    p["backend_clock_calibrated"]=True
    from wholebody_safety_interface_v055 import G1_JOINT_ORDER
    st=n.state_snapshot(1028,[0]*29,[0]*29,expected_order=G1_JOINT_ORDER)
    cfg={k:True for k in ("native_positive","native_negative","callback_timing",
                           "off_on_state_invariance","owner_map_verified")}
    result=n.reconcile_callback_and_state(p,st,calibrations=cfg)
    assert result["checks_passed"]
    assert not result["native_actor_collider_pair_certified"]
    assert not result["motor_execution_authorized"]
    assert result["source_attestation_required"]


def test_unsynced_event_step_is_refused():
    p=native_packet()
    st=n.state_snapshot(1027,[0]*29,[0]*29)
    result=n.reconcile_callback_and_state(p,st,calibrations={})
    assert "CALLBACK_STATE_STEP_MISMATCH" in result["reasons"]
    assert "BACKEND_CALLBACK_CLOCK_UNCALIBRATED" in result["reasons"]


def test_empty_callback_does_not_prove_no_collision():
    p=n.copy_native_callback([],[],decode_path=lambda _: "/World/A",step=1)
    s=n.state_snapshot(1,[0]*29,[0]*29)
    result=n.reconcile_callback_and_state(p,s,calibrations={})
    assert "NO_NATIVE_CONTACT_HEADERS" in result["reasons"]
    assert not result["native_actor_collider_pair_certified"]


@pytest.mark.parametrize("bad", [float("nan"),float("inf")])
def test_nonfinite_29dof_state_rejected(bad):
    q=[0.]*29
    q[8]=bad
    with pytest.raises(ValueError,match="NONFINITE"):
        n.state_snapshot(1,q,[0.]*29)


def test_missing_or_wrong_joint_order_refused():
    from wholebody_safety_interface_v055 import G1_JOINT_ORDER
    with pytest.raises(ValueError,match="JOINT_ORDER"):
        n.state_snapshot(1,[0]*29,[0]*29,expected_order=["x"]*29)
    with pytest.raises(ValueError,match="JOINT_ORDER"):
        n.state_snapshot(1,[0]*29,[0]*29,expected_order=list(reversed(G1_JOINT_ORDER)))
    with pytest.raises(ValueError,match="29DOF"):
        n.state_snapshot(1,[0]*28,[0]*29)
    packet=native_packet()
    blank=n.state_snapshot(1028,[0]*29,[0]*29)
    result=n.reconcile_callback_and_state(packet,blank,calibrations={})
    assert "ORIGINAL_29_JOINT_ORDER_UNVERIFIED" in result["reasons"]


def test_callback_recorder_preserves_error_not_crash():
    a,b,d=fixture()
    rec=n.LiveCallbackCopier(clock=lambda:1,stage_id=99,decode_path=d.__getitem__)
    rec([a],[b])
    assert len(rec.packets)==0 and rec.errors
    assert "MISMATCH" in rec.errors[0]


def test_original_6628_real_frames_with_raw_sha_and_censored_5145():
    summary=h.real_event_ledger()
    assert summary["real_physics_states_reaudited"]==6628
    assert summary["new_physx_episodes"]==0
    assert not summary["direct_original_wrist_hip_manifold_certified"]
    f=summary["cases"]["cmu_10_05"]
    assert f["real_200hz_steps"]==1028
    assert f["non_physics_terminal_rows_removed"]==1
    assert f["pre_event_nonfoot_peak_t_s"]==5.135
    assert f["last_real_state_before_named_contact"]["time_s"]==5.14
    assert not f["event_5p145_has_real_29dof_step"]
    assert summary["cases"]["frozen_v32"]["real_200hz_steps"]==2800
    assert summary["cases"]["cmu_10_02"]["real_200hz_steps"]==2800
    assert all(v["not_verified_physx_actor_pair"] for v in summary["cases"].values())
