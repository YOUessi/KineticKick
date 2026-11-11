"""Original G1 native poll witness, safety, provenance and fail-closed gates."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v080_contact_poll import witness,vec3,ExistingNativeContactPoll
from v080_native_poll_batch import sha,CFG,CFG_SHA

def event(a="/World/G1/left_wrist_yaw_link",b="/World/G1/left_hip_roll_link",imp=1.):
    return dict(actor_paths=[a,b],collider_paths=[a+"/mesh",b+"/mesh"],contacts=[dict(
        point_w=[0.,1.,2.],normal_w=[0.,0.,1.],impulse_w=[0.,0.,imp],impulse_scalar=None)])

def test_full_report_pair_is_witness():assert witness(event())
def test_reversed_report_pair_is_witness():
    e=event();e["actor_paths"].reverse();e["collider_paths"].reverse();assert witness(e)
def test_two_names_in_same_actor_is_not_pair():
    e=event(a="/left_wrist_yaw_link/left_hip_roll_link",b="/other")
    assert not witness(e)
def test_mixed_actor_collider_slots_cannot_manufacture_pair():
    e=event(a="/left_wrist_yaw_link",b="/other")
    e["collider_paths"]=["/other","/left_hip_roll_link"]
    assert not witness(e)
def test_zero_impulse_is_not_witness():assert not witness(event(imp=0))
def test_absent_contact_point_is_not_witness():
    e=event();e["contacts"][0]["point_w"]=None;assert not witness(e)
def test_absent_normal_is_not_witness():
    e=event();e["contacts"][0]["normal_w"]=None;assert not witness(e)
def test_absent_wrist_is_not_witness():assert not witness(event(a="/right_wrist"))
def test_infinite_or_bad_vector_fails_closed():
    assert vec3([1,float("nan"),3]) is None
    assert vec3([1,2]) is None
    assert vec3("abc") is None

def test_true_poll_empty_is_unknown_not_exclusion():
    x=object.__new__(ExistingNativeContactPoll)
    x.start=4.7;x.end=5.5;x.frames_seen=100;x.poll_calls=50
    x.headers=0;x.contact_points=0;x.report_apis={};x.events=[]
    x.dropped_events=0;x.errors=[];x.times_polled=[4.7,5.5]
    result=x.result()
    assert not result["wrist_hip_direct_contact_certified"]
    assert result["negative_interpretation"]=="UNKNOWN_NO_NATIVE_WITNESS"
def test_truncated_capture_not_certified():
    x=object.__new__(ExistingNativeContactPoll)
    x.start=4.7;x.end=5.5;x.frames_seen=100;x.poll_calls=50
    x.headers=1;x.contact_points=1;x.report_apis={};x.events=[event()]
    x.dropped_events=1;x.errors=[];x.times_polled=[4.7,5.5]
    assert not x.result()["wrist_hip_direct_contact_certified"]

def test_protocol_sha_is_locked():assert sha(CFG)==CFG_SHA
def test_protocol_has_three_pair_trials_and_hard_gates():
    c=json.loads(CFG.read_text())
    assert len(c["trial_order"])==3
    assert c["max_physics_episodes"]==6
    assert c["hard_gates"]["instep_impulse_fraction_min"]==.95
    assert c["hard_gates"]["recovery_required"] is True
    assert c["required_g1_joints"]==29 and c["physical_dt_s"]==.005

def test_no_usd_or_control_writes_in_observer():
    s=(ROOT/"tools/v080_contact_poll.py").read_text()
    for forbidden in ("API.Apply(",".CreateAttribute(",".set_joint_position_target(",".set_joint_velocity_target(",".set_world_pose("):
        assert forbidden not in s
def test_runner_stops_on_pair_physics_change():
    s=(ROOT/"tools/v080_native_poll_batch.py").read_text()
    assert "READ_ONLY_PARITY_VIOLATED_STOP" in s
    assert "GPU_MUTEX_ALREADY_LOCKED" in s
    assert "DISK_BELOW_20GIB" in s
    assert "WRITE_ONCE_COLLISION" in s
def test_original_child_controller_not_modified():
    c=json.loads(CFG.read_text())
    assert sha(ROOT/"tools/v076_run_placement_strike.py")==c["source_unchanged_sha256"]["tools/v076_run_placement_strike.py"]
    assert sha(ROOT/"tools/v076_eval_guarded.py")==c["source_unchanged_sha256"]["tools/v076_eval_guarded.py"]
