"""Strict pair identity and genuine contact-point tests (no Isaac dependency)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from audit_v058_cpu_contact_truth import native_pair_truth

def fixture():
    event=dict(time_s=1.665,
        actor_paths=["/World/envs/env_0/Robot/pelvis","/World/envs/env_0/Robot/right_hip_roll_link"],
        collider_paths=["/World/envs/env_0/Robot/pelvis/collisions","/World/envs/env_0/Robot/right_hip_roll_link/collisions"],
        reported_points=1,
        point_sample=[dict(point_w=[0.1,0.2,0.3],normal_w=[1.,0.,0.],separation_m=-0.01)])
    return dict(physx_native_report_v056=dict(errors=[],dropped_events=0,
        report_capture_complete=True,saved_interesting_events=[event]))

def test_real_two_actor_manifold_qualified():
    result=native_pair_truth(fixture(),{"pelvis","right_hip_roll_link"})
    assert len(result)==1 and result[0]["separation_m"]==-0.01

def test_no_confused_wrist_hip_identity():
    assert native_pair_truth(fixture(),{"left_wrist_yaw_link","left_hip_roll_link"})==[]

def test_missing_point_not_certified():
    f=fixture()
    f["physx_native_report_v056"]["saved_interesting_events"][0]["point_sample"]=[]
    assert not native_pair_truth(f,{"pelvis","right_hip_roll_link"})

def test_reporting_errors_fail_closed():
    f=fixture()
    f["physx_native_report_v056"]["errors"]=["buffer mismatch"]
    assert not native_pair_truth(f,{"pelvis","right_hip_roll_link"})

def test_other_world_pair_not_g1():
    f=fixture()
    f["physx_native_report_v056"]["saved_interesting_events"][0]["actor_paths"][0]="/World/Probe/A"
    assert not native_pair_truth(f,{"pelvis","right_hip_roll_link"})

def test_v058_forwarding_mandatory():
    source=(Path(__file__).resolve().parents[1]/"tools/run_placement_strike_v058.py").read_text()
    assert "sys.argv+=['--sim-device',a.sim_device]" in source
    assert "V058_NATIVE_RETURN" in source
