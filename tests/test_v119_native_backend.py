import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v119_native_backend_batch import parity,diagnose,CONFIG

def test_frozen_full_two_backend_pairs():
    x=json.loads(CONFIG.read_text())
    names=[a["name"] for a in x["matrix"]]
    assert len(names)==len(set(names))==6
    assert names[:3]==["cpu_positive_off","cpu_positive_on","cpu_separated_on"]
    assert names[3:]==["cuda_positive_off","cuda_positive_on","cuda_separated_on"]
    assert x["original_G1_motor_changed"] is False
    assert x["min_free_gib"]==20 and x["timeout_seconds"]==120

def test_identical_readonly_pairs():
    a={"device":"cpu","runtime_z_vz":[[1,.92,0.1]],"box_position_w":[0,0,0]}
    assert parity(a,dict(a))["recorded_pose_velocity_parity"]
    bad=dict(a,device="cuda:0")
    assert not parity(a,bad)["same_backend"]
    altered=dict(a,runtime_z_vz=[[1,.93,0.1]])
    assert not parity(a,altered)["recorded_pose_velocity_parity"]

def test_missing_native_evidence_fail_closed():
    x=diagnose({"actual_runtime_frames":180,"tensor_contact_counts":200,
                 "tensor_peak_force_N":667.0})
    assert not x["native_pair_witness"]
    assert x["runtime_steps"]==180
    assert x["tensor_contact_counts"]==200
