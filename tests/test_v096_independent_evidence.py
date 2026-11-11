"""Independent v096 audit tests: reject invented native manifold and broken hashes."""
import json,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v096_independent_evidence import check_frame_trace,parity
def fake():
 row=dict(device="cuda:0",layout="colliding",sensor_mode="manual",box_mode="kinematic",capture="native",report_mode="on")
 raw=dict(**row,steps_requested=180,actual_runtime_frames=180,dt_s=.005,original_G1=False,
  runtime_z_vz=[[i,0.95-i*.003,0] for i in range(1,181)],box_position_w=[0,0,0],
  observed_collision_dynamics=True,tensor_pair_witness=False,
  stage=dict(all_colliders=dict(ProbeBall=["/World/ProbeBall/geometry/mesh"],ProbeBox=["/World/ProbeBox/geometry/mesh"]),
    ball_report=True,box_report=True),native_pair_witness=False,native_events=[])
 return raw,row
def test_exact_real_200hz_step_contract():
 raw,row=fake()
 assert check_frame_trace(raw,row).shape==(180,3)
 raw["runtime_z_vz"][75][0]=77
 with pytest.raises(ValueError,match="BAD_200HZ"):check_frame_trace(raw,row)
def test_no_unproven_native_manifold():
 raw,row=fake()
 raw["native_pair_witness"]=True
 with pytest.raises(ValueError,match="UNVERIFIED_NATIVE"):check_frame_trace(raw,row)
def test_parity_rejects_different_physical_models():
 raw,row=fake()
 assert parity(raw,raw)["exact_saved_state"]
 other={**raw,"box_mode":"dynamic"}
 with pytest.raises(ValueError,match="DIFFERENT_MECHANICS"):parity(raw,other)
def test_tensor_filtered_pair_must_not_claim_native_contact():
 raw,row=fake();row["capture"]="both";raw["capture"]="both"
 raw["tensor_pair_witness"]=True
 raw["tensor_contact_points"]=[dict(point_w=[0,0,.2],normal_w=[0,0,1],force_magnitude_N=1.)]
 assert check_frame_trace(raw,row).shape==(180,3)
 assert raw["native_pair_witness"] is False

