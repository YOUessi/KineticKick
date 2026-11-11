"""v64 real-human geometric phase prior / original G1 risk contract tests.

No professional human ground-truth inference: a 120Hz optical phase fit from
one individual remains only exploratory. Real physics must decide shooting.
"""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from wholebody_safety_interface_v055 import G1_JOINT_ORDER
from train_cmu_phase_generator_v064 import fit_human,phase_features

def test_fit_20x6_limited_source_is_finite():
 n=20
 toy=np.stack([np.sin(np.linspace(0,3,n)+k*.12) for k in range(6)],axis=1)
 signal,coeff,info=fit_human(np.arange(n),toy)
 assert signal.shape==(20,6)
 assert np.all(np.isfinite(signal))
 assert np.max(abs(signal[0]))<1e-12
 assert info["source_human_count"]==1
 assert info["no_independent_human_generalization"]
 assert coeff.shape==(9,6)
 assert phase_features(np.linspace(0,1,20)).shape==(20,9)

def test_reject_bad_human_data():
 for val in [np.ones((20,8)),np.ones((19,6)),np.full((20,6),np.nan)]:
  with pytest.raises(ValueError):fit_human(None,val)

def test_v64_original_G1_motor_joint_order_guard():
 assert len(G1_JOINT_ORDER)==29
 assert G1_JOINT_ORDER[1]=="right_hip_pitch_joint"
 assert G1_JOINT_ORDER[10]=="right_knee_joint"
 assert G1_JOINT_ORDER[14]=="right_ankle_pitch_joint"
 assert G1_JOINT_ORDER[18]=="right_ankle_roll_joint"

def test_v64_protocol_strict_and_frozen():
 p=json.loads((ROOT/"configs/g1_human_phase_batch_v064.json").read_text())
 assert p["registered_before_PhysX"]
 assert p["human_training_clips"]==["10_02"]
 assert p["human_foot_ball_contact_truth"] is False
 assert p["valid_human_instep_force_groundtruth"]==0
 assert p["instep_impulse_gate"]==.95
 assert p["goal_speed_gate_mps"]==3.5
 assert p["full_duration_s"]==14
 assert len(p["cases"])==12
 assert len(set(x["id"] for x in p["cases"]))==12
 assert len(set(x["seed"] for x in p["cases"] if x["seed"]!=1201))==3
 assert {x["policy"] for x in p["cases"]}=={"v32","human_ordered","human_reversed"}
 assert all(x["output"].startswith("reports/v064_") for x in p["cases"])

def test_new_real_human_source_not_legacy_synthetic():
 p=json.loads((ROOT/"configs/cmu_phase_model_v064.json").read_text())
 assert p["train_phase_origin"].startswith("CMU inferred optical")
 assert p["human_contact_groundtruth"] is False
 assert p["elite_player_power_verified"] is False
 assert p["no_unseen_human_generalization_claim"]
 assert p["max_joint_delta_rad"]<=.055
 assert p["max_slew_per_20ms_rad"]<=.004
 for name in ("human_ordered","human_reversed"):
  c=json.loads((ROOT/f"configs/g1_human_phase_{name}_v064.json").read_text())
  assert c["phase_coordinated_v063"]["enabled"] is False if "phase_coordinated_v063" in c else True
  assert c["human_phase_v064"]["sequence"]==name
  assert c["human_phase_v064"]["human_true_contact"] is False

def test_human_reference_local_if_present():
 d=ROOT/"experiments/cmu_phase_fit_v064"
 if not (d/"human_ordered.npz").exists():
  pytest.skip("Local original CMU-derived reference not stored in git")
 original=[];times=[]
 for name in ("human_ordered","human_reversed"):
  file=d/(name+".npz")
  with np.load(file,allow_pickle=False) as z:
   a=z["delta_joint_position_rad"]
   t=z["controller_time_s"]
   joint_names=z["joint_names"].tolist()
   assert z["real_human_training_clip"].item()=="10_02"
   assert bool(z["human_optical_motion_event_not_contact"])
   assert not bool(z["source_is_verified_instep"])
  assert a.shape==(700,29)
  assert joint_names==list(G1_JOINT_ORDER)
  assert np.isfinite(a).all()
  assert abs(a).max()<=.055+1e-7
  assert abs(np.diff(a,axis=0)).max()<=.0040001
  assert np.all(a[:,[1,4,7,10,14,18]]==0)
  assert np.all(a[t<1.02]==0)
  assert np.all(a[t>=1.42]==0)
  assert np.count_nonzero(np.max(abs(a),axis=1)>1e-5)>=5
  original.append(a);times.append(t)
 assert not np.array_equal(original[0],original[1])
 assert np.array_equal(times[0],times[1])
 for n,x in zip(("human_ordered","human_reversed"),original):
  meta=json.loads((ROOT/f"configs/g1_human_phase_{n}_v064.json").read_text())["human_phase_v064"]
  h=hashlib.sha256((d/(n+".npz")).read_bytes()).hexdigest()
  assert meta["reference_sha256"]==h

def test_cpu_offline_model_no_human_impact_claim():
 c=json.loads((ROOT/"configs/g1_human_phase_batch_v064.json").read_text())
 assert c["experiments_with_unseen_human_subjects"]==0
 assert c["negative_control"].startswith("Same human fit reference")
 assert all(row["qualifier"]=="NEW_FROZEN_CONTEXT" for row in c["cases"] if row["seed"]!=1201)
