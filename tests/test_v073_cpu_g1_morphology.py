"""v73 independent G1 retarget and source isolation regression coverage."""
import sys,json
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from retarget_v073_human_brain_to_g1 import (
 load_checkpoint,autoregressive_future,select_original_window,six_20hz_human)
from wholebody_safety_interface_v055 import G1_JOINT_ORDER
from train_human_temporal_brain_v073 import TemporalBrain,sha

def test_teacher_history_rollout_starts_as_static_when_model_untrained():
 torch.manual_seed(0)
 m=TemporalBrain(24,12,hidden=16)
 inp=np.zeros((24,65),dtype=float)
 inp[-1]=np.linspace(-.1,.1,65)
 future=autoregressive_future(m,inp,np.zeros(65),np.ones(65))
 assert future.shape==(48,65)
 np.testing.assert_allclose(future,np.tile(inp[-1],(48,1)),atol=1e-6)

def test_future_six_geometry_50hz_matches_original_semantic_relations():
 x=np.zeros((48,65))
 x[:,57:]=np.linspace(0,.8,48)[:,None]*np.arange(1,9)[None]/8
 h=six_20hz_human(x)
 assert h.shape==(20,6)
 np.testing.assert_allclose(h[0],0)
 assert h[-1,0]>0
 with pytest.raises(ValueError,match="48-frame"):
  six_20hz_human(x[:40])

def test_launch_proxy_input_window_does_not_make_impact_groundtruth():
 t=np.arange(220)/120
 x=np.ones((220,65))
 valid=np.ones(220,dtype=bool)
 idx,h,f=select_original_window(x,valid,t,anchor=1.)
 assert len(h)==24 and len(f)==48 and idx>=24
 valid[idx+14]=False
 with pytest.raises(ValueError,match="dropout"):
  select_original_window(x,valid,t,anchor=1.)
 with pytest.raises(ValueError,match="not fully measured"):
  select_original_window(x,np.ones(220,bool),t,anchor=.45)

def test_human_checkpoint_requires_sha_before_any_deserialize(tmp_path):
 p=tmp_path/"untrusted.pt"
 p.write_bytes(b"not a signed original source")
 with pytest.raises(ValueError,match="changed"):
  load_checkpoint(p,"0"*64,24,12,96)

def test_cross_subject_protocol_and_true_impact_unavailable():
 c=json.loads((ROOT/"configs/v073_human_temporal_learning.json").read_text())
 assert c["test_clips"]==["11_01"]
 assert set(c["train_clips"]+c["validation_clips"])=={
  "10_01","10_02","10_03","10_05","10_06"}
 assert c["professional_instestep_contact_groundtruth"] is False
 assert c["no_CUDA_no_robot_policy_no_PhysX"] is True

def test_original_real_G1_29dof_reference_is_only_offline():
 p=ROOT/"experiments/v073_cmu_g1_brain_proposals_replay2/human_brain_to_G1_reference_audit.json"
 if not p.exists():pytest.skip("Offline source assets only on Tang")
 audit=json.loads(p.read_text())
 assert audit["no_real_robot_PhysX_actuation_or_high_power_claim"] is True
 assert audit["G1_reference_does_not_certify_foot_collision_or_followthrough"] is True
 for name,z in audit["models"].items():
  assert z["checkpoint_sha256"]
  assert z["cross_subject_heldout_3D_MAE"]>0
  if z["status"]!="OFFLINE_G1_REFERENCE_ONLY_NOT_PHYSX":continue
  q=ROOT/z["source_generated_G1_29dof_file"]
  assert sha(q)==z["source_generated_G1_29dof_sha256"]
  with np.load(q,allow_pickle=False) as data:
   d=data["joint_delta_rad"]
   assert tuple(data["joint_names"].tolist())==G1_JOINT_ORDER
   assert d.shape==(700,29) and np.isfinite(d).all()
   assert data["no_actual_PhysX_evidence"].item() is True
   assert max(abs(d).ravel())<=.05501
   assert max(abs(np.diff(d,axis=0)).ravel())<=.00401
   assert np.all(d[:,[1,4,7,10,14,18]]==0)

def test_v073_independent_audit_refuses_unsupported_physics_claim():
 p=ROOT/"reports/v073_independent_fullhuman_g1_audit.json"
 if not p.exists():pytest.skip("Independent full audit not available locally")
 x=json.loads(p.read_text())
 assert x["real_PhysX_new_rollouts"]==0
 assert not x["learned_G1_controller_evidence"]
 assert x["naturalness_metric_is_unavailable"] is True
 assert x["no_source_or_checkpoint_overwrite"]
 assert x["independent_subject11_test_checkpoint_reloads"]
 assert x["original_g1_29dof_URDF_and_joint_envelope"]
