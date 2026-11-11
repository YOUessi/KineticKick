"""v83 pre-registered CPU regression and negative-control assertions."""
import importlib.util,json,math,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v083_phase_chain_candidate import generate,offline_plan,gated_original_controller,JOINT,N,INDEX
from v083_independent_coupling_audit import audit_one
from stance_safe_temporal_policy_v067 import RealG1State

def test_all_29_joints_named_once_and_control_groups():
 assert N==29 and len(JOINT)==len(set(JOINT))==29
 plan=offline_plan()
 full=plan["ablation_controls"]["full"]
 assert full["active_joint_count"]>=20
 assert full["peak_joint_offset_rad"]>=.05
 assert full["peak_intended_joint_rate_rad_s"]>.1
 assert plan["original_G1_action_deployment"]=="BLOCKED_FAIL_CLOSED_UNTIL_NATIVE_WITNESS"

@pytest.mark.parametrize("ablation",["full","no_counter_arm","phase_reversed","weak_foot_only"])
def test_ends_zero_no_nan_and_actual_derivative(ablation):
 for t in np.linspace(0,1.05,44):
  m=generate(float(t),ablation=ablation)
  assert m.q_offset_rad.shape==(29,)
  assert m.dq_offset_rad_s.shape==(29,)
  assert np.isfinite(m.q_offset_rad).all() and np.isfinite(m.dq_offset_rad_s).all()
 for t in (0.,1.05,2.):
  assert np.max(np.abs(generate(t,ablation=ablation).q_offset_rad))<1e-8
 t=.49;eps=1e-6
 forward=generate(t+eps,ablation=ablation).q_offset_rad
 backward=generate(t-eps,ablation=ablation).q_offset_rad
 assert np.allclose((forward-backward)/(2*eps),generate(t,ablation=ablation).dq_offset_rad_s,atol=1e-4)

def test_ablation_is_real_and_directional():
 t=.48
 a=generate(t);b=generate(t,ablation="no_counter_arm")
 assert abs(a.q_offset_rad[INDEX["right_shoulder_pitch_joint"]])>.01
 assert b.q_offset_rad[INDEX["right_shoulder_pitch_joint"]]==0.
 rev=generate(t,ablation="phase_reversed")
 assert np.linalg.norm(a.q_offset_rad-rev.q_offset_rad)>.01
 assert np.count_nonzero(np.abs(generate(t,ablation="weak_foot_only").q_offset_rad)>1e-3)<4

@pytest.mark.parametrize("bad",[-1.,float("nan"),float("inf")])
def test_invalid_phase_rejected(bad):
 with pytest.raises(ValueError):generate(bad)

def test_no_verified_mesh_collider_fails_closed():
 q=np.zeros(N);lim=np.ones(N)*6;pos=np.column_stack((-np.ones(N)*3,np.ones(N)*3))
 state=RealG1State(joint_names=JOINT,physical_frame_time_s=1.1,
     joint_pos_rad=q,measured_joint_vel_rad_s=q,actual_joint_vel_limits_rad_s=lim,
     actual_joint_pos_limits_rad=pos,measured_left_support_normal_N=200.,
     measured_right_support_normal_N=0.,measured_nonfoot_contact_N=0.,
     measured_body_tilt_deg=0.,measured_support_margin_m=.1,
     actual_collider_witness=None,ball_contact_occurred=False)
 original=np.linspace(-.1,.1,N)
 candidate,decision=gated_original_controller(original,q,np.ones(N),state,.48)
 assert not decision.eligible_for_physx_deployment
 assert decision.reason=="no_verified_original_GPU_G1_mesh_collider_witness"
 assert np.array_equal(decision.action,original)
 assert np.max(np.abs(candidate.q_offset_rad))>.05

def test_original_G1_evidence_missing_rejected_not_imputed():
 with pytest.raises((ValueError,KeyError,IndexError)):
  audit_one({"case":"fake","scenario":"A","raw_sha256":"fake"},
            {"traces":[[]],"records":[{"contact_time_s":1.57}],"joint_names":JOINT,
             "joint_velocity_limits":[1]*29})

def test_protocol_non_tuning_and_nonphysical_witness():
 p=json.loads((ROOT/"configs/v083_frozen_independent_batch.json").read_text())
 assert p["no_holdout_tuning"] and p["real_physx_pair_manifold_known"] is False
 assert len(p["trials"])==6
 assert len(set(x["case"] for x in p["trials"]))==6
 assert {x["scenario"] for x in p["trials"]}=={"A","B"}
 assert len({x["source_config"] for x in p["trials"]})==3
 assert len({x["seed"] for x in p["trials"]})==2
 assert p["hard_gates"]["instep_fraction_min"]==.95
 assert p["safety"]["min_disk_GiB"]>=20

def test_stage_instrumentation_preserves_original_scientific_contract():
 a=(ROOT/"tools/v081_run_tensor_strike.py").read_text()
 b=(ROOT/"tools/v083_run_staged_strike.py").read_text()
 assert "--tensor-wrist-hip-v081" in b
 assert "V083_STAGE" in b
 assert "from v083_staged_tensor import OriginalG1TensorSelfPair" in b
 assert 'nonfoot_contact_termination' in a and 'nonfoot_contact_termination' in b
 assert b.count("self.tensor_pair_v081=OriginalG1TensorSelfPair(self.env)")==1
 # Inspector only: no source's motor controller modification.
 for tag in ("def step(self,observations,frame):","def apply(self,frames,support)",
             "self.task.commands(actions)"):
  assert tag in a and tag in b
