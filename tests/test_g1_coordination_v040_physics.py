"""v40 read-only active/disabled full PhysX outcomes and engagement regression."""
from pathlib import Path
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]
ZERO=ROOT/"reports/v040_human_proxy_zero_center_audit.json"
ACTIVE=ROOT/"reports/v040_human_proxy_center_audit.json"
@ pytest.mark.skipif(not ZERO.exists() or not ACTIVE.exists(),reason="Tang-local original 14s physics")
def test_v40_exact_zero_gates_and_original_goal_remain():
 z=json.loads(ZERO.read_text())
 assert z["passed"] and z["is_zero"]
 assert z["full_14s_physics_steps"]==2800
 assert z["gates"]["whole_ball_original_goal"]
 assert z["real_joint_speed_excess_samples"]==0
 assert z["actual_control_diagnostic"]["nonzero_decisions"]==0
 assert all(z["physical_zero_parity_streams"].values())
 assert set(z["zero_max_absolute_state_diffs"].values())=={0.}

@ pytest.mark.skipif(not ACTIVE.exists(),reason="Tang-local center trial")
def test_v40_kinematic_generator_existed_but_no_physical_action_applied():
 r=json.loads(ACTIVE.read_text())
 assert not r["passed"]
 assert r["gates"]["original_task_success"]
 assert r["gates"]["whole_ball_original_goal"]
 assert r["gates"]["physical_full_14s"]
 assert r["gates"]["actual_joint_speed_limit"]
 assert not r["gates"]["generated_ref_observable"]
 assert r["actual_control_diagnostic"]["eligible_decisions"]>0
 assert r["actual_control_diagnostic"]["nonzero_decisions"]==0
 assert r["actual_control_diagnostic"]["peak_rad"]==0.
 assert r["outgoing_speed_mps"]==r["original_outgoing_speed_mps"]

@ pytest.mark.skipif(not ACTIVE.exists(),reason="Tang-local center trial")
def test_v40_no_unseen_case_was_spent_for_nonengaging_policy():
 protocol=json.loads((ROOT/"configs/g1_coordination_physics_v040.json").read_text())
 for case in protocol["cases_in_conditional_order"][1:]:
  assert not (ROOT/case["candidate"]).exists()
 assert protocol["no_hyperparameter_search"] and protocol["no_G1_policy_promotion"]
