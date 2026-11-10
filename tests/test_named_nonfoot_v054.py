"""v54 offline signed body-contact event and G1 5ms physical parity contract."""
from pathlib import Path
import json,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from run_named_nonfoot_diagnostics_v054 import CONFIG,PROGRESS,sha,freeze
from audit_named_nonfoot_diagnostics_v054 import OUTPUT as AUDIT,COMPARE

def test_protocol_three_fixed_source_policies_and_read_only_sensor():
 c=json.loads(CONFIG.read_text())
 assert freeze(c)
 assert c["version"]=="0.1.54"
 assert c["new_physics_rollouts"]==3
 assert c["no_policy_or_ball_state_changes"]
 assert c["no_adaptive_retraining"]
 assert c["match_all_common_physical_time_fields"]
 assert c["original_nonfoot_termination_is_not_verified_fall"]
 assert c["scoped_to_known_development_center"]
 assert c["nonfoot_sensor_threshold_for_readonly_trace_N"]==5.
 assert tuple(c["original_three_center_policies"])==("frozen_v32","cmu_10_02","cmu_10_05")
 assert len(c["immutable_sha256"])==14
 assert "traces" in COMPARE and "termination_reasons" in COMPARE

@pytest.mark.skipif(not AUDIT.exists(),reason="Named body contact sensor 3-way physical batch not yet finished")
def test_all_original_instrumented_physics_and_failure_claims_audited():
 x=json.loads(AUDIT.read_text())
 assert x["version"]=="0.1.54"
 assert x["original_protocol_sha256"]==sha(CONFIG)
 assert x["original_progress_sha256"]==sha(PROGRESS)
 assert x["original_center_3_diagnostic_rollouts"]==3
 assert x["no_policy_training"]
 assert x["no_new_heldout_context_or_generalization"]
 assert len(x["results"])==3
 for k,row in x["results"].items():
  assert row["name"]==k
  assert row["source_sha256"]==sha(ROOT/json.loads(CONFIG.read_text())["original_three_center_policies"][k]["original"])
  assert row["instrumented_sha256"]==sha(ROOT/json.loads(CONFIG.read_text())["original_three_center_policies"][k]["instrumented_output"])
  assert row["data_does_not_measure_human_instestep_impact"]
  assert row["evidence_was_not_collected_in_original_v52"]
  assert row["named_nonfoot_event_count"]>=0
  assert row["named_nonfoot_event_count"]>=row["terminal_near_event_count"]
  assert 0<row["original_actual_robot_root_z_m"]<2
  assert 0<=row["original_actual_robot_root_tilt_deg"]<180
  assert set(row["parity_by_original_field"])==set(COMPARE)
  assert row["parity_full_on_all_common_physics_records"]==all(
      d["equal"] for d in row["parity_by_original_field"].values())
  for b,summary in row["named_nonfoot_body_counts_and_peak"].items():
   assert b
   assert summary["count"]>0 and summary["max_force_N"]>=5.
   assert summary["first_s"]<=summary["last_s"]
 assert x["results"]["cmu_10_05"]["original_physics_terminated"]==[["physical_nonfoot"]]
 assert x["results"]["cmu_10_05"]["original_end_s"]<14.
 assert x["results"]["frozen_v32"]["original_end_s"]==14.

@pytest.mark.skipif(not PROGRESS.exists(),reason="Original instrumented G1 batch still running")
def test_signed_instrumented_progress_does_not_skip_trials():
 c=json.loads(CONFIG.read_text())
 p=json.loads(PROGRESS.read_text())
 assert p["protocol_sha256"]==sha(CONFIG)
 if len(p["runs"])!=3:
  pytest.skip("Only a prefix of three original named body contact replays complete")
 for name,step in zip(c["original_three_center_policies"],p["runs"]):
  assert step["name"]==name
  assert step["sha256"]==sha(ROOT/step["result"])
  assert step["bytes"]>0

@pytest.mark.skipif(not (ROOT/"reports/v054_opposite_nonfoot_link_forces.json").exists(),
                    reason="Only Tang contains original 5ms PhysX named net forces")
def test_wrist_hip_opposite_forces_are_only_an_inference_not_contact_ground_truth():
 x=json.loads((ROOT/"reports/v054_opposite_nonfoot_link_forces.json").read_text())
 r=x["left_wrist_hip_symmetric_force_pair"]
 assert x["version"]=="0.1.54"
 assert x["direct_original_contact_pair_label_verified"] is False
 assert x["human_CMU_true_contact_force_verified"] is False
 assert x["need_next_step_pairwise_contact_manifold"] is True
 assert set((r["body_a"],r["body_b"]))=={"left_wrist_yaw_link","left_hip_roll_link"}
 assert abs(r["measured_t_s"]-5.145)<1e-4
 assert abs(r["force_magnitude_each_N"]-25.8901533)<.01
 assert abs(r["cosine_between_forces"]+1.)<1e-10
 assert r["net_force_cancellation_ratio"]<1e-9
 assert .08<r["position_of_link_origins_separation_m"]<.15
 assert r["body_pair_directly_measured_in_contact_manifold"] is False
