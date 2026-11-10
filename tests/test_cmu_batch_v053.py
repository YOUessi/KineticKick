"""Frozen v53 entire-batch test: no selective outputs or inflated motion/safety claims."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from run_cmu_physics_matrix_v053 import CONFIG,PROGRESS,ORDER,check,digest
from inspect_cmu_recovery_mechanism_v053 import OUTPUT as MECHANISM
from audit_cmu_physics_matrix_v053 import OUTPUT as AUDIT

def test_freeze_four_original_contexts_three_unmodified_policies():
 c=json.loads(CONFIG.read_text())
 assert check(c)
 assert c["seed"]==5301
 assert c["n_conditions"]==4
 assert c["human_impact_verified"] is False
 assert c["original_CMUs_same_subject"] is True
 assert c["do_not_upgrade_default_control"] is True
 assert c["no_adaptive_tuning"] and c["no_policy_retraining"]
 assert len(c["trials"])==4
 assert tuple(c["policies"])==ORDER
 for j,case in enumerate(c["trials"]):
  assert case["id"]==f"trial{j+1:02d}"
  assert len(case["outputs"])==3
  assert len(case["ball_offset_xy_m"])==2
  assert 0<abs(case["velocity_y_delta_mps"])<.3
  assert len(set(case["outputs"].values()))==3

def test_all_17_original_source_hashes_locked():
 c=json.loads(CONFIG.read_text())
 assert len(c["source_sha256"])==17
 for p,h in c["source_sha256"].items():
  if not (ROOT/p).exists():
   pytest.skip("Tang-only original G1 recorded physical source not mounted")
  assert digest(ROOT/p)==h,p

@pytest.mark.skipif(not MECHANISM.exists(),reason="Tang original G1 200Hz mechanism evidence unavailable")
def test_failure_is_nonfoot_termination_not_verified_root_fall():
 x=json.loads(MECHANISM.read_text())
 assert x["version"]=="0.1.53"
 assert x["no_named_contact_body_available_in_current_v52_records"]
 assert x["all_research_traces_sha_locked"]
 m=x["conditions"]
 assert m["cmu_10_05"]["physical_termination"]==[["physical_nonfoot"]]
 assert m["cmu_10_05"]["physically_observed_fall_by_tilt_or_root_height"] is False
 assert m["cmu_10_05"]["terminal_marker_sample_count"]==1
 assert m["cmu_10_05"]["total_saved_record_samples"]==1029
 assert m["cmu_10_05"]["real_5ms_samples"]==1028
 assert m["cmu_10_05"]["root_tilt_last_deg"]<60
 assert m["cmu_10_05"]["root_z_last_m"]>.45
 assert m["cmu_10_05"]["first_root_displacement_gt_5mm_s"] is not None
 assert m["frozen_v32"]["real_5ms_samples"]==2800
 assert m["cmu_10_02"]["real_5ms_samples"]==2800

@pytest.mark.skipif(not AUDIT.exists(),reason="Entire original 12-episode real G1 PhysX batch still running")
def test_all_original_cases_checked_and_signed_without_selection():
 z=json.loads(AUDIT.read_text())
 assert z["version"]=="0.1.53"
 assert z["frozen_protocol_sha256"]==digest(CONFIG)
 assert z["n_ball_conditions"]==4
 assert z["n_original_PhySX_runs"]==12
 assert z["no_intermediate_hyperparameter_tuning"] is True
 assert z["no_controller_promotion"] is True
 assert z["comparisons_are_tiny_fixed_n_not_statistical_generalization"] is True
 assert z["two_real_120hz_human_CMU_clips_are_same_person"] is True
 assert len(z["comparisons"])==4
 for item in z["comparisons"]:
  assert tuple(item["policies"])==ORDER
  for key,v in item["policies"].items():
   assert v["sha256"]==digest(ROOT/v["source"])
   assert v["success"] is all(v["gates"].values())
   assert v["qdot_overspeed_joint_sample_count"]>=0
   assert v["physical_samples"]>0
   assert not v["gates"]["actual_qdot_under_original_limits"] or v["qdot_overspeed_joint_sample_count"]==0
   assert v["actual_50hz_reference_decisions"]==0 if key=="frozen_v32" else v["actual_50hz_reference_decisions"]>=0
 assert PROGRESS.exists()
 progress=json.loads(PROGRESS.read_text())
 assert len(progress["completed"])==12
 assert progress["protocol_sha256"]==digest(CONFIG)
 for entry in progress["completed"]:
  assert entry["exit_code"]==0
  assert digest(ROOT/entry["result"])==entry["sha256"]
