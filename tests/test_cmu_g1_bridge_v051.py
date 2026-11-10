"""v51 source-correct real human optical C3D and G1 original PhysX FK bridge."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from cmu_g1_semantic_bridge_v051 import (
 human_six_timed,robot_six_timed,g1_six_semantic_relations,
 HUMAN_FEATURE_INDEX,SIX_FEATURE_NAMES,ROOT as BRIDGE_ROOT,CMU)
from g1_coordination_bridge_v039 import G1URDFFK,BODY
from g1_contact_coordination_projection_v040 import TorchG1Relations,load_robot_source
STUDY=ROOT/"configs/cmu_g1_cross_morph_v051.json"
BASE=ROOT/"data/research_human_kicks/cmu_g1_cross_morph_v051"
EVIDENCE=BASE/"cross_morphology_audit.json"
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

def test_exact_six_unambiguous_human_and_robot_segment_operators():
 assert HUMAN_FEATURE_INDEX==(0,1,4,5,6,7)
 assert len(set(SIX_FEATURE_NAMES))==6
 assert "ankle_toe" not in " ".join(SIX_FEATURE_NAMES)
 assert len(set(BODY.values()))==len(BODY)

@pytest.mark.skipif(not STUDY.exists(),reason="Frozen v51 source study missing")
def test_frozen_original_CMU_and_PhysX_SHA_and_contact_truth_restrictions():
 c=json.loads(STUDY.read_text())
 assert c["version"]=="0.1.51"
 assert c["true_human_contact_time_qualified"] is False
 assert c["training_G1_controller_in_this_trial"] is False
 assert c["new_G1_PhysX_runs_in_this_trial"]==0
 assert c["expected_human_robot_comparisons"]==4
 assert c["exclude_human_ankle_toe_features_without_qualified_G1_toes"]
 assert c["qualified_human_ball_marker_launch_clips"]==["10_02","10_05"]
 for f,h in c["source_sha256"].items():
  if not (ROOT/f).exists():pytest.skip("Tang-only old G1 14s PhysX and CMU source files not in clean clone")
  assert sha(f)==h,f

@pytest.mark.skipif(not EVIDENCE.exists(),reason="Tang real human+robot traces not in fresh clone")
def test_human_and_robot_temporal_source_uses_different_event_truth():
 c=json.loads(EVIDENCE.read_text())
 assert c["number_of_exploratory_human_robot_pairs"]==4
 assert not c["contact_conditioned_true_human_kinetics_qualified"]
 assert not c["independent_heldout_generalization"]
 assert not c["new_trained_PPO_policy"]
 assert c["original_120hz_human_features_and_50hz_G1_PhysX_URDF_compared"]
 for row in c["comparisons"]:
  assert row["events_are_NOT_groundtruth_synchronized"]
  assert not row["expected_true_human_instestep_available"]
  assert row["compare_6_3D_segment_cosine_features_not_joint_angle"]
  assert row["number_of_comparable_body_channels"]==6
  assert 0.<=row["absolute_cross_morphology_cosine_MSE"]<=4.
  assert 0.<=row["within_own_body_first_frame_relative_motion_MSE"]<=4.
  assert 0.40<row["relative_motion_amplitude_ratio_G1_to_human"]<.90
 with np.load(BASE/"cmu_g1_6d_exploratory_timeline.npz",allow_pickle=False) as z:
  assert z["relative_event_time_s"].shape==(20,)
  assert z["10_02_cos"].shape==(20,6)
  assert z["v32_frozen_cos"].shape==(20,6)
  assert not z["human_launch_time_is_verified_contact"].item()
  assert z["G1_robot_real_PhysX_impact_verified"].item()

@pytest.mark.skipif(not (ROOT/"reports/v048_preimpact_new01_original_v32.json").exists(),reason="Actual G1 physically measured state only on Tang")
def test_original_g1_torch_29dof_link_semantics_matches_independent_urdf_numpy():
 robot=load_robot_source(ROOT/"reports/v048_preimpact_new01_original_v32.json")
 sample=[60,65,72]
 angles=robot["q"][sample]
 roots=robot["root"][sample]
 torch_fk=TorchG1Relations(robot["joint_names"])
 actual=g1_six_semantic_relations(torch_fk,torch.tensor(roots,dtype=torch.float64),
                                   torch.tensor(angles,dtype=torch.float64))
 other=G1URDFFK()
 expected=[]
 def cos(a,b):
  return float((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)))
 for r,q in zip(roots,angles):
  links=other.forward(r,robot["joint_names"],q)
  pts={semantic:links[link][:3,3] for semantic,link in BODY.items()}
  trunk=.5*(pts["left_shoulder"]+pts["right_shoulder"])-.5*(pts["left_hip"]+pts["right_hip"])
  expected.append([
   cos(pts["left_hip"]-pts["left_knee"],pts["left_ankle"]-pts["left_knee"]),
   cos(pts["right_hip"]-pts["right_knee"],pts["right_ankle"]-pts["right_knee"]),
   cos(pts["left_shoulder"]-pts["left_elbow"],pts["left_wrist"]-pts["left_elbow"]),
   cos(pts["right_shoulder"]-pts["right_elbow"],pts["right_wrist"]-pts["right_elbow"]),
   cos(trunk,pts["left_knee"]-pts["left_hip"]),
   cos(trunk,pts["right_knee"]-pts["right_hip"])])
 np.testing.assert_allclose(actual,np.asarray(expected),atol=1e-12,rtol=0)

@pytest.mark.skipif(not EVIDENCE.exists(),reason="Tang real mocap / original G1 source only")
def test_actual_CMU_launch_proxy_vs_real_G1_impact_timestamps_not_equated():
 a=json.loads((ROOT/CMU/"quality_audit.json").read_text())
 rel=np.arange(-.4,0,.02).round(8)
 for x in ("10_02","10_05"):
  human,t=human_six_timed(x,rel,a)
  assert human.shape==(20,6)
  assert 2.<t<3.
  assert np.isfinite(human).all()
 robot,t=robot_six_timed("reports/v048_preimpact_new01_original_v32.json",rel)
 assert robot.shape==(20,6)
 assert abs(t-1.57)<1e-8
 assert np.isfinite(robot).all()
