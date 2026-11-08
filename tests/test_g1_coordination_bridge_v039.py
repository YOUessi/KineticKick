"""v39: exact robot URDF FK and semantic-only relation bridge contract tests."""
from pathlib import Path
import json,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from g1_coordination_bridge_v039 import G1URDFFK,audit_fk,robot_relations,URDF,CENTER,OLD_B,BODY,SEGMENT_ENDPOINTS

@pytest.mark.skipif(not URDF.exists(),reason="Original PAiD G1 URDF only available on Tang")
def test_original_g1_fk_has_all_29_movable_joints_and_40_links():
    fk=G1URDFFK()
    assert len(fk.active_names)==29
    assert len(fk.links)==40
    assert fk.root=="pelvis"

@pytest.mark.skipif(not URDF.exists() or not CENTER.exists(),reason="Local original G1 PhysX needed")
def test_original_g1_body_link_positions_match_physx_exact_fk():
    fk=G1URDFFK()
    for source in (CENTER,OLD_B):
        run=json.loads(source.read_text())
        errors=audit_fk(run,fk)
        assert len(errors)==10
        assert max(e["euclidean_error_m"] for e in errors)<2e-5

@pytest.mark.skipif(not URDF.exists() or not CENTER.exists(),reason="Local original G1 PhysX needed")
def test_robot_cosine_relation_semantics_have_11_named_components():
    fk=G1URDFFK()
    data=json.loads(CENTER.read_text())
    pos=data["kinematics"][0][75]
    links=fk.forward(pos["root_pose"],data["joint_names"],pos["joint_pos"])
    values=robot_relations(links)
    assert values.shape==(11,) and np.isfinite(values).all()
    assert (abs(values)<=1).all()

@pytest.mark.skipif(not URDF.exists() or not CENTER.exists(),reason="Local original G1 PhysX needed")
def test_fk_rejects_invalid_urdf_state_joint_schema():
    fk=G1URDFFK()
    r=json.loads(CENTER.read_text())
    row=r["kinematics"][0][0]
    with pytest.raises(ValueError):
        fk.forward(row["root_pose"],r["joint_names"][:-1],row["joint_pos"][:-1])
    with pytest.raises(ValueError):
        fk.forward([0,0,0,0,0,0,0],r["joint_names"],row["joint_pos"])

AUDIT=ROOT/"data/research_human_kicks/3dsp_cvpr2024/processed_v039/g1_bridge/bridge_audit.json"
@pytest.mark.skipif(not AUDIT.exists(),reason="Local ignored bridge data")
def test_matching_features_do_not_promote_unverified_human_reference():
    meta=json.loads(AUDIT.read_text())
    assert meta["record_count"]==2 and meta["robot_frames"]==40
    assert meta["semantic_matched_not_anatomically_equivalent"]
    assert meta["human_contact_time_unknown"]
    assert meta["human_to_robot_joint_angles_model_trained"] is False
    assert meta["actual_g1_action_policy_trained_from_3dsp"] is False
    assert meta["human_to_robot_kinematic_retargeting_validated"] is False
    for case in meta["records"]:
        assert case["fk_checked_link_positions"]==10
        assert case["maximum_fk_position_error_m"]<2e-5
