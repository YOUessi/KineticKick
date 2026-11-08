"""v39 3DSP data provenance, relation invariants, no-match-leakage and CVAE tests."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from build_professional_shot_coordination_v039 import (
    coordination,joint_points,info_parser,JOINT_ORDER,PAIRS)
from conditional_coordination_v039 import CoordinationCVAE,masked_errors

def fixture_xyz():
    rng=np.random.default_rng(3901)
    x=rng.normal(size=(20,17,3)).astype(float)
    x[:,:,2]+=2.
    return x

def test_relation_cosines_rigid_rotation_translation_and_scale_invariant():
    xyz=fixture_xyz()
    raw,mask=coordination(xyz)
    assert raw.shape==(20,11) and bool(mask.all())
    angle=.66
    R=np.array([[np.cos(angle),-np.sin(angle),0],
                [np.sin(angle),np.cos(angle),0],[0,0,1]])
    changed,valid=coordination((xyz@R.T)*3.2+np.array([23.,-11.,2.]))
    np.testing.assert_allclose(changed,raw,rtol=0,atol=4e-14)
    np.testing.assert_array_equal(valid,mask)

def test_bad_3d_pose_rejected_instead_of_inventing_validity():
    p=fixture_xyz()
    p[2,3,1]=float("nan")
    with pytest.raises(ValueError):coordination(p)
    p=fixture_xyz()
    p[0,JOINT_ORDER["Left Knee"]]=p[0,JOINT_ORDER["Left Hip"]]
    vals,valid=coordination(p)
    assert (~valid[0]).any()
    assert np.isnan(vals[0][~valid[0]]).all()
    assert np.isfinite(vals[valid]).all()

def test_named_skeleton_contract_required():
    labels={n:{"name":k,"x":float(i),"y":float(i*2),"z":float(i*3)}
            for k,i in JOINT_ORDER.items() for n in [str(i)]}
    doc={"shooter_tracklet_id":4,"keypoint_3d":labels}
    x=joint_points(doc)
    assert x.shape==(17,3)
    doc["keypoint_3d"]["1"]["name"]="NOT_RIGHT_HIP"
    with pytest.raises(ValueError):joint_points(doc)

def test_shot_annotation_is_not_impact_truth():
    text="""[info]\nid=00001\nshooter_tracklet_id=8\nlabel=Shots on target\nposition_step=40\nannotated_position=347458\nstart_position=346958\nend_position=347558\ngame=match1\n"""
    raw,t=info_parser(text.encode())
    assert raw["label"]=="Shots on target"
    assert abs(t[0]+.5)<1e-10
    assert abs(t[12]+.02)<1e-10
    assert abs(t[13]-.02)<1e-10

def test_past_conditioned_cvae_shapes_and_gradients():
    torch.manual_seed(3901)
    model=CoordinationCVAE()
    a=torch.randn(4,8,11).clamp(-1,1)
    y=torch.randn(4,12,11).clamp(-1,1)
    before=model.predict(a)
    assert before.shape==(4,12,11) and torch.isfinite(before).all()
    loss,values=model.training_objective(a,y)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    sample=model.sample(a,n_samples=3)
    assert sample.shape==(3,4,12,11)
    assert torch.isfinite(sample).all()

def test_conditional_rejects_bad_layout_and_mse_empty():
    model=CoordinationCVAE()
    with pytest.raises(ValueError):model.predict(torch.zeros(1,7,11))
    with pytest.raises(ValueError):
        masked_errors(torch.ones(1,2,11),torch.zeros(1,2,11),torch.zeros(1,2,11,dtype=torch.bool))

RAW=ROOT/"data/research_human_kicks/3dsp_cvpr2024/processed_v039/coordination_3dsp_v039.npz"
@ pytest.mark.skipif(not RAW.exists(),reason="3DSP data ignored on clone")
def test_real_data_source_and_group_heldout():
    cfg=json.loads((ROOT/"configs/professional_coordination_pilot_v039.json").read_text())
    assert hashlib.sha256(RAW.read_bytes()).hexdigest()==cfg["data_sha256"]
    z=np.load(RAW,allow_pickle=False)
    assert z["relation_cosines"].shape==(200,20,11)
    assert z["valid"].all()
    assert not bool(z["verified_event_contact_truth"])
    partition=z["split"];group=z["source_game"]
    assert {s:int((partition==s).sum()) for s in ("train","val","test")}==cfg["sample_counts"]
    sets=[set(group[partition==s]) for s in ("train","val","test")]
    assert all(not(sets[i]&sets[j]) for i,j in ((0,1),(1,2),(0,2)))
    assert len(set(group))==20
    assert np.isfinite(z["relation_cosines"]).all()
    assert np.max(np.abs(z["relation_cosines"]))<=1.000001
