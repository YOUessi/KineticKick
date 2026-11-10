"""v52 fixed real 120Hz CMU source and differentiable G1 projection checks.

Uses Tang-only original source material; clean Git clone may skip dataset tests.
None of these tests certify professional kick force, human contact truth or
physical action safety, which require separate full original PhysX rollouts.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from build_cmu_g1_candidate_v052 import (
    CONFIG, OUTPUT, ROBOT, V51, digest, verified_inputs, phase_window,
    full_reference, joint_limits, g1_six_differentiable)
from cmu_g1_semantic_bridge_v051 import g1_six_semantic_relations
from g1_contact_coordination_projection_v040 import TorchG1Relations, load_robot_source


def test_registered_protocol_no_false_ground_truth_or_unseen_claim():
    c=json.loads(CONFIG.read_text())
    assert c["version"]=="0.1.52"
    assert c["human_clips"]==["10_02","10_05"]
    assert c["training_human_subjects"]==1
    assert c["verified_human_contact"] is False
    assert c["human_explosive_instep_verified"] is False
    assert c["no_human_generalization_claim"] is True
    assert c["control_is_trained"] is False
    assert c["output_50hz_controller_steps"]==700
    assert c["max_joint_rad"]==.006
    assert c["max_slew_per_20ms_rad"]==.0012


def test_reject_tampered_source_freeze(tmp_path):
    c=json.loads(CONFIG.read_text())
    assert all(len(h)==64 for h in c["frozen_sha256"].values())
    for path,expected in c["frozen_sha256"].items():
        if not (ROOT/path).exists():
            pytest.skip("Tang-only original mocap or G1 evidence unavailable")
        assert digest(ROOT/path)==expected,path


def test_staggered_phase_envelope_is_bounded_and_zero_near_real_contact():
    r=np.arange(-.4,0,.02).round(8)
    g=phase_window(r)
    assert g.shape==(20,)
    assert 0<=g.min() and g.max()<=1.
    assert g[0]==g[-1]==0.
    assert np.any(g[3:12]>.95)
    assert np.all(g[r>=-.06-1e-9]==0.)
    assert (np.diff(g[:5])>=-1e-10).all()
    assert (np.diff(g[-6:])<=1e-10).all()


@pytest.mark.skipif(not ROBOT.exists(),reason="Original actual 700x29 G1 motion is Tang-only")
def test_six_differentiable_channels_match_v51_and_nonzero_29dof_gradients():
    robot=load_robot_source(ROBOT)
    fk=TorchG1Relations(robot["joint_names"])
    r=torch.tensor(robot["root"][[50,64,77]],dtype=torch.float64)
    q=torch.tensor(robot["q"][[50,64,77]],dtype=torch.float64,requires_grad=True)
    now=g1_six_differentiable(fk,r,q)
    old=g1_six_semantic_relations(fk,r.detach(),q.detach())
    assert now.shape==(3,6)
    np.testing.assert_allclose(now.detach().numpy(),old,atol=1e-12,rtol=0)
    now[:,[0,2,4]].sum().backward()
    assert torch.isfinite(q.grad).all()
    assert q.grad.abs().max()>1e-7


@pytest.mark.skipif(not ROBOT.exists(),reason="Tang-only original physical motion required")
def test_bounded_zero_20_sample_trajectory_rejected_as_null_advance():
    robot=load_robot_source(ROBOT)
    c=json.loads(CONFIG.read_text())
    limits=joint_limits(robot["joint_names"])
    assert limits.shape==(29,2)
    with pytest.raises(ValueError,match="unobservable"):
        full_reference(np.zeros((20,29)),np.arange(-.4,0,.02),robot,c)


@pytest.mark.skipif(not V51.exists(),reason="Original CMU source/physical evidence is Tang-only")
def test_v51_real_optical_source_contract_before_new_motion_generation():
    c=json.loads(CONFIG.read_text())
    humans, robot=verified_inputs(c)
    assert len(humans["motions"])>=6
    assert robot["q"].shape==(700,29)
    assert abs(robot["contact_time"]-1.57)<1e-7


@pytest.mark.parametrize("clip",["10_02","10_05"])
def test_generated_outputs_remain_bounded_and_sha_pinned(clip):
    out=OUTPUT/clip/"g1_joint_delta_50hz.npz"
    meta=OUTPUT/clip/"candidate_audit.json"
    phys=ROOT/("configs/cmu_real_120hz_"+clip+"_physx_v052.json")
    if not (out.exists() and meta.exists() and phys.exists()):
        pytest.skip("No generated v52 candidate yet")
    a=json.loads(meta.read_text())
    c=json.loads(phys.read_text())
    assert a["human_optical_launch_is_NOT_contact"]
    assert not a["actual_PhysX_policy_tested"]
    assert not a["human_kick_force_or_instep_class_supervised"]
    assert a["human_subjects_in_training"]==1
    assert a["reference_safety"]["active_controller_frames"]>0
    assert a["optimization"]["proxy_MSE_after"]<a["optimization"]["proxy_MSE_before"]
    assert a["output_sha256"]==digest(out)
    assert c["human_coordination_reference"]["trajectory_sha256"]==digest(out)
    assert a["physics_config_sha256"]==digest(phys)
    with np.load(out,allow_pickle=False) as z:
        delta=z["delta_joint_position_rad"]
        t=z["controller_time_s"]
        assert delta.shape==(700,29)
        assert np.isfinite(delta).all()
        assert abs(delta).max()<=.0060005
        assert abs(np.diff(delta,axis=0)).max()<=.0012005
        assert np.all(delta[t>=float(z["robot_PhysX_impact_time_s"])-.06-1e-6]==0)
        assert not z["human_impact_time_verified"].item()
        assert not z["source_is_verified_instep"].item()
        assert len(z["six_named_geometry_channels"])==6
        assert np.count_nonzero(abs(delta)>1e-7)>3
