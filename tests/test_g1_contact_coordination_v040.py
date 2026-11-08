"""v40: full-body differentiable URDF projection and frozen human-proxy G1 ref.

These are CPU/numerical interface tests; they are NOT PhysX or actual
professional human-movement validation.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from g1_contact_coordination_projection_v040 import (
    TorchG1Relations,load_robot_source,interpolate_source,
    build_human_relative_target,soft_gate)
from g1_coordination_bridge_v039 import G1URDFFK,robot_relations
from coordination_reference_v040 import CoordinationReferenceTask
from post_shot_recovery import PostShotRecoveryTask

ROBOT=ROOT/"reports/v032_learned_center.json"
TRAJ=ROOT/"experiments/coordination_projection_v040/train_clip00001_center/g1_joint_delta_50hz.npz"
META=ROOT/"experiments/coordination_projection_v040/train_clip00001_center/generation_audit.json"
STUDY=ROOT/"configs/g1_coordination_physics_v040.json"

def test_torch_urdf_fk_matches_independent_numpy_world_link_frames():
    robot=load_robot_source(ROBOT)
    model=TorchG1Relations(robot["joint_names"])
    other=G1URDFFK()
    sample=[0,10,65,75,76,78,80,400,699]
    r=torch.tensor(robot["root"][sample],dtype=torch.float64)
    q=torch.tensor(robot["q"][sample],dtype=torch.float64,requires_grad=True)
    ours,links=model(r,q,with_links=True)
    expected=[]
    for n,i in enumerate(sample):
        fk=other.forward(robot["root"][i],robot["joint_names"],robot["q"][i])
        expected.append(robot_relations(fk))
        for foot in ("left_ankle_roll_link","right_ankle_roll_link"):
            np.testing.assert_allclose(links[foot][n,:3,3].detach().numpy(),
                                       fk[foot][:3,3],atol=3e-12,rtol=0)
    np.testing.assert_allclose(ours.detach().numpy(),np.asarray(expected),atol=3e-12,rtol=0)
    ours[:,7:9].sum().backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()
    assert float(q.grad.abs().max())>1e-5

def test_model_rejects_29joint_mismatch_and_nonfinite():
    robot=load_robot_source(ROBOT)
    with pytest.raises(ValueError):TorchG1Relations(robot["joint_names"][:-1])
    model=TorchG1Relations(robot["joint_names"])
    root=torch.tensor(robot["root"][:1],dtype=torch.float64)
    q=torch.tensor(robot["q"][:1],dtype=torch.float64)
    with pytest.raises(ValueError):model(root,q[:,:28])
    q[0,0]=float("nan")
    with pytest.raises(ValueError):model(root,q)

def test_relative_proxy_not_absolute_anatomical_target():
    robot=np.tile(np.linspace(-.8,.8,11),(20,1))
    human=np.tile(np.linspace(.5,-.5,11),(20,1))
    # Holding human pose constant must yield NO cross-morphology movement.
    target,delta=build_human_relative_target(robot,human,scale=.14)
    np.testing.assert_array_equal(target,robot)
    np.testing.assert_array_equal(delta,np.zeros_like(delta))
    human[10,5]+=1.
    target,delta=build_human_relative_target(robot,human,scale=.14)
    assert np.isclose(delta[10,5],.14*.3)
    assert np.isclose(target[10,5],min(1.,robot[10,5]+.042))

def test_software_gate_fades_to_exact_zero_before_impact():
    contact=1.57
    times=contact-.5+.04*np.arange(20)
    gate=soft_gate(times,contact)
    assert gate.shape==(20,)
    assert np.all((gate>=0)&(gate<=1))
    assert np.any(gate>0)
    assert gate[-1]==0
    assert gate[0]==0
    assert np.all(gate[times>=contact-.04-1e-9]==0)

def test_generated_reference_source_and_envelope():
    study=json.loads(STUDY.read_text())
    meta=json.loads(META.read_text())
    assert hashlib.sha256(TRAJ.read_bytes()).hexdigest()==study["generated_q_trajectory_sha256"]
    assert meta["human_clip"]=="00001"
    assert not meta["human_source_qualified"]
    assert not meta["trained_neural_generating_policy"]
    assert meta["optimization"]["coord_mse_generated"]<meta["optimization"]["coord_mse_baseline"]
    with np.load(TRAJ,allow_pickle=False) as arr:
        delta=arr["delta_joint_position_rad"]
        times=arr["controller_time_s"]
        assert list(arr["joint_names"])==load_robot_source(ROBOT)["joint_names"]
        assert not bool(arr["source_is_verified_instep"])
    assert delta.shape==(700,29) and np.isfinite(delta).all()
    assert float(abs(delta).max())<=.0060005
    assert float(abs(np.diff(delta,axis=0)).max())<=.0012005
    assert np.any(abs(delta)>1e-7)
    assert (delta[times>=1.57-.04-1e-7]==0).all()
    assert (delta[times<=1.05+1e-7]==0).all()

def test_disabled_reference_adapter_returns_original_parent_object():
    from unittest.mock import patch
    task=object.__new__(CoordinationReferenceTask)
    task.coord_enabled=False
    task.n=1
    task.debug=[]
    value=torch.randn(1,29)
    with patch.object(PostShotRecoveryTask,"commands",return_value=value) as mock:
        result=task.commands(None)
    assert result is value
    mock.assert_called_once_with(None)

def test_safety_constraints_prespecified_without_tuning_after_results():
    study=json.loads(STUDY.read_text())
    assert study["no_hyperparameter_search"]
    assert study["no_G1_policy_promotion"]
    assert not study["human_contact_time_truth_available"]
    assert study["max_full_physical_rollouts"]==4
    assert len(study["cases_in_conditional_order"])==3
    for key in ("center","heldout_v028_near_slow_17","heldout_v028_left_near_slow"):
        assert any(x["id"]==key for x in study["cases_in_conditional_order"])
