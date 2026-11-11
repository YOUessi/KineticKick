"""Synthetic unit fixtures are NOT real PhysX clearance evidence."""
import numpy as np
from tools.stance_safe_temporal_policy_v067 import (
    GROUPS,G1_JOINT_ORDER,N,NativeGeometryWitness,RealG1State,stage_29dof_action)

def state(witness=None,**kw):
    args=dict(joint_names=G1_JOINT_ORDER,physical_frame_time_s=1.46,
       joint_pos_rad=np.zeros(N),measured_joint_vel_rad_s=np.zeros(N),
       actual_joint_vel_limits_rad_s=np.ones(N)*5.,
       actual_joint_pos_limits_rad=np.tile([[-3.,3.]],(N,1)),
       measured_left_support_normal_N=140.,
       measured_right_support_normal_N=0.,
       measured_nonfoot_contact_N=0.,
       measured_body_tilt_deg=1.,
       measured_support_margin_m=.10,
       actual_collider_witness=witness,ball_contact_occurred=False)
    args.update(kw)
    return RealG1State(**args)
def vectors():
    return np.zeros(N),np.ones(N)*.045,np.zeros(N),np.ones(N)*.25
def test_missing_GPU_collision_groundtruth_returns_identical_original_action():
    orig,goal,prev,scale=vectors();orig[:]=.13
    a=stage_29dof_action(orig,goal,prev,scale,state())
    assert np.array_equal(a.action,orig)
    assert a.reason=='no_verified_original_GPU_G1_mesh_collider_witness'
    assert a.eligible_for_physx_deployment is False
    assert a.certified_hardware_collision_safety is False
def test_witness_fixture_exercises_nontrivial_wholebody_bounded_interface():
    witness=NativeGeometryWitness('cuda:0','/World/envs/env_0/Robot/left_wrist_yaw_link/collisions',
                   '/World/envs/env_0/Robot/left_hip_roll_link/collisions',.06,True,True)
    orig,goal,prev,scale=vectors()
    decision=stage_29dof_action(orig,goal,prev,scale,state(witness))
    assert decision.eligible_for_physx_deployment is True
    assert np.count_nonzero(decision.actual_joint_delta_rad)>20
    assert 0<max(decision.actual_joint_delta_rad)<=.006
    assert decision.certified_hardware_collision_safety is False
    for group,names in GROUPS.items():
        assert any(decision.actual_joint_delta_rad[G1_JOINT_ORDER.index(x)]>0 for x in names)
    assert stage_29dof_action(orig,goal,prev,scale,
                state(witness,measured_nonfoot_contact_N=10.)).eligible_for_physx_deployment is False
    assert stage_29dof_action(orig,goal,prev,scale,
                state(witness,measured_support_margin_m=None)).eligible_for_physx_deployment is False
    assert stage_29dof_action(orig,goal,prev,scale,
                state(witness,ball_contact_occurred=True)).eligible_for_physx_deployment is False
def test_missing_29_joint_order_and_lone_foot_correction_rejected():
    orig,goal,prev,scale=vectors()
    try:stage_29dof_action(orig,goal,prev,scale,state(joint_names=tuple(reversed(G1_JOINT_ORDER))))
    except ValueError:pass
    else:raise AssertionError('Wrong joint ordering accepted')
    a=stage_29dof_action(orig,goal,prev,scale,state())
    assert not a.eligible_for_physx_deployment
