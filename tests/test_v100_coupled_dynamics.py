import copy, math, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import pytest
from v100_coupled_dynamics import require_joint_groups, physical_grid, describe_episode,canonical_hash

def names():
 return ['waist_'+str(i) for i in range(3)]+['left_hip_'+str(i) for i in range(3)]+['left_knee_x']+['left_ankle_'+str(i) for i in range(2)]+['right_hip_'+str(i) for i in range(3)]+['right_knee_x']+['right_ankle_'+str(i) for i in range(2)]+['left_shoulder_'+str(i) for i in range(3)]+['left_elbow_x']+['left_wrist_'+str(i) for i in range(3)]+['right_shoulder_'+str(i) for i in range(3)]+['right_elbow_x']+['right_wrist_'+str(i) for i in range(3)]

def fixture(n=2800,contact=2.):
 vals=[]
 for k in range(n):
  t=.005*(k+1)
  vals.append(dict(time_s=t,joint_pos=[0.]*29,joint_vel=[.02*(j+1) for j in range(29)],whole_body_com_xyz=[0.,-t,.68],
   support_foot_force_N={'left':75.,'right':15.},nonfoot_contact_N=0.))
 return dict(joint_names=names(),joint_velocity_limits=[[10.]*29],records=[dict(contact_time_s=contact)],
             traces=[vals])
def test_groups_all29():
 g=require_joint_groups(names())
 assert set(g)=={'waist','left_leg','right_leg','left_arm','right_arm'}
 assert sum(map(len,g.values()))==29
 with pytest.raises(ValueError):require_joint_groups(names()[:-1])
def test_complete_and_censored():
 d=describe_episode(fixture())
 assert d['actual_unique_200hz_steps']==2800 and d['completed_full_14s']
 assert d['pre_400ms_observed'] and d['post_1p25_observed']
 assert d['support_sensor_N_not_expert_GRF']['left']['precontact_contact_fraction_20N']==1.
 assert d['support_sensor_N_not_expert_GRF']['right']['precontact_contact_fraction_20N']==0.
 assert not d['post_contact_window_right_censored']
 d=describe_episode(fixture(n=420))
 assert d['pre_400ms_observed'] and not d['post_1p25_observed']
 assert d['post_contact_window_right_censored'] and d['late_recovery_unknown']
 assert d['support_sensor_N_not_expert_GRF']['left']['post_1p25_contact_fraction_20N'] is None
def test_terminal_marker_not_real_step():
 f=fixture(n=420)
 last=copy.deepcopy(f['traces'][0][-1]);last['time_s']+=1e-6
 f['traces'][0].append(last)
 d=describe_episode(f)
 assert d['terminal_duplicate_rows']==1 and d['actual_unique_200hz_steps']==420
def test_strict_200hz_fail_closed():
 f=fixture();f['traces'][0][20]['time_s']+=.002
 with pytest.raises(ValueError,match='NOT_TRUE_200HZ_GRID'):describe_episode(f)
def test_invalid_speed_and_human_claim_denied():
 f=fixture();f['traces'][0][0]['joint_vel'][0]=100.
 d=describe_episode(f)
 assert not d['actual29_qdot_all_within_limits']
 assert d['descriptive_only_not_elite_human_naturalness']
 assert canonical_hash(f['traces'])!=canonical_hash(f['records'])

def test_nonterminal_extra_step_rejected():
 f=fixture(n=420)
 f['traces'][0][15]['time_s']=f['traces'][0][14]['time_s']+1e-6
 with pytest.raises(ValueError):describe_episode(f)
def test_invalid_29_joint_position_shape_rejected():
 f=fixture(n=420);f['traces'][0][0]['joint_pos']=[0.]*28
 with pytest.raises(ValueError,match='INVALID_joint_pos'):describe_episode(f)
def test_raw_human_grf_cannot_be_claimed():
 f=fixture(n=420)
 d=describe_episode(f)
 assert d['support_sensor_N_not_expert_GRF']['left']['not_force_plate_GRF']
 assert d['descriptive_only_not_elite_human_naturalness']
