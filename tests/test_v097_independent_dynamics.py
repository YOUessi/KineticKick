"""Fail-closed 200Hz event-time accounting and independent 29DoF gates."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import v097_independent_dynamics as audit
def test_exact_locked_cases():
 assert audit.CASES==('G_v32','G_cmu02','G_cmu05','H_v32','H_cmu02','H_cmu05')
def test_terminal_marker_is_not_physical_step():
 row={'time_s':1.0,'joint_pos':[0]*29,'joint_vel':[0]*29,'whole_body_com_xyz':[0,0,1],
      'ball_xyz':[0,0,0],'root_xyz':[0,0,1],'nonfoot_contact_N':0,'terminated':False}
 term=dict(row,time_s=1.000001,terminated=True)
 core,markers=audit.separate_true_rows([row,term])
 assert len(core)==1 and len(markers)==1
def test_terminal_state_drift_rejected():
 row={'time_s':1.0,'joint_pos':[0]*29,'joint_vel':[0]*29,'whole_body_com_xyz':[0,0,1],
      'ball_xyz':[0,0,0],'root_xyz':[0,0,1],'nonfoot_contact_N':0,'terminated':False}
 term=dict(row,time_s=1.000001,terminated=True,joint_pos=[1]*29)
 with pytest.raises(ValueError): audit.separate_true_rows([row,term])
def test_goal_does_not_override_legal_instep_and_recovery():
 result=audit.check_gates(2800,[],1.1,0.95,0.7298,True,5.1,0.,True)
 assert not result['strict_pass'] and 'instep_ge_95pct' in result['failed']
 result=audit.check_gates(785,[{'terminated':True}],1.1,0.4,1.,True,5.1,25.89,False)
 assert not result['strict_pass'] and '14s_recovery' in result['failed']
