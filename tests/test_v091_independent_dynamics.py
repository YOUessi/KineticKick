"""v91 independent 200Hz diagnostic and physical gate coverage."""
import sys
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v091_independent_dynamics import separate_true_rows,check_gates,CASES
def sample(t,terminated=False):
 return {"time_s":t,"joint_pos":[0]*29,"joint_vel":[0]*29,
  "whole_body_com_xyz":[0,0,.75],"ball_xyz":[0,0,.1],"root_xyz":[0,0,.75],
  "nonfoot_contact_N":0.,"terminated":terminated}
def test_true_step_vs_terminal_annotation_not_counted_twice():
 rows=[sample(.005),sample(.01),sample(.010001,True)]
 phys,extra=separate_true_rows(rows)
 assert len(phys)==2 and len(extra)==1
def test_malicious_terminal_state_rewritten_rejected():
 rows=[sample(.005),sample(.01),sample(.010001,True)]
 rows[-1]["root_xyz"]=[999,0,.75]
 with pytest.raises(ValueError,match="Changed physical state"):separate_true_rows(rows)
def test_true_dt_not_confused_with_terminal_marker():
 phys,annotation=separate_true_rows([sample(.005),sample(.010),sample(.015)])
 assert len(phys)==3 and len(annotation)==0
def test_all_hard_gates_conjunction_requires_instep_and_speed_and_recovery():
 ok=check_gates(2800,[],1.5,.85,1,True,4,0,True)
 assert ok["strict_pass"] is True
 assert all(ok["gates"].values())
 fail=check_gates(2800,[],1.5,.85,.7298,True,4,0,True)
 assert not fail["strict_pass"] and "instep_ge_95pct" in fail["failed"]
 fail2=check_gates(2800,[],1.5,.85,1,True,3.49,0,True)
 assert not fail2["strict_pass"] and "ball_speed_proxy_ge_3p5mps" in fail2["failed"]
def test_early_termination_not_14s_recovery():
 r=check_gates(320,[sample(1.600001,True)],1.565,.9,1,True,5.14,180,False)
 assert not r["strict_pass"] and "full_14s_original_physx" in r["failed"]
 assert "nonfoot_strict_lt20N" in r["failed"]
def test_joint_overspeed_and_no_contact_both_rejected():
 r=check_gates(2800,[],None,1.02,1,True,None,0,True)
 assert not r["strict_pass"]
 assert "true_29dof_speed_limits" in r["failed"]
 assert "physically_observed_foot_ball_contact" in r["failed"]
def test_v91_matrix_four_separate_physical_executions():
 assert len(CASES)==4 and len(set(CASES))==4
