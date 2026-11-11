"""v64 physical audit: regression on null/inert controllers and ball-only cheats."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_v064_real_human_physics import peak_ball_speed,action_magnitude,physical_kinetics_proxies

def test_read_200hz_ball_speed_window():
 raw=dict(traces=[[dict(time_s=t,ball_velocity=[x,0,0])
  for t,x in [(0.0,0.),(1.0,0.),(1.5,2.),(1.57,7.),(1.575,4.2),(1.60,5.7),(1.7,9.)]]],
  records=[dict(contact_time_s=1.57)])
 assert abs(peak_ball_speed(raw)-5.7)<1e-9

def test_no_measured_contact_no_claim():
 raw=dict(traces=[[dict(time_s=0,ball_velocity=[0,0,0])]],records=[dict(contact_time_s=None)])
 assert peak_ball_speed(raw) is None
 assert not physical_kinetics_proxies(raw)["contact_observed"]

def test_inactive_candidate_is_explicit_negative():
 assert not action_magnitude({})["active"]
 trial=dict(human_phase_v064_summary=dict(mode="human_ordered",nonzero_20ms_control_steps=0,
  eligible_20ms_control_steps=0,max_action_delta_rad=0))
 z=action_magnitude(trial)
 assert z["active"] and z["actual_nonzero_20ms"]==0 and z["actual_max_delta_rad"]==0

def test_joint_torque_power_proxy_not_human_measure():
 q=dict(time_s=1.2,joint_vel=[1.0]*29,joint_torque_estimate=[2.0]*29,ball_velocity=[0,0,0])
 raw=dict(records=[dict(contact_time_s=1.2)],traces=[[q]])
 m=physical_kinetics_proxies(raw)
 assert m["human_muscular_power_truth"] is False
 assert m["robotics_PD_torque_qdot_abs_integral_proxy_J"]==pytest.approx(29*.005*2.)
