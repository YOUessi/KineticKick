"""Proof-of-enforcement tests for offline-only 200Hz/29DoF joint projection."""
import importlib.util,sys
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1]/"tools/v098_safe_trajectory_filter.py"
sp=importlib.util.spec_from_file_location("v098_safe_project",SRC)
s=importlib.util.module_from_spec(sp);sys.modules[sp.name]=s;sp.loader.exec_module(s)
def measured(q=None,v=None,a=200.):
 return s.MeasuredG1(q0=[0.]*29 if q is None else q,
   dq0=[0.]*29 if v is None else v,urdf_low=[-1.]*29,
   urdf_high=[1.]*29,original_speed_limit=[2.]*29,
   conservative_accel_limit=[a]*29,urdf_source_sha_verified=True)
def wanted(mag=0.):
 return np.full((20,29),mag,dtype=float)
def test_all29_80_substeps_bounded():
 out=s.project_full29(wanted(.25),measured())
 assert out["offline_joint_kinematics_feasible"]
 assert not out["physical_actuation_authorized"]
 q=np.asarray(out["q_200hz"]);v=np.asarray(out["dq_200hz"])
 assert q.shape==v.shape==(80,29)
 assert (abs(v)<=2.0000001).all()
 assert (abs(q)<=.99700001).all()
 assert np.max(np.abs(np.diff(v,axis=0)))<=1.0000001
def test_outward_high_limit_intervention():
 q=[0.]*29;q[0]=.995
 out=s.project_full29(wanted(1.),measured(q=q))
 assert out["offline_joint_kinematics_feasible"]
 assert out["interventions"]>0
 assert max(np.asarray(out["q_200hz"])[:,0])<=.99700001
 assert not out["physical_actuation_authorized"]
def test_unsafe_actual_q_not_hidden_by_projection():
 q=[0.]*29;q[0]=1.1
 out=s.project_full29(wanted(),measured(q=q))
 assert not out["offline_joint_kinematics_feasible"]
 assert out["rejection"]=="ACTUAL_Q0_OUTSIDE_CONSERVATIVE_URDF_BOUNDS"
def test_unstoppable_near_joint_limit():
 q=[0.]*29;q[0]=.995
 v=[0.]*29;v[0]=1.
 out=s.project_full29(wanted(),measured(q,v,a=1.))
 assert not out["offline_joint_kinematics_feasible"]
 assert out["rejection"]=="UNSTOPPABLE_CURRENT_MOTION_OR_EMPTY_JOINT_INTERSECTION"
def test_unknown_true_urdf_denies():
 r=measured()
 from dataclasses import replace
 out=s.project_full29(wanted(),replace(r,urdf_source_sha_verified=False))
 assert out["rejection"]=="UNQUALIFIED_SOURCE_URDF_SHA"
def test_corrupted_input_or_limit_denies():
 out=s.project_full29(np.zeros((20,28)),measured())
 assert out["rejection"]=="INVALID_FULL29_MEASUREMENT_OR_CANDIDATE"
 assert s.project_full29(np.full((20,29),np.nan),measured())["offline_joint_kinematics_feasible"] is False
def test_cannot_grant_motor_authority_even_with_successful_projection():
 out=s.project_full29(wanted(),measured())
 assert "MISSING_NATIVE_AUTHENTICATED_COLLIDER_PAIR_AND_CONTACT_NORMAL" in out["required_qualifications"]
 assert "MISSING_INVERSE_DYNAMICS_TORQUE_EFFORT_LIMIT_CERTIFICATION" in out["required_qualifications"]
 assert "MISSING_QUALIFIED_ELITE_HUMAN_MOTION_AND_REAL_IMPULSE_LABEL" in out["required_qualifications"]
 assert out["physical_actuation_authorized"] is False
