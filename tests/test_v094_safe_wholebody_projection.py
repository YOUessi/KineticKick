"""Offline support-geometry safety filter tests; never invoke real G1 actuation."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("s",ROOT/"tools/v094_safe_wholebody_projection.py")
s=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=s
spec.loader.exec_module(s)
def request(distance=.061,calibrated=True):
 Z=[[0.]*29 for _ in range(3)]
 JL=[p[:] for p in Z];JL[0][1]=1.
 JR=[p[:] for p in Z]
 G=[0.]*29;G[0]=-1.
 return s.CoordinatedVelocityRequest(
  desired_all29_joint_velocity_rad_s=[1.]*29,
  measured_all29_joint_velocity_rad_s=[0.]*29,
  all29_joint_speed_limits_rad_s=[2.]*29,
  all29_accel_limits_rad_s2=[200.]*29,
  measured_foot_normal_loads_N=[150.,0.],
  left_support_linear_J=JL,right_support_linear_J=JR,
  shape=s.CollisionShapeCertificate("/G1/wrist","/G1/hip",distance,G,True,calibrated,.005))
class TestOfflineWholeBodyGate(unittest.TestCase):
 def test_real_backend_not_calibrated_fails_closed(self):
  out=s.filter_offline_candidate(request(calibrated=False))
  self.assertFalse(out["physical_actuation_authorized"])
  self.assertFalse(out["offline_projection_feasible"])
  self.assertIn("NATIVE_PAIR_STREAM_UNCALIBRATED",out["reasons"])
 def test_missing_true_pair_fails_closed(self):
  out=s.filter_offline_candidate(replace(request(),shape=None))
  self.assertIn("MISSING_NATIVE_SHAPE_DISTANCE_CERTIFICATE",out["reasons"])
 def test_collision_and_support_constraints_actually_bound(self):
  out=s.filter_offline_candidate(request())
  self.assertTrue(out["offline_projection_feasible"],out)
  v=out["candidate_all29_rad_s"]
  self.assertLessEqual(v[0],.2+1e-6)
  self.assertLessEqual(abs(v[1]),.03+1e-6)
  self.assertEqual(len(v),29)
  self.assertFalse(out["physical_actuation_authorized"])
 def test_infeasible_clearance_rejected(self):
  r=request(distance=.01)
  r=replace(r,shape=replace(r.shape,distance_gradient_dq=[0.]*29))
  out=s.filter_offline_candidate(r)
  self.assertFalse(out["offline_projection_feasible"])
  self.assertIsNone(out["candidate_all29_rad_s"])
 def test_stale_geometry_rejected(self):
  r=request()
  out=s.filter_offline_candidate(replace(r,shape=replace(r.shape,age_s=.02)))
  self.assertIn("STALE_COLLIDER_GEOMETRY",out["reasons"])
 def test_no_planted_support_rejected(self):
  r=replace(request(),measured_foot_normal_loads_N=[10.,10.])
  out=s.filter_offline_candidate(r)
  self.assertIn("NO_CONFIRMED_PLANTED_SUPPORT",out["reasons"])
 def test_bad_29_schema_rejected(self):
  r=replace(request(),desired_all29_joint_velocity_rad_s=[0.]*28)
  out=s.filter_offline_candidate(r)
  self.assertIn("INVALID_MEASURED_29DOF_OR_STANCE_JACOBIAN",out["reasons"])
 def test_unsupported_physics_rate(self):
  out=s.filter_offline_candidate(replace(request(),dt_s=.01))
  self.assertIn("REQUIRES_200HZ_MEASURED_STATE",out["reasons"])
if __name__=="__main__":unittest.main()
