"""CPU regression tests: v36 exact goal-plane geometry and stop protocol."""
from __future__ import annotations
import json,sys,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_prestrike_v036 import actual_goal_plane

class TestV36GoalPlane(unittest.TestCase):
    def test_center_vs_full_sphere_are_different_planes(self):
        rows=[dict(time_s=1.0,ball_xyz=[0.1,-1.0,.11]),
              dict(time_s=2.0,ball_xyz=[0.5,-3.0,.11])]
        midpoint=actual_goal_plane(rows,-2.,contact_s=.5)
        trailing=actual_goal_plane(rows,-2.11,contact_s=.5)
        self.assertAlmostEqual(midpoint["xyz_m"][0],.3)
        self.assertAlmostEqual(midpoint["time_s"],1.5)
        self.assertAlmostEqual(trailing["xyz_m"][0],.322)
        self.assertIsNone(actual_goal_plane(rows,-2.11,contact_s=3.))

    def test_no_unobserved_crossing_is_not_a_goal(self):
        rows=[dict(time_s=.1,ball_xyz=[0.,0.,.11]),
              dict(time_s=.2,ball_xyz=[0.,-.5,.11])]
        self.assertIsNone(actual_goal_plane(rows,-1.5,0.))

@unittest.skipUnless((ROOT/"reports/v036_prestrike_independent_audit.json").exists(),
                     "Raw v36 PhysX outcomes are ignored from GitHub")
class TestV36ActualReports(unittest.TestCase):
    def test_fixed_two_old_misses_keep_real_failure(self):
        payload=json.loads((ROOT/"reports/v036_prestrike_independent_audit.json").read_text())
        self.assertFalse(payload["both_old_misses_fixed"])
        self.assertTrue(payload["stop_protocol_applied"])
        self.assertFalse(payload["center_or_strong_run"])
        self.assertFalse(payload["new_fresh_conditions_accessed"])
        self.assertEqual(len(payload["results"]),2)
        self.assertTrue(payload["results"][0]["passed"])
        self.assertFalse(payload["results"][1]["passed"])
        self.assertLess(payload["results"][1]["full_sphere_lateral_margin_m"],0.)
        for row in payload["results"]:
            self.assertTrue(row["gates"]["real_joint_velocity_limit"])
            self.assertTrue(row["gates"]["prestrike_residual_contract"])
            self.assertTrue(row["gates"]["record_replay_matches"])
            self.assertTrue(row["gates"]["source_contract"])
            self.assertTrue(row["gates"]["protected_first_600ms"])
            self.assertGreater(row["applied_residual"]["peak_rad"],0)

if __name__=="__main__":unittest.main()
