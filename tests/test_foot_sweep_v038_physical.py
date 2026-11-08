"""Frozen v38 bad-zero -> corrected-zero -> physical negative decision gates."""
from pathlib import Path
import json,unittest,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))

@unittest.skipUnless((ROOT/"reports/v038_candidate_near_slow_17_audit.json").exists(),
                     "Tang-local raw physics required")
class TestV38OriginalPhysicalEvidence(unittest.TestCase):
    def setUp(self):
        self.study=json.loads((ROOT/"configs/foot_sweep_direction_study_v038.json").read_text())
        self.zero=json.loads((ROOT/"reports/v038_zero_corrected_independent_audit_v3.json").read_text())
        self.fail=json.loads((ROOT/"reports/v038_candidate_near_slow_17_audit.json").read_text())
    def test_original_disabled_bug_cannot_be_reported_as_valid(self):
        raw=self.study["initial_invalid_zero_attempt"]
        self.assertFalse(raw["physics_acceptance_claim"])
        self.assertTrue(raw["original_failed_output_path"])
        self.assertIn("enabled=False",raw["status"])
        self.assertIn("physical_nonfoot",str(raw["original_physics_terminated"]))
    def test_corrected_disabled_zero_preserves_actual_original_miss(self):
        self.assertEqual(self.zero["status"],"pass")
        self.assertTrue(self.zero["guard"]["zero_equivalence"])
        self.assertFalse(self.zero["original_task"]["success"])
        self.assertTrue(all(self.zero["zero_exact_stream"].values()))
        self.assertEqual(set(self.zero["zero_max_abs_state_diffs"].values()),{0.0})
        self.assertEqual(self.zero["actual_feedback_active_decisions"],0)
        self.assertEqual(self.zero["physical_steps"],2800)
    def test_candidate_negative_cannot_be_promoted_by_speed(self):
        self.assertEqual(self.fail["status"],"fail")
        self.assertFalse(self.fail["guard"]["physical_complete"])
        self.assertFalse(self.fail["guard"]["real_qdot"])
        self.assertFalse(self.fail["guard"]["original_standing"])
        self.assertFalse(self.fail["guard"]["original_full_goal"])
        self.assertTrue(self.fail["guard"]["speed_retention"])
        self.assertLess(self.fail["physical_steps"],2800)
        self.assertIn("physical_nonfoot",str(self.fail["termination_reasons"]))
        self.assertGreater(self.fail["maximum_measured_joint_speed_ratio"],1.)
        self.assertLess(self.fail["full_ball_goal_margin"]["signed_nearest_lateral_margin_m"],0)
    def test_prespecified_stop_protects_unseen_cases(self):
        for case in self.study["candidate_cases_order"][1:]:
            self.assertFalse((ROOT/case["candidate"]).exists())
        self.assertEqual(self.study["new_fresh_cases_used"],False)

if __name__=="__main__":unittest.main()
