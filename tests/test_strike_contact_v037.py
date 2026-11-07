"""v37: registered read-only paired contact evidence and direction math.

Runs without PhysX; Tang-local raw files are optional for public-code clones.
"""
from pathlib import Path
import json,sys,unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_ball_impulse import heading,heading_change,balance_bins
from compare_strike_contact_v037 import paired,old_v33,vfoot

class TestMeasuredImpulseMath(unittest.TestCase):
    def test_heading_additive_impulses(self):
        normal=np.asarray([.5,-2.,.2]);tangent=np.asarray([-.8,.1,-.05])
        self.assertGreater(heading(normal),0)
        self.assertLess(heading(normal+tangent),0)
        self.assertAlmostEqual(heading_change(normal,normal+tangent),
                               heading(normal+tangent)-heading(normal),places=8)

    def test_balance_rejects_missing_contact_force(self):
        row=dict(step_start_s=0.005,step_end_s=.01,
                 velocity_before=[0,0,0],velocity_after=[1,0,0],
                 pairs={name:{"normal":[0,0,0],"tangent":[0,0,0]}
                        for name in ("right_foot","left_foot","other_robot","ground")})
        x=balance_bins([row],.43,[0,0,-9.81])
        self.assertFalse(x["passed"])
        self.assertGreater(x["maximum_bin_residual_Ns"],.02)

    def test_strict_reference_metadata_cannot_be_silently_ignored(self):
        from analyze_strike_contact_v037 import controlled_semantic_equivalence
        with self.assertRaises((KeyError,ValueError)):
            controlled_semantic_equivalence({}, {}, dict(changed_fields=["unexpected"],
                missing_fields=[],streams={"traces":{"equal":True}}),{})

DATA=(ROOT/"reports/v037_paired_contact_mechanics.json")
@unittest.skipUnless(DATA.exists(),"Tang-only raw audited contact evidence unavailable")
class TestRealSignedImpulseEvidence(unittest.TestCase):
    def setUp(self):
        self.m=json.loads(DATA.read_text())
    def test_all_three_replays_are_audited(self):
        self.assertTrue(self.m["all_readonly_parity"])
        self.assertTrue(self.m["all_momentum_balances_verified"])
        self.assertEqual(len(self.m["recorded_cases"]),4)
        for row in self.m["recorded_cases"]:
            if row.get("max_5ms_balance_residual_Ns") is not None:
                self.assertLess(row["max_5ms_balance_residual_Ns"],.02)
    def test_trained_candidate_succeeds_in_only_first_old_miss(self):
        status=self.m["original_full_ball_goal_results"]
        self.assertTrue(status["v36_A"]);self.assertFalse(status["v36_B"])
    def test_full_contact_vs_normal_only_changes_direction(self):
        for k in ("left_near_slow","near_slow_17"):
            p=self.m["matched_pairs"][k]
            self.assertGreater(abs(p["new_tangent_heading_effect_deg"]),1.0)
    def test_matched_candidate_improves_one_case_but_worsens_other(self):
        a=self.m["matched_pairs"]["left_near_slow"]
        b=self.m["matched_pairs"]["near_slow_17"]
        self.assertGreater(a["total_heading_change_deg"],0.)
        self.assertLess(b["total_heading_change_deg"],0.)
    def test_archived_v32_b_provenance_exceptions_are_explicit(self):
        p=ROOT/"reports/v037_audit_v32_near_slow_17.json"
        x=json.loads(p.read_text())["acquisition"]
        self.assertTrue(x["physical_action_streams_exact"])
        self.assertFalse(x["full_original_metadata_identical"])
        self.assertEqual(len(x["documented_source_exceptions"]),3)
        self.assertTrue(x["read_only_exact_replay"])

if __name__=="__main__":unittest.main()
