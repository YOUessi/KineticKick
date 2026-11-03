"""Regression checks over ACTUAL ISAAC LAB POWER CALIBRATION OUTPUT.
These assert empirical observations from a fixed-base script, not a learned policy.
"""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class TestMeasuredKicks(unittest.TestCase):
    def setUp(self):
        self.samples=json.loads((ROOT/"reports/calibrate_power.json").read_text())
        self.cfg=json.loads((ROOT/"configs/kick_modes_v1.json").read_text())

    def test_all_3_modes_and_power_levels_have_measured_ball_impacts(self):
        self.assertEqual(len(self.samples),9)
        self.assertEqual({x["kind"] for x in self.samples},{"instep","medial","lateral"})
        for x in self.samples:
            self.assertIsNotNone(x["impact"])
            self.assertGreater(x["peak_contact_force_N"],20)
            self.assertGreater(x["peak_horizontal_ball_speed_mps"],0.7)

    def test_power_levels_increase_measured_speed_per_kick(self):
        for mode in ("instep","medial","lateral"):
            speeds={}
            for row in self.samples:
                if row["kind"]==mode:
                    speeds[row["power"]]=row["peak_horizontal_ball_speed_mps"]
            self.assertLess(speeds["soft"],speeds["medium"])
            self.assertLess(speeds["medium"],speeds["strong"])

    def test_original_contact_surfaces(self):
        for x in self.samples:
            face=x["impact"]["face_est"]
            if x["kind"]=="medial":self.assertEqual(face,"medial")
            if x["kind"]=="lateral":self.assertEqual(face,"lateral")
            if x["kind"]=="instep":
                self.assertIn(face,{"toe","sole"}) # honestly NOT true instep
        self.assertNotIn("soccer_shoe",self.cfg["foot"].lower())

    def test_duration_monotonic_and_reference_robot_unchanged(self):
        durations=self.cfg["power_duration_seconds"]
        self.assertGreater(durations["soft"],durations["medium"])
        self.assertGreater(durations["medium"],durations["strong"])
        self.assertIn("37DOF",self.cfg["robot"])
        self.assertIn("original",self.cfg["foot"])

    def test_knee_only_is_a_distinct_but_uncalibrated_motor_pattern(self):
        rows=json.loads((ROOT/"reports/calibrate_knee_power.json").read_text())
        self.assertEqual(len(rows),3)
        self.assertEqual({x["kind"] for x in rows},{"knee_only"})
        self.assertTrue(all(x["impact"] is not None for x in rows))
        by={x["power"]:x["peak_horizontal_ball_speed_mps"] for x in rows}
        # A faster knee command is not guaranteed to make the ball faster:
        self.assertGreater(by["medium"],by["soft"])
        self.assertGreater(by["medium"],by["strong"])
        knee=self.cfg["modes"]["knee_only"]
        self.assertEqual(knee["hip_start"],knee["hip_end"])

    def test_top_of_original_g1_foot_instep_strike(self):
        rows=json.loads((ROOT/"reports/dorsal_kick_power.json").read_text())
        self.assertEqual(len(rows),3)
        self.assertIn("instep_top",self.cfg["modes"])
        powers={r["power"]:r for r in rows}
        self.assertEqual(set(powers),{"soft","medium","strong"})
        for level,r in powers.items():
            impact=r["impact"]
            self.assertIsNotNone(impact)
            self.assertEqual(impact["face_est"],"top")
            self.assertTrue(impact["top_patch_confident"])
            self.assertGreater(impact["top_normal_world"][0],0.80)
            self.assertGreater(r["peak_horizontal_ball_speed_mps"],2.0)
        self.assertLess(powers["soft"]["peak_horizontal_ball_speed_mps"],
                        powers["medium"]["peak_horizontal_ball_speed_mps"])
        self.assertLess(powers["medium"]["peak_horizontal_ball_speed_mps"],
                        powers["strong"]["peak_horizontal_ball_speed_mps"])

if __name__=="__main__":
    unittest.main()
