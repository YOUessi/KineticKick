"""Real local model checks for recurrent state and observation contract."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from soccer_onnx_policy import OnnxSoccerPolicy


@unittest.skipUnless((ROOT / ".pretrained_checkpoints/paid_official/policy_30000.onnx").exists(),
                     "Official weights remain local to Tang")
class OfficialPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = OnnxSoccerPolicy(ROOT / ".pretrained_checkpoints/paid_official/policy_30000.onnx", 1)

    def test_reset_restores_identical_first_action(self):
        obs = np.zeros((1, 160), np.float32)
        first = self.policy.step(obs, 0)
        self.policy.step(np.ones_like(obs), 1)
        self.policy.reset()
        np.testing.assert_array_equal(first, self.policy.step(obs, 0))
        self.assertEqual(first.shape, (1, 29))

    def test_wrong_dimensions_and_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            self.policy.step(np.zeros((1, 166), np.float32), 0)
        with self.assertRaises(ValueError):
            self.policy.step(np.full((1, 160), np.nan, np.float32), 0)

    def test_joint_order_mismatch_fails_before_rollout(self):
        wrong = self.policy.joint_names.copy()
        wrong[0], wrong[1] = wrong[1], wrong[0]
        with self.assertRaises(ValueError):
            self.policy.validate_joints(wrong)

    def test_ball_position_changes_action(self):
        obs = np.zeros((1, 160), np.float32)
        a = self.policy.step(obs, 0)
        self.policy.reset()
        obs[0, -6:-3] = [.3, .7, -.6]
        b = self.policy.step(obs, 0)
        self.assertGreater(float(np.max(abs(a - b))), 1e-4)


if __name__ == "__main__":
    unittest.main()
