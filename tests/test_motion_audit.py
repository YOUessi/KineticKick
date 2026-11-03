import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from motion_audit import (
    compare_to_reference, inspect_coordination, lag_seconds, load_clip,
    summarize, velocity_from_positions
)


def sample(n=101, fps=50):
    t = np.arange(n) / fps
    q = np.stack([np.sin(t), np.cos(t)], axis=1)
    jv = np.stack([np.cos(t), -np.sin(t)], axis=1)
    pos = np.zeros((n, 3, 3))
    pos[:, 0, 0] = t
    quat = np.zeros((n, 3, 4))
    quat[..., 0] = 1
    return {
        "fps": np.array([fps]),
        "joint_pos": q, "joint_vel": jv,
        "body_pos_w": pos, "body_quat_w": quat,
        "body_lin_vel_w": np.zeros((n, 3, 3)),
        "body_ang_vel_w": np.zeros((n, 3, 3)),
    }


class TestAudit(unittest.TestCase):
    def test_validation_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "soccer-standard-001.npz"
            np.savez_compressed(p, **sample())
            clip = load_clip(p)
            row = summarize(clip, p)
            self.assertEqual(row["frames"], 101)
            self.assertEqual(row["joints"], 2)
            self.assertAlmostEqual(row["root_path_length_m"], 2.0, places=5)
            self.assertLess(row["pos_velocity_consistency_mae_rad_s"], .001)
            self.assertEqual(row["source_kind"], "reference_motion_not_robot_rollout")

    def test_prevent_nan(self):
        with tempfile.TemporaryDirectory() as d:
            arr = sample()
            arr["joint_pos"][3, 1] = np.nan
            p = Path(d) / "bad.npz"
            np.savez_compressed(p, **arr)
            with self.assertRaises(ValueError):
                load_clip(p)

    def test_comparison_same_motion(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "motion.npz"
            np.savez_compressed(p, **sample())
            clip = load_clip(p)
            result = compare_to_reference(clip, clip)
            self.assertLess(result["joint_angle_mae_rad"], 1e-10)
            self.assertLess(result["root_centered_body_pos_mpJPE_m"], 1e-10)

    def test_lag_sign(self):
        b = np.zeros(100)
        b[20:25] = [0.25, 0.75, 1., .75, .25]
        a = np.zeros_like(b)
        a[3:] = b[:-3]
        self.assertAlmostEqual(lag_seconds(a, b, 50), .06, places=5)
        self.assertIsNone(lag_seconds(a, np.zeros_like(b), 50))

    def test_velocity_numeric(self):
        n = 300
        fps = 100
        t = np.arange(n) / fps
        q = np.sin(t)[:, None]
        v = velocity_from_positions(q, fps).squeeze()
        self.assertLess(np.max(abs(v[10:-10] - np.cos(t[10:-10]))), .002)

    def test_reject_cross_variant_semantic_joint_map(self):
        clip = sample()
        clip["fps"] = 50.0
        # Built-in G1 joint count cannot be used for PAiD 29-DoF or other schema.
        with self.assertRaisesRegex(ValueError, "matching _num_joints"):
            inspect_coordination(clip, {"_num_joints": 37, "_verified_against_source": True})
        with self.assertRaisesRegex(ValueError, "_verified_against_source"):
            inspect_coordination(clip, {"_num_joints": 2, "_verified_against_source": False})


if __name__ == "__main__":
    unittest.main()
