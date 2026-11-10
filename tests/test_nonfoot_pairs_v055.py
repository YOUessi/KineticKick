"""Synthetic tests for net-force hypotheses; they do NOT certify actual pair truth."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from scan_nonfoot_force_pairs_v055 import pair_hypotheses
class TestForcePairV055(unittest.TestCase):
    def sample(self, vec_a=(12.,16.,0.),vec_b=(-12.,-16.,0.),time_a=5.145,time_b=5.145):
        return [dict(time_s=time_a,body="left_wrist_yaw_link",force_w_N=list(vec_a)),
                dict(time_s=time_b,body="left_hip_roll_link",force_w_N=list(vec_b))]
    def test_balanced_pair_is_only_hypothesis(self):
        r=pair_hypotheses(self.sample())
        self.assertEqual(len(r),1)
        self.assertFalse(r[0]["true_physx_contact_pair_verified"])
        self.assertFalse(r[0]["true_collider_paths_verified"])
    def test_unrelated_time_is_not_a_pair(self):
        self.assertFalse(pair_hypotheses(self.sample(time_b=5.160)))
    def test_unbalanced_and_weak_pairs_rejected(self):
        self.assertFalse(pair_hypotheses(self.sample(vec_b=(-12.,-15.,0.))))
        self.assertFalse(pair_hypotheses(self.sample(vec_a=(1.,2.,0.),vec_b=(-1.,-2.,0.))))
    def test_malformed_nonfinite_force_rejected(self):
        with self.assertRaises(ValueError):
            pair_hypotheses(self.sample(vec_a=(float("nan"),2.,3.)))
if __name__=="__main__":unittest.main()
