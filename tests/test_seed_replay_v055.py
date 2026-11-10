"""Seed audit tests use synthetic fixtures only; never claim fixture = real physics."""
import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from audit_seed_replay_v055 import compare_seed_pair,EVIDENCE_KEYS

class TestSeedReplayIndependenceV055(unittest.TestCase):
    def physical_fixture(self):
        return dict(seed=1201,traces=[[{"time_s":.005,"joint_pos":[0.]*29}]],
                    contacts=[],records=[{"goal":False}],sequence_records=[],
                    termination_reasons=[["physical_nonfoot"]],
                    nonfoot_contact_diagnostics=[{"body":"left_hip_roll_link"}])
    def test_different_metadata_seed_does_not_create_independent_motion(self):
        a=self.physical_fixture()
        b=copy.deepcopy(a);b["seed"]=1509
        c=compare_seed_pair(a,b)
        self.assertTrue(c["identical_physics_and_contact_evidence"])
        self.assertTrue(all(c["compared_fields"][k] for k in EVIDENCE_KEYS))
    def test_actual_physics_change_is_detected(self):
        a=self.physical_fixture()
        b=copy.deepcopy(a);b["seed"]=1509
        b["traces"][0][0]["joint_pos"][6]=.001
        c=compare_seed_pair(a,b)
        self.assertFalse(c["identical_physics_and_contact_evidence"])
        self.assertFalse(c["identical_200hz_joint_traces"])
    def test_contact_change_is_detected_without_trajectory_change(self):
        a=self.physical_fixture()
        b=copy.deepcopy(a);b["seed"]=1509
        b["nonfoot_contact_diagnostics"]=[]
        c=compare_seed_pair(a,b)
        self.assertFalse(c["identical_physics_and_contact_evidence"])
        self.assertTrue(c["identical_200hz_joint_traces"])
    def test_same_seed_rejected(self):
        a=self.physical_fixture()
        with self.assertRaises(ValueError):compare_seed_pair(a,copy.deepcopy(a))
if __name__=="__main__":unittest.main()
