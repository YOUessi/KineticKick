import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from running_sequence_metrics import summarize_recovery


class RecoveryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(time_s=(i+1)*.005,physics_dt_s=.005,root_xyz=[0,0,.8],root_tilt_deg=0,
            nonfoot_contact_N=0,root_linear_velocity=[0,0,0],root_angular_velocity=[0,0,0],
            support_foot_force_N={'left':150,'right':150}) for i in range(1000)]

    def passed(self):
        return summarize_recovery(self.rows,expected_duration_s=5)['standing_success']

    def test_complete_stationary_supported_hold(self):
        self.assertTrue(self.passed())

    def test_missing_physics_evidence_is_not_a_hold(self):
        del self.rows[950]
        self.assertFalse(self.passed())

    def test_late_motion_does_not_pass_from_last_frame_alone(self):
        self.rows[900]['root_linear_velocity']=[1,0,0]
        self.assertFalse(self.passed())

    def test_earlier_fall_cannot_be_erased_by_standing_end(self):
        self.rows[10]['terminated']=True
        self.assertFalse(self.passed())
