"""Reject slow rolling and recovery re-strikes even when the legacy trap passed."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import test_inside_receive_metrics as fixtures
from balanced_receive_metrics import summarize_balanced_receive

class BalancedReceiveAcceptance(unittest.TestCase):
    def scenario(self):
        r=fixtures.InsideReceiveAcceptance().scenario('ground')
        r['inside_control_events']=[dict(kind='learned_standing_recovery',time_s=.30,ball_foot_clearance_m=.04)]
        return r
    def test_clear_release_and_retained_ball(self):
        self.assertTrue(summarize_balanced_receive(self.scenario(),'ground')['success'])
    def test_rolling_ball_is_not_stopped(self):
        r=self.scenario()
        for row in r['traces'][0][30:]:row['ball_velocity']=[.12,0.,0.]
        value=summarize_balanced_receive(r,'ground')
        self.assertTrue(value['loose_v1_success']);self.assertFalse(value['success'])
    def test_recovery_foot_strike_rejected(self):
        r=self.scenario();r['traces'][0][35]['robot_ball_contact_force_N']={'right_ankle_roll_link':4.}
        self.assertFalse(summarize_balanced_receive(r,'ground')['success'])

if __name__=='__main__':unittest.main()
