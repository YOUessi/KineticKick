from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import test_balanced_receive_metrics as fixtures
from reactive_receive_metrics import summarize_reactive_receive

class ReactiveAcceptanceTests(unittest.TestCase):
    def scenario(self):
        r=fixtures.BalancedReceiveAcceptance().scenario()
        r['reactive_events']=[dict(kind='react_to_observed_ball',time_s=.05)]
        for s in r['inside_sensor_trace']:s['right_foot_position']=[0,0,.035]
        return r
    def test_grounded_reaction_with_retained_ball_passes(self):
        self.assertTrue(summarize_reactive_receive(self.scenario(),'ground')['success'])
    def test_prepositioned_foot_is_rejected(self):
        r=self.scenario()
        for s in r['inside_sensor_trace']:s['right_foot_position'][2]=.25
        self.assertFalse(summarize_reactive_receive(r,'ground')['success'])
    def test_previous_residual_speed_is_rejected(self):
        r=self.scenario()
        for row in r['traces'][0][30:]:row['ball_velocity']=[.034,0,0]
        value=summarize_reactive_receive(r,'ground')
        self.assertTrue(value['balanced_v2_success']);self.assertFalse(value['success'])

if __name__=='__main__':unittest.main()
