import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from task_strike_rewards import task_focused_value

class TaskFocusedReward(unittest.TestCase):
    def test_surviving_after_missed_shot_cannot_match_a_scored_shot(self):
        r=task_focused_value(torch.tensor([10.,10.,-100.]),torch.tensor([150.,150.,150.]),torch.tensor([False,True,False]),torch.tensor([0.,1.,0.]),torch.zeros(3))
        self.assertLessEqual(float(r[0]),.5)
        self.assertEqual(float(r[1]),10.)
        self.assertEqual(float(r[2]),-100.)

    def test_goal_event_rewards_contact_quality_and_approach_stays_guided(self):
        r=task_focused_value(torch.tensor([10.,10.,10.]),torch.tensor([50.,150.,150.]),torch.tensor([False,True,True]),torch.tensor([0.,1.,.2]),torch.tensor([0.,1.,1.]))
        self.assertEqual(float(r[0]),10.)
        self.assertGreater(float(r[1]),10*float(r[2]))

if __name__=='__main__':unittest.main()
