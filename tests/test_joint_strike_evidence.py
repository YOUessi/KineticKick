import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from joint_strike_evidence import validate_feedback

class FeedbackEvidence(unittest.TestCase):
    def setUp(self):
        self.config=dict(maximum_joint_residual_rad=.25,maximum_phase_shift_s=.15,phase_rate_limit=.30)
        self.report=dict(points=[[0.,-4.]],kinematics=[[{'time_s':0.},{'time_s':.02}]],
            traces=[[{'time_s':(i+1)*.005,'physics_dt_s':.005,'contact_force_N':3. if i>=3 else 0.,'terminated':False,'ball_xyz':[0.,-4.,.113],'robot_ball_contact_force_N':{'right_ankle_roll_link':3. if i>=3 else 0.}} for i in range(8)]],
            joint_strike={'feedback':[
                dict(time_s=0.,phase_s=0.,phase_action=0.,residual_rad=[0.]*29,contact_latched=False,measured_ball_xyz=[0.,-4.,.113]),
                dict(time_s=.02,phase_s=.02,phase_action=0.,residual_rad=[0.]*29,contact_latched=True,measured_ball_xyz=[0.,-4.,.113])]})

    def test_complete_causal_state_feedback(self):
        validate_feedback(self.report,self.config)

    def test_rejects_jointly_truncated_command_and_feedback_evidence(self):
        self.report['joint_strike']['feedback']=self.report['joint_strike']['feedback'][:1]
        self.report['kinematics'][0]=self.report['kinematics'][0][:1]
        with self.assertRaises(ValueError):validate_feedback(self.report,self.config)

    def test_rejects_missing_noncausal_and_unmeasured_feedback(self):
        changes=[lambda r:r['joint_strike'].update(feedback=[]),
            lambda r:r['joint_strike']['feedback'][0].update(contact_latched=True),
            lambda r:r['joint_strike']['feedback'][1].update(phase_s=.03),
            lambda r:r['joint_strike']['feedback'][1].update(measured_ball_xyz=[0.,-3.,.113]),
            lambda r:r['joint_strike']['feedback'][1].update(residual_rad=[float('nan')]*29),
            lambda r:r['joint_strike']['feedback'][1].update(residual_rad=[.251]*29)]
        for change in changes:
            with self.subTest(change=change):
                report=copy.deepcopy(self.report);change(report)
                with self.assertRaises(ValueError):validate_feedback(report,self.config)

if __name__=='__main__':unittest.main()
