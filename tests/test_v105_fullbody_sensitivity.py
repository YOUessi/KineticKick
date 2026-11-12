import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v105_fullbody_sensitivity as v

def fake(n=120,delta=0):
    t=np.arange(n)*.005
    q=np.zeros((n,29))
    dq=np.zeros_like(q)
    q[40:,0]=delta
    dq[40:,0]=2*delta
    return dict(t=t,q=q,dq=dq,ix={"left_arm":[0,1],"right_leg":[2,3]})

class TestV105Sensitivity(unittest.TestCase):
    def test_onset_frozen_threshold(self):
        x=fake();y=fake(delta=.02)
        d=v.sensitivity(x,y,(1e-5,.1),contact_time=.5)
        self.assertEqual(d["common_real_200hz_steps"],120)
        self.assertAlmostEqual(d["thresholds"][0]["all29_first_onset_s"],.2)
        self.assertAlmostEqual(d["thresholds"][0]["lead_before_contact_s"],.3)
        self.assertIsNone(d["thresholds"][1]["all29_first_onset_s"])
        self.assertIsNone(d["thresholds"][0]["group_first_onset_s"]["right_leg"])
    def test_reverse_symmetric(self):
        x=fake();y=fake(delta=.02)
        a=v.sensitivity(x,y,(1e-5,),.5)
        b=v.sensitivity(y,x,(1e-5,),.5)
        self.assertEqual(a["thresholds"],b["thresholds"])
    def test_self_negative_control(self):
        d=v.sensitivity(fake(),fake(),(1e-7,1e-5),.5)
        self.assertTrue(d["self_observed_null"])
        self.assertTrue(all(x["all29_first_onset_s"] is None for x in d["thresholds"]))
    def test_right_censor_shorter(self):
        a=v.sensitivity(fake(120),fake(90),(.00001,),.5)
        self.assertTrue(a["right_censored_by_shorter_trajectory"])
        self.assertEqual(a["common_real_200hz_steps"],90)
    def test_time_mismatch_fails(self):
        b=fake();b["t"][5]+=.1
        with self.assertRaises(ValueError):v.sensitivity(fake(),b,(1e-5,),.5)
    def test_nonfinite_fails(self):
        b=fake();b["q"][5,2]=np.nan
        with self.assertRaises(ValueError):v.sensitivity(fake(),b,(1e-5,),.5)
    def test_missing_joint_group_fails(self):
        with self.assertRaises(ValueError):v.onset(np.array([0.,.005]),np.zeros((2,29)),np.zeros((2,29)),[],.01)
    def test_bad_threshold_fails(self):
        with self.assertRaises(ValueError):v.sensitivity(fake(),fake(),(0,),.5)
    def test_protocol_fails_on_actuation(self):
        import json
        cfg=json.loads((Path(__file__).resolve().parents[1]/"configs/v105_frozen_protocol.json").read_text())
        self.assertEqual(len(v.prereg(cfg)),6)
        cfg["safe_policy_actuation_allowed"]=True
        with self.assertRaises(ValueError):v.prereg(cfg)
    def test_write_once_fails(self):
        import json,tempfile
        cfg=json.loads((Path(__file__).resolve().parents[1]/"configs/v105_frozen_protocol.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"audit.json";p.write_text("{}")
            with self.assertRaises(FileExistsError):v.evaluate(cfg,{},p)

if __name__=="__main__":unittest.main()
