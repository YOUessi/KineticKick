import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v104_paired_dynamics as v

def fake(n=110):
    t=np.arange(n)*.005
    q=np.zeros((n,29));dq=np.zeros_like(q)
    return {"t":t,"q":q,"dq":dq,"ix":{"full":list(range(29))}}

class TestPairedPhysics(unittest.TestCase):
    def test_unchanged(self):
        x=fake();r=v.match_prefix(x,fake())
        self.assertIsNone(r["first_full29_state_divergence_s"])
        self.assertEqual(r["observed_matched_200hz_steps"],110)
        self.assertTrue(r["unchanged_within_fixed_tolerance"])
    def test_real_divergence(self):
        a=fake();b=fake();b["q"][2:,10]=.015;b["dq"][2:,10]=.25
        r=v.match_prefix(a,b);self.assertAlmostEqual(r["first_full29_state_divergence_s"],.01)
        self.assertGreater(r["group_rms_joint_velocity_delta_rad_s"]["full"],0)
    def test_censoring(self):
        a=fake(110);b=fake(90)
        r=v.match_prefix(a,b);self.assertEqual(r["observed_matched_200hz_steps"],90)
        self.assertTrue(r["earlier_termination_causes_pair_right_censoring"])
    def test_no_fabricated_time(self):
        b=fake();b["t"][5]+=.5
        with self.assertRaises(ValueError):v.match_prefix(fake(),b)
    def test_short_prefix_denied(self):
        with self.assertRaises(ValueError):v.match_prefix(fake(1),fake(1))
    def test_event_contact_uses_real_step_only(self):
        state=fake(90)
        rows=[{"time_s":float(t),"nonfoot_contact_N":0.} for t in state["t"]]
        rows[5]["nonfoot_contact_N"]=23.
        r=v.first_unsafe_nonfoot({"traces":[rows]},state)
        self.assertAlmostEqual(r["first_over_20N_s"],.025)
        self.assertFalse(r["true_native_collision_pair_certified"])
    def test_censored_nonfoot_no_absence_claim(self):
        state=fake(90)
        rows=[{"time_s":float(t),"nonfoot_contact_N":0.} for t in state["t"]]
        r=v.first_unsafe_nonfoot({"traces":[rows]},state)
        self.assertTrue(r["no_event_observed_before_termination_only"])
    def test_nan_phase_denied(self):
        with self.assertRaises(ValueError):v.phase_correlation(np.full(90,np.nan),np.ones(90))
    def test_ill_conditioned_phase_flag(self):
        z=v.phase_correlation(np.ones(90),np.ones(90))
        self.assertFalse(z["observable"])
    def test_activity_phase_not_physical_truth(self):
        n=100;x=np.sin(np.arange(n)*.2);y=x.copy()
        self.assertTrue(v.phase_correlation(x,y)["observable"])

if __name__=="__main__":unittest.main()
