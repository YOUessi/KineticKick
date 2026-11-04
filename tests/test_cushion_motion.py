"""Trajectory properties needed for interception without command discontinuities."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from cushion_motion import descending_time,hermite


class CushionMotionTests(unittest.TestCase):
    def test_prediction_selects_descending_crossing(self):
        for z,vz,h in [(1.05,0.,.3),(.5,3.,.3),(.9,-1.,.2)]:
            t=descending_time(z,vz,h)
            self.assertGreaterEqual(t,0)
            self.assertAlmostEqual(z+vz*t-4.905*t*t,h)
            self.assertLess(vz-9.81*t,0)

    def test_descending_ball_below_plane_has_no_future_intercept(self):
        self.assertIsNone(descending_time(.1,-2.,.3))

    def test_trajectory_velocity_matches_position_derivative_and_endpoints(self):
        p0=np.array([0.,-.38,.15]);p1=np.array([0.,-.40,.154]);v0=np.zeros(3);v1=np.array([0.,.135,-.69]);duration=.24
        for t in (.03,.12,.21):
            p,v=hermite(p0,p1,v0,v1,duration,t);eps=1e-6
            before=hermite(p0,p1,v0,v1,duration,t-eps)[0]
            after=hermite(p0,p1,v0,v1,duration,t+eps)[0]
            np.testing.assert_allclose((after-before)/(2*eps),v,atol=1e-7)
        for t,position,velocity in [(0.,p0,v0),(duration,p1,v1)]:
            p,v=hermite(p0,p1,v0,v1,duration,t)
            np.testing.assert_allclose(p,position,atol=1e-12)
            np.testing.assert_allclose(v,velocity,atol=1e-12)


if __name__=='__main__':unittest.main()
