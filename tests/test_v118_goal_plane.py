from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v118_goal_plane_evidence import interpolate_crossing

def test_interpolation_brackets_real_goal_plane_without_future():
    trace=[{"time_s":.005,"ball_xyz":[1.,-4.,.11]},
           {"time_s":.010,"ball_xyz":[1.1,-7.,.11]}]
    m=interpolate_crossing(trace,[1.5,-5.5])
    assert m["crossing_observed"]
    assert m["ball_center_at_original_goal_plane_xyz_m"][0]==pytest.approx(1.05)
    assert m["time_s"]==pytest.approx(.0075)
    assert not m["extrapolated"]
def test_missing_crossing_not_fabricated():
    trace=[{"time_s":.005,"ball_xyz":[1.,-4.,.11]},
           {"time_s":.010,"ball_xyz":[1.1,-4.1,.11]}]
    assert not interpolate_crossing(trace,[1.5,-5.5])["crossing_observed"]
def test_non200hz_rejected():
    trace=[{"time_s":.005,"ball_xyz":[1.,-4.,.11]},
           {"time_s":.015,"ball_xyz":[1.1,-7.,.11]}]
    with pytest.raises(ValueError,match="200HZ"):
        interpolate_crossing(trace,[1.5,-5.5])
