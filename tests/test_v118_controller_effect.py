import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v118_controller_earliest_effect import onset

def test_first_actual_measured_effect_and_no_effect():
    a=dict(t=np.arange(.005,.505,.005),vel=np.zeros((100,29)))
    b=dict(t=a["t"].copy(),vel=a["vel"].copy())
    assert onset(a,b)["measured_first_29dof_qdot_diff_s"] is None
    b["vel"][20,8]=.1
    assert onset(a,b)["measured_first_29dof_qdot_diff_s"]==pytest.approx(.105)
def test_different_time_grid_rejected():
    a=dict(t=np.arange(.005,.055,.005),vel=np.zeros((10,29)))
    b=dict(t=a["t"]+.001,vel=a["vel"])
    with pytest.raises(ValueError,match="TIME_GRID"):onset(a,b)
