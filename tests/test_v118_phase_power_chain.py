from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v118_phase_power_chain_audit import phase_chain,_first_sustained_effort

def a():
    t=np.arange(.005,2.505,.005)
    n=len(t)
    vel=np.zeros((n,29));tau=np.ones((n,29));support=np.ones((n,2))*100.
    # Physically measured surrogate for unit tests only; NEVER a real trial.
    mask=(t>1.20)&(t<1.30)
    vel[mask,0]=2.
    return dict(t=t,vel=vel,tau=tau,support=support,lim=np.ones(29),
                impact=1.5,names=[f"joint_{i}" for i in range(29)],
                groups=dict(waist=[0],left_leg=list(range(1,8)),
                  right_leg=list(range(8,15)),left_arm=list(range(15,22)),
                  right_arm=list(range(22,29))))
def test_actual_power_onset_and_negative_control():
    z=phase_chain(a())
    assert z["first_sustained_precontact_positive_motor_power_s"]["waist"]==pytest.approx(1.205)
    assert z["chronology_onset_differs_from_reversal"]
    assert z["positive_motor_work_proxy_not_biological_power"]
    assert z["post1250_censored"]
def test_censored_1250_never_claimed_complete():
    z=phase_chain(a())
    assert z["post1250_censored"]
    assert z["phase_windows"]["post_late"]["partial"]
def test_shape_and_200hz_validation():
    x=a();x["vel"]=np.zeros((len(x["t"]),28))
    with pytest.raises(ValueError,match="MISSING_OR_INVALID"):
        phase_chain(x)
    x=a();x["t"][2]=x["t"][1]
    with pytest.raises(ValueError,match="NON_200HZ"):
        phase_chain(x)
def test_power_below_threshold_yields_unknown_onset_not_imputation():
    x=a();x["vel"][:]=0.
    z=phase_chain(x)
    assert all(v is None for v in z["first_sustained_precontact_positive_motor_power_s"].values())
