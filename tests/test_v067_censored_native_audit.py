"""Any missing native CUDA proof or original nonfoot must FAIL CLOSED."""
import copy
import pytest
from tools.audit_v067_censored_native_output import qualify_partial

def test_no_raw_is_not_g1():
    with pytest.raises(ValueError):
        qualify_partial({},'V058_NATIVE_RETURN cuda:0',{'seed':6719})

def test_cannot_infer_nonfoot_from_goal_or_early_stop_label():
    mock=dict(root_fixed=False,seed=6719)
    with pytest.raises(ValueError):
        qualify_partial(mock,'Environment device    : cuda:0',{'seed':6719})
