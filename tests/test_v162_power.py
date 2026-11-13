import numpy as np
import pytest
from v162_strike_power import coefficients


def test_policy_uses_both_continuous_inputs_even_between_anchors():
    speeds=[3.6,4.1,4.6];aims=[-.2,.2]
    config={'speeds_mps':speeds,'aims_m':aims,
            'coefficient_grid':[[np.full(12,3*s+7*x) for x in aims] for s in speeds]}
    np.testing.assert_allclose(coefficients(config,3.85,.04),3*3.85+7*.04)
    np.testing.assert_allclose(coefficients(config,4.6,.2),3*4.6+7*.2)
    with pytest.raises(ValueError):coefficients(config,4.7,0)


def test_local_inverse_uses_both_request_errors_and_bounds_adjustments():
    config={'map_type':'local_inverse','domain':{'speed':[3.,4.],'aim':[-.2,.2]},
            'base':np.zeros(15).tolist(),'observation_mean':[3.5,0.],
            'scaled_inverse':[[2.,1.]]*15,'absolute_delta_limit':[.3]*15}
    np.testing.assert_allclose(coefficients(config,3.55,-.02),.08)
    np.testing.assert_allclose(coefficients(config,3.90,.10),.30)
    with pytest.raises(ValueError):coefficients(config,3.8,-.4)
