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
