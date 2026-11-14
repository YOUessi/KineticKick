"""Regression for the two-step rebound of a joint initialized at its original limit."""

import numpy as np
from v164_initial_state import project_start


def test_init_reserves_outward_travel_and_preserves_the_unaffected_joint():
    source_q = np.array([0.26, -0.26, 0.20])
    source_dq = np.array([1.5, -1.0, 0.0])
    limits = np.tile([-0.26, 0.26, 8.0, 30.0], (3, 1))
    settings = {"rebound_reserve_rad": 0.04, "reaction_steps": 2}
    start, velocity, changes = project_start(
        source_q,
        source_dq,
        limits,
        ["right_roll", "left_roll", "knee"],
        settings,
        0.005,
    )
    np.testing.assert_allclose(start, [0.205, -0.210, 0.20])
    np.testing.assert_array_equal(velocity, source_dq)
    np.testing.assert_array_equal(source_q, [0.26, -0.26, 0.20])
    assert [entry["joint"] for entry in changes] == ["right_roll", "left_roll"]
