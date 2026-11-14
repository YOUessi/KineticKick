"""Guard the transition from physical right-foot impact to commanded follow-through."""

import json
from pathlib import Path

import numpy as np
from v164_strike_feedback import make_follow


def test_follow_pulse_requires_right_foot_impact_and_releases_after_landing():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "configs/v164_strike_10_05.json").read_text())
    names = [
        "left_hip_yaw",
        "right_hip_pitch",
        "right_hip_roll",
        "right_knee",
        "right_ankle_pitch",
        "left_hip_roll",
        "left_knee",
        "right_hip_yaw",
    ]
    schema = {"joint_names": [name + "_joint" for name in names]}
    apply = make_follow(schema, source["loaded_follow"])
    initial = np.linspace(-0.4, 0.3, len(names))
    # A loaded contact on a different link cannot release the right leg.
    event = {"first_contact_s": 0.5, "first_right_ball_s": None, "t": 0.6}
    np.testing.assert_array_equal(apply(event, initial.copy()), initial)
    event.update(first_right_ball_s=1.0, t=1.0)
    np.testing.assert_array_equal(apply(event, initial.copy()), initial)
    event["t"] = 1.1
    expected = initial.copy()
    expected[1:] += source["loaded_follow"]["joint_offsets_rad"][:7]
    np.testing.assert_allclose(apply(event, initial.copy()), expected)
    event["t"] = 1.68
    np.testing.assert_array_equal(apply(event, initial.copy()), initial)
