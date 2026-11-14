"""Release the G1 swing leg after its first load-bearing right-foot/ball impact."""

import numpy as np
from v161_strike_point import smooth


def make_follow(schema, cfg):
    """Run bounded hip/knee/ankle follow-through once the first real foot contact is observed."""
    names = [
        "right_hip_pitch",
        "right_hip_roll",
        "right_knee",
        "right_ankle_pitch",
        "left_hip_roll",
        "left_knee",
    ]
    if cfg.get("include_hip_yaw"):
        names.append("right_hip_yaw")
    indices = [schema["joint_names"].index(n + "_joint") for n in names]
    coeff = np.array(cfg["joint_offsets_rad"], dtype=float).reshape(2, len(indices))
    onset = cfg.get("onset_ramp_s", 0.075)
    if not 0.0 < onset <= 0.20 or not np.all(np.isfinite(coeff)):
        raise ValueError("Invalid follow-through pulse")

    def apply(ctx, target):
        first = ctx["first_right_ball_s"]
        if first is None:
            return target
        elapsed = ctx["t"] - first
        if elapsed < 0 or elapsed >= 0.67:
            return target
        # First project the swing ankle forward while the ball leaves, then blend
        # to the human leg trajectory as the support of the landing foot rises.
        advance = smooth(elapsed / onset) * smooth((0.33 - elapsed) / 0.13)
        land = smooth((elapsed - 0.16) / 0.09) * smooth((0.67 - elapsed) / 0.20)
        target[indices] += np.array([advance, land]) @ coeff
        return target

    return apply
