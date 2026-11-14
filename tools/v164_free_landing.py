"""Contact-triggered step placement for the follow-through of a human-derived shot.

Compute a capture-point foot-placement error from the current pelvis velocity and
the reference. A correction to the stepping leg's world-space ankle position is
realized only by its named hip and knee drives. The reference source clock is retained.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def build(schema, ref, cfg, original):
    hip_names = ("hip_pitch", "hip_roll", "hip_yaw", "knee")
    joint_ids = np.array(
        [
            [schema["joint_names"].index(f"{side}_{j}_joint") for j in hip_names]
            for side in ("left", "right")
        ]
    )
    links = np.array(
        [
            [schema["body_names"].index(f"{side}_{j}_link") for j in hip_names]
            for side in ("left", "right")
        ]
    )
    ankles = [
        schema["body_names"].index(f"{side}_ankle_roll_link")
        for side in ("left", "right")
    ]
    axes = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    gains = np.asarray(cfg["gains"], dtype=float)
    delay = float(cfg.get("delay", 0.07))
    damping = float(cfg.get("ik_damping", 0.013))
    if not np.isfinite([delay, damping]).all() or delay < 0 or damping <= 0:
        raise ValueError("Expected a nonnegative delay and positive IK damping")
    if gains.shape != (6,) or not np.isfinite(gains).all():
        raise ValueError("Expected six finite swing and stance feedback gains")

    def command(ctx, unused):
        target = original(ctx, unused)
        first = ctx["first_right_ball_s"]
        if first is None or ctx["t"] <= first + delay:
            return target
        frame = ctx["frame"]
        # The source leaves one foot raised on its last frame. Release foot placement
        # gradually for the existing two-foot transition.
        ramp = min(1.0, (ctx["t"] - first - delay) / 0.10)
        fade = np.clip(
            1.0 - (ctx["t"] - (len(ref["joint_pos"]) - 1) * 0.02) / 0.45, 0, 1
        )
        if fade <= 0:
            return target
        idx = min(frame + 4, len(ref["joint_pos"]) - 1)
        pos = ctx["body_pos"]
        rv = ctx["root_velocity"]
        # Position error is in world coordinates, allowing the support/swing leg to
        # compensate actual drift even while the human turns through a large yaw.
        ep = ctx["root_pos"][:2] - ref["body_pos_w"][frame, 0, :2]
        ev = rv[:2] - ref["body_lin_vel_w"][idx, 0, :2]
        tau = np.sqrt(max(0.35, ctx["root_pos"][2]) / 9.81)
        for side in range(2):
            ankle = ankles[side]
            q = ctx["body_quat"][links[side]]
            directions = Rotation.from_quat(q[:, [1, 2, 3, 0]]).apply(axes)
            jacobian = np.cross(directions, pos[ankle] - pos[links[side]]).T
            reference_point = ref["body_pos_w"][idx, ankle] - ref["body_pos_w"][idx, 0]
            relative = reference_point - (pos[ankle] - ctx["root_pos"])
            load = np.clip((ctx["foot_load"][side] - 15.0) / 185.0, 0.0, 1.0)
            # Free foot steps toward the extrapolated deviation to brake it.
            # A loaded foot can create horizontal acceleration through the support leg.
            stride = gains[0] * tau * ev + gains[1] * ep + gains[2] * relative[:2]
            plant = gains[3] * tau * ev + gains[4] * ep
            # Reference height is used only for a swing foot.
            delta = (1.0 - load) * np.r_[stride, gains[5] * relative[2]] + load * np.r_[
                plant, 0.0
            ]
            delta = np.clip(delta, -0.17, 0.17)
            correction = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(3), delta
            )
            target[joint_ids[side]] += ramp * fade * np.clip(correction, -0.35, 0.35)
        return target

    return command
