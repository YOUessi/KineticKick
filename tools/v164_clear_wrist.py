"""Extend the tracking runner with measured bilateral wrist-to-hip clearance feedback."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import v164_support_track as tracking
from scipy.spatial.transform import Rotation
from v164_clear_legs import LegClearance, closest_capsule_pair


def link_capsule(robot, link):
    shape = robot.find(f"link[@name='{link}']/collision")
    origin = shape.find("origin")
    cylinder = shape.find("geometry/cylinder")
    center = np.fromstring(origin.get("xyz"), sep=" ")
    rotation = Rotation.from_euler(
        "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    )
    half = 0.5 * float(cylinder.get("length")) * rotation.as_matrix()[:, 2]
    return np.stack((center - half, center + half)), float(cylinder.get("radius"))


class WristClearance:
    def __init__(self, schema, urdf, cfg):
        self.cfg = cfg
        self.sides = []
        self.previous_gap = [None, None]
        self.ticks = [0, 0]
        self.max_shift = [0.0, 0.0]
        robot = ET.parse(urdf)
        for side in ("left", "right"):
            wrist = side + "_wrist_yaw_link"
            hip = side + "_hip_roll_link"
            roll = side + "_shoulder_roll_joint"
            shoulder = side + "_shoulder_roll_link"
            axis = np.fromstring(
                robot.find(f"joint[@name='{roll}']/axis").get("xyz"), sep=" "
            )
            self.sides.append(
                (
                    schema["body_names"].index(wrist),
                    schema["body_names"].index(hip),
                    schema["body_names"].index(shoulder),
                    schema["joint_names"].index(roll),
                    axis,
                    link_capsule(robot, wrist),
                    link_capsule(robot, hip),
                )
            )

    def __call__(self, ctx, target):
        positions = ctx["body_pos"]
        rotations = Rotation.from_quat(ctx["body_quat"][:, [1, 2, 3, 0]]).as_matrix()
        for side, (
            wrist,
            hip,
            shoulder,
            roll,
            axis,
            (wrist_local, wrist_radius),
            (hip_local, hip_radius),
        ) in enumerate(self.sides):
            wrist_line = positions[wrist] + wrist_local @ rotations[wrist].T
            hip_line = positions[hip] + hip_local @ rotations[hip].T
            gap, on_wrist, on_hip = closest_capsule_pair(
                wrist_line, wrist_radius, hip_line[None], np.array([hip_radius])
            )
            previous = self.previous_gap[side]
            self.previous_gap[side] = gap
            if previous is None or gap > self.cfg["activation_distance_m"]:
                continue
            closing_speed = min((gap - previous) / ctx["dt"], 0.0)
            deficit = self.cfg["margin_m"] - gap - self.cfg["preview_s"] * closing_speed
            if deficit <= 0.0:
                continue
            normal = on_wrist - on_hip
            normal /= max(np.linalg.norm(normal), 1e-8)
            derivative = (
                np.cross(rotations[shoulder] @ axis, on_wrist - positions[shoulder])
                @ normal
            )
            delta = self.cfg["gain"] * deficit * derivative / (derivative**2 + 0.01)
            change = np.clip(
                delta, -self.cfg["max_shoulder_rad"], self.cfg["max_shoulder_rad"]
            )
            target[roll] += change
            self.ticks[side] += 1
            self.max_shift[side] = max(self.max_shift[side], abs(float(change)))
        return target


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_known_args()[0]
    settings = json.loads(args.config.read_text())["wrist_clearance"]
    wrist = None

    class BothClearances(LegClearance):
        def __init__(self, schema, cfg, urdf):
            nonlocal wrist
            super().__init__(schema, cfg, urdf)
            wrist = WristClearance(schema, urdf, settings)

        def __call__(self, ctx, target):
            return wrist(ctx, super().__call__(ctx, target))

    tracking.LegClearance = BothClearances
    tracking.main()
    tracking.b.save_json(
        args.output / "wrist_feedback.json",
        {
            "code_sha256": tracking.b.digest(Path(__file__)),
            "settings": settings,
            "corrected_ticks_left_right": wrist.ticks,
            "max_shoulder_rad_left_right": wrist.max_shift,
        },
    )


if __name__ == "__main__":
    main()
