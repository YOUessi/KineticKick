"""Protect the free leg when its original G1 shin/sole capsules approach the support leg."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

LEFT_SHIN, LEFT_SOLE, RIGHT_SHIN, RIGHT_SOLE = range(4)
CROSS_LEG_PAIRS = (
    (LEFT_SHIN, RIGHT_SOLE),
    (RIGHT_SHIN, LEFT_SOLE),
    (LEFT_SHIN, RIGHT_SHIN),
)


def closest_capsule_pair(shin_axis, shin_radius, other_axes, other_radii):
    """Return signed surface gap and closest points on both capsule center lines."""
    start, end = shin_axis
    shin_direction = end - start
    other_starts = other_axes[:, 0]
    other_directions = other_axes[:, 1] - other_starts
    separation = start - other_starts
    shin_sq = shin_direction @ shin_direction
    cross = other_directions @ shin_direction
    other_sq = np.sum(other_directions * other_directions, axis=1)
    shin_projection = separation @ shin_direction
    other_projection = np.sum(other_directions * separation, axis=1)
    determinant = shin_sq * other_sq - cross * cross

    # Solve for the closest unconstrained lines, then clamp to finite segments.
    along_shin = np.clip(
        (cross * other_projection - other_sq * shin_projection)
        / np.maximum(determinant, 1e-15),
        0,
        1,
    )
    along_other = (cross * along_shin + other_projection) / other_sq
    along_shin = np.where(
        along_other < 0,
        np.clip(-shin_projection / shin_sq, 0, 1),
        np.where(
            along_other > 1,
            np.clip((cross - shin_projection) / shin_sq, 0, 1),
            along_shin,
        ),
    )
    along_other = np.clip(along_other, 0, 1)
    displacement = (
        separation
        + along_shin[:, None] * shin_direction
        - along_other[:, None] * other_directions
    )
    gaps = np.linalg.norm(displacement, axis=1) - shin_radius - other_radii
    nearest = gaps.argmin()
    shin_point = start + along_shin[nearest] * shin_direction
    other_point = (
        other_starts[nearest] + along_other[nearest] * other_directions[nearest]
    )
    return float(gaps[nearest]), shin_point, other_point


class LegClearance:
    def __init__(self, schema, cfg, urdf):
        self.cfg = cfg
        self.feet = []
        self.shin = []
        self.segments = []
        self.radii = []
        sides = ("left", "right")
        xml = ET.parse(urdf)
        for side in sides:
            self.feet.append(schema["body_names"].index(side + "_ankle_roll_link"))
            self.shin.append(schema["body_names"].index(side + "_knee_link"))
            for link in ("knee", "ankle_roll"):
                endpoints = []
                radii = []
                for collision in xml.findall(
                    f"link[@name='{side}_{link}_link']/collision"
                ):
                    cylinder = collision.find("geometry/cylinder")
                    origin = collision.find("origin")
                    center = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                    rotation = Rotation.from_euler(
                        "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                    )
                    half_axis = (
                        0.5 * float(cylinder.get("length")) * rotation.as_matrix()[:, 2]
                    )
                    endpoints.append([center - half_axis, center + half_axis])
                    radii.append(float(cylinder.get("radius")))
                self.segments.append(np.array(endpoints))
                self.radii.append(np.array(radii))
        self.axes = []
        self.channels = []
        self.link_ids = []
        for side in sides:
            names = [
                side + "_" + joint
                for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
            ]
            self.channels.append(
                [schema["joint_names"].index(name + "_joint") for name in names]
            )
            self.link_ids.append(
                [schema["body_names"].index(name + "_link") for name in names]
            )
            self.axes.append(
                np.array(
                    [
                        np.fromstring(
                            xml.find(f"joint[@name='{name}_joint']/axis").get("xyz"),
                            sep=" ",
                        )
                        for name in names
                    ]
                )
            )
        self.previous_gaps = None
        self.corrections = 0

    def __call__(self, ctx, target):
        positions = ctx["body_pos"]
        rotations = Rotation.from_quat(ctx["body_quat"][:, [1, 2, 3, 0]]).as_matrix()
        links = (self.shin[0], self.feet[0], self.shin[1], self.feet[1])
        segments = [
            positions[link] + ends @ rotations[link].T
            for link, ends in zip(links, self.segments)
        ]
        pelvis_lateral_axis = rotations[0, :, 1]
        foot_spacing = (
            positions[self.feet[0]] - positions[self.feet[1]]
        ) @ pelvis_lateral_axis

        gaps = []
        witnesses = []
        for shin, other in CROSS_LEG_PAIRS:
            gap, on_shin, on_other = closest_capsule_pair(
                segments[shin][0],
                self.radii[shin][0],
                segments[other],
                self.radii[other],
            )
            # Keep witness ordering left, right independently of the geometry pair.
            witnesses.append(
                (on_shin, on_other) if shin == LEFT_SHIN else (on_other, on_shin)
            )
            gaps.append(gap)
        gaps = np.array(gaps)
        gap_rates = (
            np.zeros(3)
            if self.previous_gaps is None
            else (gaps - self.previous_gaps) / ctx["dt"]
        )
        self.previous_gaps = gaps
        loads = np.asarray(ctx["foot_load"])
        if loads.min() >= self.cfg.get("swing_load", 85.0):
            return target
        free_leg = loads.argmin()
        deficit = (
            self.cfg["margin"] - gaps - self.cfg["horizon"] * np.minimum(gap_rates, 0.0)
        )
        closest = deficit.argmax()
        required_separation = max(0.0, deficit[closest])
        if foot_spacing <= self.cfg["width"] and required_separation > 0.0:
            points = witnesses[closest]
            normal = points[free_leg] - points[1 - free_leg]
            normal /= max(1.0e-8, np.linalg.norm(normal))
            joint_links = self.link_ids[free_leg]
            axes = np.einsum("kij,kj->ki", rotations[joint_links], self.axes[free_leg])
            gap_jacobian = (
                np.cross(axes, points[free_leg] - positions[joint_links]) @ normal
            )
            offset = (
                gap_jacobian
                * required_separation
                / (gap_jacobian @ gap_jacobian + 0.015)
            )
            self.corrections += 1
            target[self.channels[free_leg]] += np.clip(
                self.cfg["gain"] * offset,
                -self.cfg["max_target"],
                self.cfg["max_target"],
            )
        return target
