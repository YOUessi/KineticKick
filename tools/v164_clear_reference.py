"""Separate upper-arm, elbow, wrist and trunk/hip source capsules through shoulder roll."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation
from v162_human_armclear import witnesses
from v162_human_support import source

ASSETS = Path("/home/you/projects/kinetickick")
URDF = "configs/paid_original/paid_main_resolved.urdf"
SCHEMA = "reports/paid_physics_parity_clean_paid.json"


def capsules(xml, name):
    result = []
    for shape in xml.findall(f"link[@name='{name}']/collision"):
        cyl = shape.find("geometry/cylinder")
        ori = shape.find("origin")
        center = np.fromstring(ori.get("xyz", "0 0 0"), sep=" ")
        rot = Rotation.from_euler(
            "xyz", np.fromstring(ori.get("rpy", "0 0 0"), sep=" ")
        )
        end = rot.as_matrix()[:, 2] * (0.5 * float(cyl.get("length")))
        result.append(
            (np.stack([center - end, center + end]), float(cyl.get("radius")))
        )
    if not result:
        raise ValueError("No capsule collision shapes on " + name)
    return result


class ArmClearance:
    def __init__(self, robot, xml, side):
        self.robot = robot
        self.side = side
        self.sign = 1.0 if side == "left" else -1.0
        self.joint = robot.model.joints[
            robot.model.getJointId(side + "_shoulder_roll_joint")
        ].idx_q
        self.pairs = [
            (side + "_wrist_yaw_link", side + "_hip_roll_link"),
            (side + "_elbow_link", "torso_link"),
            (side + "_shoulder_yaw_link", "torso_link"),
        ]
        self.shapes = {n: capsules(xml, n) for pair in self.pairs for n in pair}
        self.margin = np.array([0.012, 0.004, 0.004])

    def gaps(self, q):
        p, r = self.robot.poses(q)
        segments = {
            n: [
                (p[self.robot.index[n]] + e @ r[self.robot.index[n]].T, radius)
                for e, radius in group
            ]
            for n, group in self.shapes.items()
        }
        return np.array(
            [
                min(
                    np.linalg.norm(np.subtract(*witnesses(a, b))) - ar - br
                    for a, ar in segments[one]
                    for b, br in segments[two]
                )
                for one, two in self.pairs
            ]
        )

    def project(self, q):
        initial = np.array([self.gaps(row) for row in q])
        minimum = np.zeros(len(q))
        corrections = q.copy()
        for i, row in enumerate(q):
            if np.all(initial[i] >= self.margin):
                continue
            k = self.joint
            maximum = max(
                self.sign * (self.robot.low[k] - row[k]),
                self.sign * (self.robot.high[k] - row[k]),
            )
            low = 0.0
            for high in np.arange(0.025, maximum + 0.025, 0.025):
                high = min(high, maximum)
                corrections[i, k] = row[k] + self.sign * high
                if np.all(self.gaps(corrections[i]) >= self.margin):
                    for _ in range(10):
                        mid = (low + high) / 2
                        corrections[i, k] = row[k] + self.sign * mid
                        if np.all(self.gaps(corrections[i]) >= self.margin):
                            high = mid
                        else:
                            low = mid
                    minimum[i] = high
                    break
                low = high
            else:
                raise ValueError(
                    f"No feasible outward correction for {self.side} at frame {i}"
                )
        # Build a smooth envelope that remains above the per-frame required angle.
        spread = maximum_filter1d(minimum, size=7, mode="nearest")
        smooth = gaussian_filter1d(spread, sigma=1.0, radius=3, mode="nearest")
        corrections[:, self.joint] = q[:, self.joint] + self.sign * np.maximum(
            smooth, minimum
        )
        final = np.array([self.gaps(row) for row in corrections])
        if (final < self.margin - 1.0e-5).any():
            raise ValueError(
                "An outward angle failed the post-projection collision check"
            )
        info = {
            "pairs": self.pairs,
            "initial_first_m": initial[0].tolist(),
            "initial_minimum_m": initial.min(axis=0).tolist(),
            "frames_initial_overlap": (initial < 0).sum(axis=0).tolist(),
            "final_minimum_m": final.min(axis=0).tolist(),
            "maximum_joint_adjustment_rad": float(np.max(abs(corrections - q))),
        }
        return corrections, info


def project(input_path, output, assets):
    xml = ET.parse(assets / URDF)
    schema = json.loads((assets / SCHEMA).read_text())
    robot = source.Robot(assets / URDF, schema)
    with np.load(input_path, allow_pickle=False) as data:
        ref = {k: data[k] for k in data.files}
    if (
        ref["joint_names"].tolist() != schema["joint_names"]
        or ref["body_names"].tolist() != schema["body_names"]
        or ref["fps"].tolist() != [50]
        or len(ref["joint_pos"]) < 3
    ):
        raise ValueError("Expected canonical 50 Hz retarget with 29 named axes")
    q = np.asarray(ref["joint_pos"][:, np.argsort(robot.runtime_indices)], dtype=float)
    info = {}
    for side in ("left", "right"):
        q, info[side] = ArmClearance(robot, xml, side).project(q)
    posrot = [robot.poses(row) for row in q]
    rotation = Rotation.from_quat(ref["root_quat_wxyz"][:, [1, 2, 3, 0]]).as_matrix()
    positions = ref["root_pos_w"][:, None] + np.einsum(
        "tij,tkj->tki", rotation, np.stack([item[0] for item in posrot])
    )
    rotations = rotation[:, None] @ np.stack([item[1] for item in posrot])
    foot_ids = [robot.index[side + "_ankle_roll_link"] for side in ("left", "right")]
    feet_error = float(
        np.max(
            np.linalg.norm(
                positions[:, foot_ids] - ref["body_pos_w"][:, foot_ids], axis=-1
            )
        )
    )
    velocity_ratio = float(np.max(np.abs(np.diff(q, axis=0)) * 50.0 / robot.vlim))
    if (
        feet_error > 1.0e-6
        or velocity_ratio > 1.000001
        or np.any(q < robot.low - 1.0e-6)
        or np.any(q > robot.high + 1.0e-6)
    ):
        raise ValueError("Reference feet or original joint limits failed verification")
    updated = q[:, robot.runtime_indices]
    ref.update(
        joint_pos=updated.astype("f4"),
        joint_vel=np.gradient(updated, 0.02, axis=0, edge_order=2).astype("f4"),
        body_pos_w=positions.astype("f4"),
        body_quat_w=source.angle_quaternions(rotations).astype("f4"),
        body_lin_vel_w=np.gradient(positions, 0.02, axis=0, edge_order=2).astype("f4"),
        body_ang_vel_w=source.world_angular_velocity(rotations, 0.02).astype("f4"),
    )
    # Preserve stored source fields and distinguish generated motion descriptors from source ones.
    measured = ref["human_segment_direction_root"].astype(float)
    measured /= np.linalg.norm(measured, axis=-1, keepdims=True)
    actual = np.array([robot.features(row)[0] for row in q])
    ref["ik_feature_error_deg"] = np.rad2deg(
        np.arccos(np.clip((actual * measured).sum(-1), -1, 1))
    ).astype("f4")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        np.savez_compressed(f, **ref)
    info.update(
        input_sha256=source.sha(input_path),
        output_sha256=source.sha(output),
        urdf_sha256=source.sha(assets / URDF),
        feet_max_difference_m=feet_error,
        joint_interval_velocity_max_ratio=velocity_ratio,
        frames=len(q),
    )
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    args = parser.parse_args()
    print(
        json.dumps(
            project(args.input, args.output, args.assets), indent=2, allow_nan=False
        )
    )
