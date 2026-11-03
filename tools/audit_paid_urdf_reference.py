#!/usr/bin/env python3
"""KineticKick Gate A: compare untouched PAiD reference body traces with original/local URDF FK.

CPU-only. Does NOT report a trained physical rollout. Body names in .npz are inferred
with a multi-frame bipartite assignment; this is a diagnostic, not a verified labeling.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rotation

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "configs/paid_original/main.urdf"
LOCAL = ROOT / "data/assets/g1_29dof_lafan1/robot_description/g1/g1_29dof_rev_1_0.urdf"
MOTION = ROOT / "third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
REPORT = ROOT / "reports/paid_reference_urdf_fk_audit.json"
CSV = ROOT / "reports/paid_reference_urdf_fk_matches.csv"
# Observed ACTUAL Isaac PhysX articulation order (2026-10-22 probe, not XML DFS order).
RUNTIME_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def load_urdf(path: Path):
    root = ET.parse(path).getroot()
    tree = {}
    jj = []
    for j in root.findall("joint"):
        name = j.attrib["name"]
        kind = j.attrib["type"]
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]
        origin = j.find("origin")
        xyz = [float(x) for x in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
        rpy = [float(x) for x in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
        axis = j.find("axis")
        ax = [float(x) for x in axis.attrib["xyz"].split()] if axis is not None else [0, 0, 1]
        tree.setdefault(parent, []).append((name, kind, child, np.asarray(xyz), Rotation.from_euler("xyz", rpy), np.asarray(ax)))
        if kind in ("revolute", "continuous", "prismatic"):
            jj.append(name)
    assert len(jj) == 29, (path, len(jj))
    return tree, jj


def fk_at_frame(tree, joint_values, root_xyz, root_quat_wxyz):
    q = dict(joint_values)
    w, x, y, z = root_quat_wxyz
    poses = {"pelvis": (np.asarray(root_xyz), Rotation.from_quat([x, y, z, w]))}
    active_links = ["pelvis"]
    todo = ["pelvis"]
    while todo:
        parent = todo.pop(0)
        p, rp = poses[parent]
        for jname, kind, child, delta, r_delta, axis in tree.get(parent, ()):
            v = q.get(jname, 0.0)
            rj = Rotation.from_rotvec(axis * v) if kind in ("revolute", "continuous") else Rotation.identity()
            translation = delta + (axis * v if kind == "prismatic" else 0)
            poses[child] = (p + rp.apply(translation), rp * r_delta * rj)
            if kind in ("revolute", "continuous", "prismatic"):
                active_links.append(child)
            todo.append(child)
    assert len(active_links) == 30, len(active_links)
    return [(b, poses[b][0], poses[b][1]) for b in active_links]


def run():
    src = np.load(MOTION)
    frames = [0, 20, 60, 100, 140, 190, min(src["joint_pos"].shape[0]-1, 250)]
    arr = {k: src[k][frames] for k in ["joint_pos", "body_pos_w", "body_quat_w"]}
    assert arr["joint_pos"].shape[1] == 29 and arr["body_pos_w"].shape[1] == 30
    qwxyz = arr["body_quat_w"]
    src_rot = Rotation.from_quat(qwxyz[:, :, [1, 2, 3, 0]].reshape(-1, 4))
    src_rot_matrix = src_rot.as_matrix().reshape(len(frames), 30, 3, 3)
    results = {}
    csv_rows = []
    for label, path in (("PAiD_original", ORIGINAL), ("LAFAN_local", LOCAL)):
        tree, urdf_order = load_urdf(path)
        for name, order in (("isaac_runtime", RUNTIME_JOINTS), ("urdf_xml", urdf_order)):
            assert set(order) == set(urdf_order)
            output = []
            for k in range(len(frames)):
                x = arr["joint_pos"][k]
                mapping = dict(zip(order, x))
                pose = fk_at_frame(tree, mapping, arr["body_pos_w"][k, 0], arr["body_quat_w"][k, 0])
                output.append(pose)
            link_names = [z[0] for z in output[0]]
            xyz = np.stack([np.stack([z[1] for z in frame]) for frame in output])
            rot_matrix = np.stack([np.stack([z[2].as_matrix() for z in frame]) for frame in output])
            # distance across frames, with orientation as an additional discriminator
            dpos = np.linalg.norm(xyz[:, :, None, :] - arr["body_pos_w"][:, None, :, :], axis=-1)
            theta = np.arccos(np.clip((np.einsum("fiab,fjab->fij", rot_matrix, src_rot_matrix)-1)/2, -1, 1))
            cost = np.mean(dpos + 0.035 * theta, axis=0)
            rows, cols = linear_sum_assignment(cost)
            assert len(rows) == 30
            assigned_error = dpos[:, rows, cols]
            summary = {
                "model": label,
                "joint_order": name,
                "frames": frames,
                "assigned_body_count": len(rows),
                "matched_mean_position_error_m": round(float(assigned_error.mean()), 6),
                "matched_max_position_error_m": round(float(assigned_error.max()), 6),
                "matched_p90_position_error_m": round(float(np.percentile(assigned_error, 90)), 6),
                "mean_body_quaternion_error_deg": round(float(np.rad2deg(theta[:, rows, cols].mean())), 5),
                "matches": [{"body": link_names[i], "npz_body_index": int(j),
                             "position_mean_error_m": round(float(dpos[:, i, j].mean()), 6),
                             "position_max_error_m": round(float(dpos[:, i, j].max()), 6)}
                            for i, j in zip(rows, cols)],
                "caveat": "Assignment-based identity inference only; verify against original Isaac runtime.",
            }
            results[f"{label}__{name}"] = summary
            print(f"{label:15} {name:14} mean={summary['matched_mean_position_error_m']:.5f} "
                  f"p90={summary['matched_p90_position_error_m']:.5f} "
                  f"max={summary['matched_max_position_error_m']:.5f} "
                  f"orientation_deg={summary['mean_body_quaternion_error_deg']:.3f}", flush=True)
            for item in summary["matches"]:
                csv_rows.append({"model": label, "joint_order": name, **item})
    results["source"] = {"motion": str(MOTION), "fps": float(src["fps"][0]), "frames_checked": frames,
                         "notes": "FK consistency does not validate contact physics; model geometry is checked separately."}
    REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    best = min((v for v in results.values() if isinstance(v, dict) and "model" in v),
               key=lambda d: d["matched_mean_position_error_m"])
    print("BEST_MODEL_JOINT_ORDER", best["model"], best["joint_order"])
    print("WORST_BODIES", sorted(best["matches"], key=lambda x: x["position_mean_error_m"], reverse=True)[:7])
    print("REPORT", REPORT)


if __name__ == "__main__":
    run()
