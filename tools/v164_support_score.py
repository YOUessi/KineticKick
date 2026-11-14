"""Audit whole-clip tracking and terminal support against the original 50 Hz source clock."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from v161_evaluate import (
    DEFAULT_URDF,
    digest,
    quat_rotation,
    reconstruct_com,
    report,
    tail_duration,
)
from v163_diagnose_fidelity import series_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "reference", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--clearance", type=Path)
    a = p.parse_args()
    physical = report(a.run)
    summary = json.loads((a.run / "tracking_result.json").read_text())
    trial = json.loads((a.run / "trial.json").read_text())
    original_sha = digest(a.reference)
    if a.clearance:
        transform = json.loads(a.clearance.read_text())
        if (
            transform["input_sha256"] != original_sha
            or transform["output_sha256"] != trial["reference_sha256"]
            or transform["urdf_sha256"] != physical["urdf_sha256"]
        ):
            raise ValueError("Unverified reference correction")
    elif trial["reference_sha256"] != original_sha:
        raise ValueError("Reference mismatch")
    with (
        np.load(a.reference, allow_pickle=False) as r,
        np.load(a.run / "rollout.npz", allow_pickle=False) as trace,
    ):
        if (
            r["body_names"].tolist() != summary["source_body_names"]
            or r["joint_names"].tolist() != summary["source_joint_names"]
        ):
            raise ValueError("Physical and reference name mismatch")
        if r["fps"].tolist() != [50] or not np.allclose(
            r["time_s"], np.arange(len(r["time_s"])) * 0.02
        ):
            raise ValueError("Bad source timing")
        count = len(r["time_s"]) - 1
        ix = np.arange(3, 4 * count, 4)
        if not np.allclose(trace["time_s"][ix], r["time_s"][1:]):
            raise ValueError("Source clock mismatch")
        pos = trace["body_position_w_m"]
        root = trace["root_pose_wxyz"]
        quat = trace["body_quat_wxyz"]
        t = trace["time_s"]
        q = trace["joint_q_rad"]
        fidelity = series_metrics(
            pos[ix],
            r["body_pos_w"][1:],
            q[ix],
            r["joint_pos"][1:],
            quat_rotation(root[ix, 3:]),
            quat_rotation(r["body_quat_w"][1:, 0]),
            summary["source_body_names"],
        )
        xml = ET.parse(DEFAULT_URDF)
        com, _ = reconstruct_com(xml, summary["source_body_names"], pos, quat)
        vel = np.diff(com, axis=0, prepend=np.full_like(com[:1], np.nan)) / 0.005
        loads = trace["foot_ground_normal_load_N"]
        stable = (
            (t > r["time_s"][-1])
            & (root[:, 2] > 0.55)
            & (
                quat_rotation(root[:, 3:]).as_matrix()[:, 2, 2]
                >= np.cos(np.deg2rad(25.0))
            )
            & (np.linalg.norm(trace["root_velocity_w"][:, :2], axis=1) < 0.30)
            & np.all(loads > 30.0, axis=1)
            & (loads.sum(axis=1) >= 80.0)
            & (trace["nonfoot_net_contact_peak_N"] < 20.0)
            & (np.linalg.norm(vel[:, :2], axis=1) <= 0.30)
            & (np.abs(vel[:, 2]) <= 0.20)
            & (~trace["robot_ball_contact"])
        )
        tail = tail_duration(stable, 0.005)
        edges = [
            xml.find(f'joint[@name="{n}"]/limit') for n in summary["source_joint_names"]
        ]
        low = [float(e.get("lower")) for e in edges]
        high = [float(e.get("upper")) for e in edges]
        err = np.maximum(np.array(low) - q, q - np.array(high))
        jj = np.unravel_index(err.argmax(), err.shape)
        peak = {
            "joint": summary["source_joint_names"][jj[1]],
            "time_s": float(t[jj[0]]),
            "excess_rad": max(0.0, float(err[jj])),
        }
    gates = {
        key: physical["checks"][key]
        for key in [
            "complete_14s",
            "no_mid_episode_state_writes",
            "no_nonfoot",
            "original_q_bounds",
            "original_qdot_bounds",
            "no_fall",
        ]
    }
    gates["terminal_support"] = tail >= 2.5
    out = {
        "source_sha256": original_sha,
        "rollout_sha256": physical["input_sha256"]["rollout.npz"],
        "contact_sha256": physical["input_sha256"]["native_contacts.jsonl.gz"],
        "source_intervals": count,
        "original_source_fidelity": fidelity,
        "strict_standing_tail_s": tail,
        "clock_complete": count / 50,
        "native_pair_events": physical["native_pair_events"],
        "first_fall_s": physical["first_fall_s"],
        "joint_position_excess": peak,
        "q_violation_cells": physical["q_position_violation_cells"],
        "dq_violation_cells": physical["qdot_violation_cells"],
        "gates": gates,
        "passed": all(gates.values()),
    }
    with a.output.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                k: out[k]
                for k in ["strict_standing_tail_s", "gates", "joint_position_excess"]
            }
        )
    )


if __name__ == "__main__":
    main()
