#!/usr/bin/env python3
"""Audit humanoid reference motions and compare aligned robot rollouts.

Requires numpy and scipy. No Isaac Sim runtime or GPU is needed.
Contact events are NOT inferred from human/robot pose; use actual labels later.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter, correlate


FIELDS = ("fps", "joint_pos", "joint_vel", "body_pos_w",
          "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def load_clip(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        missing = set(FIELDS).difference(f.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        d = {k: np.asarray(f[k]).copy() for k in FIELDS}
        if "kick_leg" in f:
            d["kick_leg"] = str(f["kick_leg"].item())
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    n, joints = d["joint_pos"].shape
    assert 10 <= fps <= 1000, f"bad fps {fps}"
    assert n >= 9 and joints > 0, "bad motion length/joint count"
    assert d["joint_vel"].shape == (n, joints)
    assert d["body_pos_w"].ndim == 3 and d["body_pos_w"].shape[0] == n
    assert d["body_pos_w"].shape[-1] == 3
    bodies = d["body_pos_w"].shape[1]
    for key, dims in (
        ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)
    ):
        assert d[key].shape == (n, bodies, dims), f"bad {key} shape"
    for key in FIELDS[1:]:
        if not np.isfinite(d[key]).all():
            raise ValueError(f"{path}: {key} contains NaN/Inf")
    d["fps"] = fps
    return d


def velocity_from_positions(q, fps):
    # Differentiate a mild, fixed-window denoising fit. Protect short clips.
    win = min(11, q.shape[0] if q.shape[0] % 2 else q.shape[0] - 1)
    smooth = savgol_filter(q, window_length=win, polyorder=3, axis=0)
    return np.gradient(smooth, 1.0 / fps, axis=0, edge_order=2)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def summarize(clip, path):
    fps = clip["fps"]
    q = clip["joint_pos"]
    v_reported = clip["joint_vel"]
    v_smoothed = velocity_from_positions(q, fps)
    accel = np.gradient(v_smoothed, 1.0 / fps, axis=0, edge_order=2)
    jerk = np.gradient(accel, 1.0 / fps, axis=0, edge_order=2)
    # Discard boundary values, which are unreliable for higher derivatives.
    trim = min(5, (q.shape[0] - 1) // 4)
    inner = slice(trim, -trim if trim else None)
    root_xyz = clip["body_pos_w"][:, 0, :]
    drift = np.linalg.norm(np.diff(root_xyz, axis=0), axis=-1)
    quat_norm = np.linalg.norm(clip["body_quat_w"], axis=-1)
    return {
        "clip": str(path),
        "category": "stylized" if "stylized" in str(path) else "standard",
        "kick_leg": clip.get("kick_leg", "unlabeled"),
        "frames": len(q),
        "fps": fps,
        "duration_s": round((len(q) - 1) / fps, 4),
        "joints": int(q.shape[1]),
        "bodies": int(clip["body_pos_w"].shape[1]),
        "joint_velocity_rms_rad_s": round(_rms(v_reported), 5),
        "joint_acceleration_rms_rad_s2": round(_rms(accel[inner]), 5),
        "joint_jerk_rms_rad_s3": round(_rms(jerk[inner]), 5),
        "pos_velocity_consistency_mae_rad_s": round(
            float(np.mean(np.abs(v_smoothed[inner] - v_reported[inner]))), 5
        ),
        "root_path_length_m": round(float(drift.sum()), 5),
        "quat_unit_norm_mean_abs_err": round(float(np.mean(np.abs(quat_norm - 1))), 6),
        "source_kind": "reference_motion_not_robot_rollout",
    }


def _temporal_resample(x, n):
    """Linear resampling for audit ONLY, not suitable for final naturalness scores."""
    src = np.linspace(0, 1, x.shape[0])
    dst = np.linspace(0, 1, n)
    flat = x.reshape(x.shape[0], -1)
    out = np.stack([np.interp(dst, src, flat[:, j])
                    for j in range(flat.shape[1])], axis=1)
    return out.reshape((n,) + x.shape[1:])


def compare_to_reference(ref, rollout):
    """Preliminary robot rollout diagnostic (NOT a human beauty metric)."""
    if ref["joint_pos"].shape[1:] != rollout["joint_pos"].shape[1:]:
        raise ValueError("Joint schemas differ: provide matched G1 order")
    if ref["body_pos_w"].shape[1:] != rollout["body_pos_w"].shape[1:]:
        raise ValueError("Body schemas differ: provide matched G1 order")
    n = ref["joint_pos"].shape[0]
    pred_q = _temporal_resample(rollout["joint_pos"], n)
    pred_xyz = _temporal_resample(rollout["body_pos_w"], n)
    ref_xyz = ref["body_pos_w"]
    # Root translation invariant, but NOT orientation invariant.
    pred_relative = pred_xyz - pred_xyz[:, :1, :]
    ref_relative = ref_xyz - ref_xyz[:, :1, :]
    return {
        "joint_angle_mae_rad": float(np.abs(ref["joint_pos"] - pred_q).mean()),
        "root_centered_body_pos_mpJPE_m": float(
            np.linalg.norm(ref_relative - pred_relative, axis=-1).mean()
        ),
        "timing_caveat": "Normalized-duration alignment can hide timing error; "
                         "report separately with synchronized kick events.",
        "interpretation": "Motion tracking diagnostics, NOT a naturalness score",
    }


def lag_seconds(signal_a, signal_b, fps, max_lag_s=0.6):
    """Positive lag => a follows b. Requires known semantic joint mapping."""
    a = np.asarray(signal_a, dtype=float)
    b = np.asarray(signal_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("Expected two same-length one-dimensional signals")
    a = a - a.mean()
    b = b - b.mean()
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    xc = correlate(a, b, mode="full", method="fft")
    lags = np.arange(-len(b) + 1, len(a))
    valid = np.abs(lags) <= min(int(round(max_lag_s * fps)), len(a) - 1)
    idx = np.argmax(np.abs(xc[valid]))
    return float(lags[valid][idx] / fps)


def inspect_coordination(clip, joint_map):
    """Only valid with explicit simulator-exported semantic joint names."""
    # Never apply a built-in 37-DoF G1 mapping to PAiD 29-DoF NPZ motions.
    if (joint_map.get("_num_joints") != clip["joint_pos"].shape[1]
            or joint_map.get("_verified_against_source") is not True):
        raise ValueError("joint_map must state matching _num_joints and "
                         "_verified_against_source=true after runtime validation")
    required = ("swing_hip_pitch", "swing_knee", "swing_ankle_pitch",
                "waist_yaw", "contralateral_shoulder_pitch")
    if not all(k in joint_map for k in required):
        raise ValueError("Required semantic joint map: " + ", ".join(required))
    q = clip["joint_vel"]
    for name in required:
        i = joint_map[name]
        if type(i) is not int or not (0 <= i < q.shape[1]):
            raise ValueError(f"invalid index for {name}: {i}")
    hip = q[:, joint_map["swing_hip_pitch"]]
    knee = q[:, joint_map["swing_knee"]]
    ankle = q[:, joint_map["swing_ankle_pitch"]]
    yaw = q[:, joint_map["waist_yaw"]]
    shoulder = q[:, joint_map["contralateral_shoulder_pitch"]]
    fps = clip["fps"]
    return {
        "hip_relative_knee_lag_s": lag_seconds(hip, knee, fps),
        "knee_relative_ankle_lag_s": lag_seconds(knee, ankle, fps),
        "waist_relative_shoulder_lag_s": lag_seconds(yaw, shoulder, fps),
        "note": "exploratory phase-lag diagnostics; calibrate against true "
                "human motion and labelled contact event, never call it 'beauty'",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="PAiD motion .npz file or directory")
    p.add_argument("--output", default="reports/reference_inventory.csv")
    p.add_argument("--reference", help="optional NPZ reference for rollout diagnostic")
    p.add_argument("--joint-map", help="optional JSON map from simulator joint_names")
    args = p.parse_args()
    target = Path(args.input)
    paths = sorted(target.rglob("*.npz")) if target.is_dir() else [target]
    if not paths:
        raise SystemExit(f"No .npz motions in {target}")
    rows = []
    for path in paths:
        clip = load_clip(path)
        row = summarize(clip, path)
        if args.joint_map:
            with open(args.joint_map, encoding="utf-8") as file:
                mapping = json.load(file)
            row.update(inspect_coordination(clip, mapping))
        rows.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with out.open("w", newline="", encoding="utf-8") as file:
        wr = csv.DictWriter(file, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"Audited {len(rows)} reference clips -> {out}")
    print(f"Categories: standard={sum(r['category']=='standard' for r in rows)}, "
          f"stylized={sum(r['category']=='stylized' for r in rows)}")
    if args.reference:
        if len(paths) != 1:
            raise SystemExit("--reference requires --input as one NPZ")
        result = compare_to_reference(load_clip(args.reference), load_clip(paths[0]))
        cmp_path = out.with_suffix(".comparison.json")
        cmp_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Motion tracking diagnostic -> {cmp_path}")


if __name__ == "__main__":
    main()
