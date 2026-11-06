"""Inspect an original CMU C3D without relabeling robot motion as human truth.

This bounded reader supports Intel, floating-point, no-analog C3D files. It is
not a general C3D library. Unknown layouts are rejected. Exported coordinates
remain in the source laboratory frame; only mm -> m is applied. No filtering,
interpolation, joint-center fitting, skill classification or impact detection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

# CMU's documented 41 physical human markers. LFIN/RFIN are physical knuckle
# markers, unlike the artificial finger/thumb joints in the ASF skeleton.
CMU_HUMAN_MARKERS = frozenset(
    "LFHD RFHD LBHD RBHD C7 T10 CLAV STRN RBAC "
    "LSHO LUPA LELB LFRM LWRA LWRB LFIN "
    "RSHO RUPA RELB RFRM RWRA RWRB RFIN "
    "LFWT RFWT LBWT RBWT LTHI LKNE LSHN LANK LHEE LTOE LMT5 "
    "RTHI RKNE RSHN RANK RHEE RTOE RMT5".split()
)


def _parameters(blob: bytes) -> dict:
    if len(blob) < 1024 or blob[1] != 80:
        raise ValueError("Not a supported C3D header")
    start = (blob[0] - 1) * 512
    if start < 512 or start + 4 > len(blob) or blob[start + 3] != 84:
        raise ValueError("Only Intel C3D parameter encoding is supported")
    stop = start + blob[start + 2] * 512
    groups, records = {}, []
    pos = start + 4
    while pos + 2 <= min(stop, len(blob)):
        length, group_id = struct.unpack_from("<bb", blob, pos)
        if length == 0 or group_id == 0:
            break
        name_end = pos + 2 + abs(length)
        name = blob[pos + 2:name_end].decode("ascii").upper()
        offset = struct.unpack_from("<H", blob, name_end)[0]
        if offset == 0:
            break
        next_pos = name_end + offset
        if offset < 2 or next_pos > min(stop, len(blob)):
            raise ValueError("C3D parameter record exceeds its block")
        payload = blob[name_end + 2:next_pos]
        if group_id < 0:
            groups[-group_id] = name
        else:
            dtype, ndim = struct.unpack_from("<bB", payload)
            if dtype not in (-1, 1, 2, 4):
                raise ValueError("Unsupported C3D parameter type")
            dims = tuple(payload[2:2 + ndim])
            size = abs(dtype) * int(np.prod(dims, dtype=np.int64))
            raw = payload[2 + ndim:2 + ndim + size]
            if len(raw) != size:
                raise ValueError("Truncated C3D parameter value")
            records.append((group_id, name, dtype, dims, raw))
        pos = next_pos
    return {f"{groups[g]}:{n}": (t, d, r) for g, n, t, d, r in records}


def _number(parameters: dict, name: str, default=None):
    if name not in parameters:
        return default
    dtype, dims, raw = parameters[name]
    if dtype not in (1, 2, 4) or len(raw) != dtype:
        raise ValueError(f"Expected numeric scalar {name}")
    return struct.unpack({1: "<b", 2: "<h", 4: "<f"}[dtype], raw)[0]


def _strings(parameters: dict, name: str) -> list[str]:
    dtype, dims, raw = parameters[name]
    if dtype != -1 or not dims:
        raise ValueError(f"Expected fixed-width strings {name}")
    width = dims[0]
    if width == 0:
        return []
    return [raw[i:i + width].decode("ascii").strip(" \x00")
            for i in range(0, len(raw), width)]


def read_cmu_c3d(path: Path) -> tuple[dict, dict]:
    """Return source metadata plus arrays; invalid observations become NaN."""
    blob = path.read_bytes()
    parameters = _parameters(blob)
    used, analog_count, first, last = struct.unpack_from("<HHHH", blob, 2)
    scale = struct.unpack_from("<f", blob, 12)[0]
    data_block = struct.unpack_from("<H", blob, 16)[0]
    fps = struct.unpack_from("<f", blob, 20)[0]
    frames = last - first + 1
    if scale >= 0 or analog_count != 0:
        raise ValueError("Only floating-point, no-analog C3D captures are supported")
    if not (1 <= frames * used <= 5_000_000) or fps <= 0:
        raise ValueError("Invalid or excessive capture dimensions")
    for key, value in [("POINT:USED", used), ("POINT:FRAMES", frames),
                       ("POINT:DATA_START", data_block), ("POINT:RATE", fps)]:
        if _number(parameters, key) != value:
            raise ValueError(f"Header/parameter mismatch: {key}")
    units = _strings(parameters, "POINT:UNITS")
    if units != ["mm"]:
        raise ValueError(f"Only explicit POINT:UNITS=mm is supported; got {units}")
    labels = _strings(parameters, "POINT:LABELS")
    if "POINT:LABELS2" in parameters:
        labels += _strings(parameters, "POINT:LABELS2")
    if len(labels) != used or len(set(labels)) != used:
        raise ValueError("C3D point labels must match and uniquely identify points")
    start = (data_block - 1) * 512
    required = frames * used * 4 * 4
    if len(blob) < start + required:
        raise ValueError("Truncated C3D point records")
    points = np.frombuffer(blob, dtype="<f4", count=frames * used * 4,
                           offset=start).reshape(frames, used, 4)
    packed = points[..., 3].astype(np.int32)
    valid = (packed >= 0) & np.isfinite(points[..., :3]).all(axis=2)
    xyz_m = points[..., :3].astype(np.float64) / 1000.0
    xyz_m[~valid] = np.nan
    human_indices = [i for i, label in enumerate(labels)
                     if label.split(":")[-1] in CMU_HUMAN_MARKERS
                     and not label.startswith("Soccerball:")]
    if len(human_indices) != 41:
        raise ValueError("Capture does not contain exactly the documented 41 human markers")
    ball_indices = [i for i, label in enumerate(labels) if label.startswith("Soccerball:")]
    observed_names = [labels[i].split(":")[-1] for i in human_indices]
    if set(observed_names) != CMU_HUMAN_MARKERS:
        raise ValueError("Human marker set contains duplicates or omissions")
    summary = {
        "schema_version": 1,
        "source_file": str(path.resolve()),
        "source_sha256": hashlib.sha256(blob).hexdigest(),
        "source_class": "original_human_optical_mocap",
        "capture": {
            "frames": frames, "fps": fps, "sample_interval_s": 1.0 / fps,
            "first_source_frame": first, "last_source_frame": last,
            "last_relative_timestamp_s": (frames - 1) / fps,
            "stored_points": used, "measured_human_markers": len(human_indices),
            "ball_markers": len(ball_indices),
            "other_points_not_used_as_measured_markers": used - len(human_indices) - len(ball_indices),
            "source_units": "mm", "export_units": "m",
            "coordinate_frame": "source laboratory frame; no orientation/translation normalization",
            "event_count": _number(parameters, "EVENT:USED", 0),
            "force_platform_count": _number(parameters, "FORCE_PLATFORM:USED", 0),
            "analog_channels": _number(parameters, "ANALOG:USED", 0),
        },
        "annotations": {"skill_level": None, "kick_subtype": None,
                        "contact_frame": None, "ball_center_calibration": None,
                        "human_joint_centers": "not exported; physical surface markers only"},
        "quality": {
            "human_valid_fraction": float(valid[:, human_indices].mean()),
            "human_marker_valid_fraction": {labels[i]: float(valid[:, i].mean()) for i in human_indices},
            "ball_marker_valid_fraction": {labels[i]: float(valid[:, i].mean()) for i in ball_indices},
            "ball_marker_valid_source_frame_ranges": {},
            "body_shape": "subject-specific ASF bone lengths available separately; no measured mass/inertias or SMPL shape",
            "processing": "No interpolation, filtering, coordinate alignment or impact annotation performed.",
            "validity_limit": "A nonnegative stored residual flag means present in this processed C3D, not independently verified accuracy or absence of upstream gap filling.",
        },
        "limits": [
            "CMU early-session quality caveat applies to low subject number 10.",
            "Ball marker positions are not a calibrated ball center and contain gaps.",
            "No force plate, contact impulse or annotated foot-ball event is present.",
            "Professional skill and an instep contact surface are unverified.",
            "Additional C3D model-output points are not physical measured markers.",
        ],
    }
    for i in ball_indices:
        indices = np.flatnonzero(valid[:, i])
        ranges = []
        if len(indices):
            splits = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            ranges = [[int(s[0] + first), int(s[-1] + first)] for s in splits]
        summary["quality"]["ball_marker_valid_source_frame_ranges"][labels[i]] = ranges
    arrays = {
        "fps": np.array(fps), "time_s": np.arange(frames, dtype=float) / fps,
        "source_frame_numbers": np.arange(first, last + 1),
        "human_marker_names": np.array(observed_names),
        "human_source_labels": np.array([labels[i] for i in human_indices]),
        "human_markers_m": xyz_m[:, human_indices],
        "human_marker_valid": valid[:, human_indices],
        "human_residual_mm": np.where(valid[:, human_indices], (packed[:, human_indices] & 255) * abs(scale), np.nan),
        "human_zero_residual_flag": valid[:, human_indices] & ((packed[:, human_indices] & 255) == 0),
        "ball_marker_names": np.array([labels[i] for i in ball_indices]),
        "ball_markers_m": xyz_m[:, ball_indices],
        "ball_marker_valid": valid[:, ball_indices],
        "ball_residual_mm": np.where(valid[:, ball_indices], (packed[:, ball_indices] & 255) * abs(scale), np.nan),
        "ball_zero_residual_flag": valid[:, ball_indices] & ((packed[:, ball_indices] & 255) == 0),
        "source_sha256": np.array(summary["source_sha256"]),
        "source_class": np.array(summary["source_class"]),
    }
    return summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmu-c3d", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    summary, arrays = read_cmu_c3d(args.cmu_c3d)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.export, **arrays)
        summary["export"] = {"path": str(args.export.resolve()),
                             "sha256": hashlib.sha256(args.export.read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"frames": summary["capture"]["frames"],
                      "measured_human_markers": summary["capture"]["measured_human_markers"],
                      "report": str(args.report), "export": str(args.export) if args.export else None}))


if __name__ == "__main__":
    main()
