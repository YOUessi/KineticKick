#!/usr/bin/env python3
"""v144 SHA-locked actual CMU six-clip 120Hz segment-rate proxy pilot + v53/v54 provenance.

CPU-only. A marker direction rate is NOT human joint angular velocity, impact, GRF,
power, elite motion, or a real G1 torque command. No phase inference on unanchored clips.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/you/projects/kinetickick/data/research_human_kicks/cmu_soccer_120hz_v050")
REAL = Path("/home/you/projects/kinetickick")
FREEZE_SHA = "4c220f2967fa6d9440e5159f4e4b6d30b911f537"
CONFIG = ROOT / "configs/v144_frozen_temporal_human_and_physical_audit.json"
QUALIFIED = "continuous_fast_outgoing_ball_marker_motion_no_true_contact_truth"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen():
    locked = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", FREEZE_SHA + ":" +
         str(CONFIG.relative_to(ROOT))])
    if CONFIG.read_bytes() != locked:
        raise ValueError("FROZEN_V144_CONFIG_CHANGED_AFTER_RESULTS")
    cfg = json.loads(locked)
    if (cfg["rate_hz"] != 120 or cfg["angle_center_halfspan_frames"] != 2
            or cfg["pre_window_s"] != 0.6
            or cfg["source_clips"] != ["10_01", "10_02", "10_03", "10_05", "10_06", "11_01"]
            or cfg["strong_launch_anchor_clips"] != ["10_02", "10_05"]
            or cfg["max_new_gpu_physx_runs"] != 0
            or cfg["min_event_subjects_for_holdout"] != 2
            or any(cfg["evidence_policy"].values())):
        raise ValueError("V144_PROTOCOL_ASSUMPTIONS_DRIFT")
    return cfg


def exact_sha(path, digest):
    if sha256(path) != digest:
        raise ValueError("SOURCE_SHA256_DRIFT: " + str(path))


def fetch_qualified_sources(cfg, data_root=DATA):
    audit_path = data_root / "processed_v050/quality_audit.json"
    manifest_path = data_root / "original_source_manifest.json"
    exact_sha(audit_path, cfg["source_data_sha256"]["quality_audit"])
    exact_sha(manifest_path, cfg["source_data_sha256"]["source_manifest"])
    audit, manifest = json.loads(audit_path.read_text()), json.loads(manifest_path.read_text())
    if (audit["version"] != "0.1.50" or manifest["version"] != "0.1.50"
            or audit["accurate_verified_human_impact_time_truth_sample_count"] != 0
            or audit["accurate_instestep_contact_force_sample_count"] != 0
            or audit["elite_professional_biomechanics_groundtruth_count"] != 0
            or manifest["source_is_certified_professional_elite_player"] is not False):
        raise ValueError("UNQUALIFIED_OR_DRIFTED_HUMAN_SOURCE")
    entries = {Path(row["path"]).name: row for row in manifest["files"]}
    rows = {row["id"]: row for row in audit["motions"]}
    if len(rows) != 6 or set(rows) != set(cfg["source_clips"]):
        raise ValueError("UNEXPECTED_HUMAN_CLIP_DENOMINATOR")
    for clip in cfg["source_clips"]:
        subject = clip.split("_", 1)[0]
        if rows[clip]["subject"] != subject:
            raise ValueError("SUBJECT_PROVENANCE_DRIFT")
        for extension in ("c3d", "amc"):
            name = clip + "." + extension
            if name not in entries:
                raise ValueError("MISSING_RAW_MOTION_MANIFEST")
            original = data_root / "raw" / subject / name
            if not original.is_file() or original.stat().st_size != entries[name]["bytes"]:
                raise ValueError("RAW_MOTION_SIZE_OR_PATH_DRIFT")
            exact_sha(original, entries[name]["sha256"])
    return rows, manifest


def marker_xyz(path, *, expected_frames, expected_fps):
    import ezc3d
    c3d = ezc3d.c3d(str(path))
    points = c3d["data"]["points"]
    labels = [str(s).strip().upper() for s in
              c3d["parameters"]["POINT"]["LABELS"]["value"]]
    rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    units = list(c3d["parameters"]["POINT"]["UNITS"]["value"])
    if (rate != expected_fps or units != ["mm"] or
            points.shape[0] != 4 or points.shape[-1] != expected_frames or
            not 0 < len(labels) <= points.shape[1]):
        raise ValueError("WRONG_OR_CORRUPT_REAL_C3D_METADATA")
    xyz = np.asarray(points[:3].transpose(2, 1, 0), float) * .001
    # CMU uses nonfinite / zero marker samples for tracking dropout.
    # Keep that missingness; per-segment masks reject it, never interpolate.
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("DUPLICATE_OR_MISSING_MARKER_NAMES")
    return labels, xyz


def named_markers(labels, xyz, pairs):
    required = {k for pair in pairs.values() for k in pair}
    index = {}
    for name in required:
        matches = [i for i, label in enumerate(labels)
                   if label.endswith(":" + name)]
        if len(matches) != 1:
            raise ValueError("AMBIGUOUS_ANATOMICAL_MARKER:" + name)
        index[name] = matches[0]
    return {name: xyz[:, i, :].copy() for name, i in index.items()}


def observed_rate(vector, *, span=2, fps=120):
    """Unsigned 4-frame orientation-rate proxy; NO zero/NaN imputation."""
    if vector.ndim != 2 or vector.shape[1] != 3 or len(vector) <= 2 * span:
        raise ValueError("INVALID_SEGMENT_VECTOR")
    norm = np.linalg.norm(vector, axis=1)
    valid = np.all(np.isfinite(vector), axis=1) & (norm > .05)
    unit = np.divide(vector, norm[:, None], out=np.zeros_like(vector),
                     where=valid[:, None])
    raw = np.full(len(vector), np.nan, dtype=float)
    c = np.einsum("ij,ij->i", unit[:-2 * span], unit[2 * span:])
    ok = valid[:-2 * span] & valid[2 * span:]
    raw[span:-span] = np.where(
        ok, np.arccos(np.clip(c, -1.0, 1.0)) * fps / (2 * span), np.nan)
    return raw, valid


def pre_onset_descriptor(rate, anchor, *, fps=120, pre_s=.6, min_valid=50):
    n = int(round(fps * pre_s))
    if not isinstance(anchor, int) or isinstance(anchor, bool) or anchor < n or anchor > len(rate):
        raise ValueError("MISSING_FROZEN_PRE_ONSET_WINDOW")
    window = rate[anchor - n:anchor]
    good = np.isfinite(window)
    if int(good.sum()) < min_valid:
        raise ValueError("INSUFFICIENT_UNIMPUTED_SEGMENT_WINDOW")
    ix = int(np.nanargmax(window))
    return {
        "valid_pre_frames": int(good.sum()),
        "window_frames": n,
        "max_orientation_rate_proxy_rad_per_s": float(window[ix]),
        "median_orientation_rate_proxy_rad_per_s": float(np.nanmedian(window)),
        "p90_orientation_rate_proxy_rad_per_s": float(np.nanpercentile(window, 90)),
        "peak_pre_ball_marker_onset_s": round((n - ix) / fps, 6),
        "first_order_time_index_pre": ix,
        "no_true_impact_time_or_force_witness": True,
    }


def six_clip_pilot(cfg, source_rows, root=DATA):
    segments = cfg["segment_vectors"]
    output = {}
    all_subjects, event_subjects = set(), set()
    for clip in cfg["source_clips"]:
        qa = source_rows[clip]
        subject = qa["subject"]
        all_subjects.add(subject)
        labels, xyz = marker_xyz(root / "raw" / subject / (clip + ".c3d"),
                                 expected_frames=qa["original_C3D_frames"],
                                 expected_fps=cfg["rate_hz"])
        points = named_markers(labels, xyz, segments)
        actual_strong = qa["observed_ball_kinematics_quality"] == QUALIFIED
        registered_strong = clip in cfg["strong_launch_anchor_clips"]
        if actual_strong != registered_strong:
            raise ValueError("UNDECLARED_EVENT_RECLASSIFICATION")
        if registered_strong:
            event = qa["inferred_event"]
            if (not isinstance(event, dict) or
                    event.get("verified_true_foot_ball_contact") is not False or
                    event.get("exact_contact_time_accuracy_proven") is not False or
                    "inferred_ball_launch_onset_frame" not in event):
                raise ValueError("STRONG_BALL_MARKER_EVENT_PROVENANCE_CHANGED")
            anchor = int(event["inferred_ball_launch_onset_frame"])
            event_subjects.add(subject)
        else:
            anchor = None
        summary = {}
        traces = {}
        for segment, pair in segments.items():
            rate, valid = observed_rate(
                points[pair[1]] - points[pair[0]],
                fps=cfg["rate_hz"], span=cfg["angle_center_halfspan_frames"])
            if valid.mean() < cfg["min_valid_ratio_per_segment"]:
                raise ValueError("REAL_HUMAN_MARKER_SEGMENT_QUALITY_TOO_LOW:" + clip + ":" + segment)
            traces[segment] = rate
            summary[segment] = {
                "valid_direct_marker_direction_fraction": float(valid.mean()),
                "all_clip_p90_orientation_rate_proxy_rad_per_s": float(np.nanpercentile(rate, 90)),
                "pre_ball_marker_launch_descriptor": (
                    pre_onset_descriptor(rate, anchor,
                        fps=cfg["rate_hz"], pre_s=cfg["pre_window_s"],
                        min_valid=cfg["min_pre_event_valid_frames"])
                    if anchor is not None else None),
            }
        output[clip] = {
            "subject_id": subject, "frames_original_120hz": len(xyz),
            "unverified_ball_marker_launch_anchor_frame": anchor,
            "anchor_type": ("WEAK_OBSERVED_MARKER_LAUNCH_NOT_CONTACT"
                            if anchor is not None else "NO_QUALIFIED_EVENT_ANCHOR"),
            "measured_human_impact": False, "measured_human_GRF": False,
            "professional_footballer_certified": False,
            "segment_direction_rates_are_not_joint_angular_velocity": True,
            "segments": summary,
        }
    return {
        "raw_source_clips": len(output), "raw_source_subjects": len(all_subjects),
        "qualified_weak_ball_marker_launch_events": sum(
            v["unverified_ball_marker_launch_anchor_frame"] is not None
            for v in output.values()),
        "distinct_subjects_with_weak_event": len(event_subjects),
        "subject_disjoint_temporal_event_model_evaluation_eligible":
             len(event_subjects) >= cfg["min_event_subjects_for_holdout"],
        "clips_without_valid_ball_marker_anchor": sum(
            v["unverified_ball_marker_launch_anchor_frame"] is None
            for v in output.values()),
        "raw_original_c3d_frames": sum(v["frames_original_120hz"] for v in output.values()),
        "per_clip": output,
    }


def audit_historical_v53(cfg, *, physical_root=REAL):
    audit_path = physical_root / "reports/v053_cmu_frozen_matrix_audit.json"
    exact_sha(audit_path, cfg["source_data_sha256"]["v053_original_strict_audit"])
    audit = json.loads(audit_path.read_text())
    expected_policies = {"frozen_v32", "cmu_10_02", "cmu_10_05"}
    if (audit["n_original_PhySX_runs"] != 12 or audit["n_ball_conditions"] != 4
            or len(audit["comparisons"]) != 4 or
            not audit["no_intermediate_hyperparameter_tuning"] or
            not audit["no_controller_promotion"]):
        raise ValueError("V53_AUDIT_DENOMINATOR_OR_TUNING_DRIFT")
    seen = set()
    success = Counter()
    failures = Counter()
    verified_bytes = 0
    for cell in audit["comparisons"]:
        if cell["id"] in seen or set(cell["policies"]) != expected_policies:
            raise ValueError("V53_MISSING_OR_DUPLICATE_POLICY_CELL")
        seen.add(cell["id"])
        for name, result in cell["policies"].items():
            filename = result["source"]
            p = physical_root / filename
            if p.is_symlink() or not p.is_file():
                raise ValueError("V53_SOURCE_MISSING_OR_SYMLINKED")
            exact_sha(p, result["sha256"])
            verified_bytes += p.stat().st_size
            if result["success"] != all(result["gates"].values()):
                raise ValueError("V53_UNJUSTIFIED_STRICT_SUCCESS")
            success[name] += int(result["success"])
            for key, admitted in result["gates"].items():
                if admitted is not True:
                    failures[(name, key)] += 1
    if any(success[x] != 0 or audit["summary"][x]["success_count"] != 0
           for x in expected_policies):
        raise ValueError("HISTORICAL_PHYSICAL_NEGATIVE_INCORRECT")
    return {
        "historical_real_PhysX_trials_sha_verified": cfg["original_v53_expected_runs"],
        "historical_source_bytes_sha_verified": verified_bytes,
        "ball_conditions": len(seen),
        "strict_success_by_policy": dict(success),
        "failed_gates_by_policy": {
            name: {key: failures[(name, key)] for n, key in failures if n == name}
            for name in sorted(expected_policies)},
        "new_physx_episodes": 0, "no_training_promoted": True,
    }


def run():
    import v142_native_pair_ledger as old
    cfg = frozen()
    qa, manifest = fetch_qualified_sources(cfg)
    human = six_clip_pilot(cfg, qa)
    physical = audit_historical_v53(cfg)
    v54 = old.historical_recheck()
    if (human["raw_original_c3d_frames"] != 3358
            or human["qualified_weak_ball_marker_launch_events"] != 2
            or human["distinct_subjects_with_weak_event"] != 1
            or v54["historical_physics_5ms_frames"] != cfg["original_v54_expected_frames"]):
        raise ValueError("PREDECLARED_DATA_DENOMINATOR_OR_PHYSX_DRIFT")
    return {
        "version": cfg["version"], "protocol_sha": FREEZE_SHA,
        "original_source_licence_independent_verification": False,
        "upstream_TLS_and_signature_authoritative_attestation": False,
        "human_real_C3D_marker_temporal_pilot": human,
        "historical_v53_strict_physics": physical,
        "historical_v54_requalification": v54,
        "native_original_G1_physx_pair_manifold_truth": "UNKNOWN",
        "new_real_PhysX_episodes": 0, "new_motor_training_or_actuation": 0,
        "professional_human_power_chain_learned": False,
        "strict_physical_shot_success_claimed": False,
        "no_source_or_evidence_mutation": True,
    }


def write_once(path, obj):
    path = Path(path)
    if path.is_symlink() or not path.parent.is_dir() or path.exists():
        raise ValueError("EVIDENCE_OUTPUT_OVERWRITE_OR_UNSAFE_PATH")
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf8") as stream:
        stream.write(payload)
    return sha256(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    r = run()
    hash_ = write_once(args.output, r)
    h = r["human_real_C3D_marker_temporal_pilot"]
    print(json.dumps({
        "output": str(args.output), "sha256": hash_,
        "original_c3d_clips": h["raw_source_clips"],
        "c3d_frames": h["raw_original_c3d_frames"],
        "weak_ball_marker_event_clips": h["qualified_weak_ball_marker_launch_events"],
        "event_subjects": h["distinct_subjects_with_weak_event"],
        "heldout_event_subject_available": h["subject_disjoint_temporal_event_model_evaluation_eligible"],
        "original_v53_source_trials_sha_verified": r["historical_v53_strict_physics"]["historical_real_PhysX_trials_sha_verified"],
        "original_v54_200hz_frames_sha_verified": r["historical_v54_requalification"]["historical_physics_5ms_frames"],
        "new_physx": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
