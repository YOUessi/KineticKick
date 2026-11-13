#!/usr/bin/env python3
"""Read-only qualification of actual CMU subject10 C3D soccer-like motion.

C3D 120Hz human marker position is NOT measured ground reaction, precise
instep impact time, professional biomechanical provenance or G1 training.
All source bytes stay local. Report metadata and SHA digests only.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import math
from pathlib import Path
import sys

from v142_native_pair_ledger import frozen, safe_write_once, sha256

DATA_ROOT = Path("/home/you/projects/kinetickick/data/research_human_kicks/cmu_soccer_120hz_v050")
CLIPS = ("10_02", "10_05")


def metadata_c3d(path):
    """Use actual C3D parser and check 3D arrays, not file-name heuristics."""
    import ezc3d
    c3d = ezc3d.c3d(str(path))
    point = c3d["data"]["points"]
    analog = c3d["data"]["analogs"]
    fps = float(c3d["header"]["points"]["frame_rate"])
    frames = int(point.shape[-1])
    if point.shape[0] != 4 or frames < 2 or fps != 120.0:
        raise ValueError("C3D_REAL_120HZ_SHAPE_OR_RATE_FAILED")
    analog_channels = int(analog.shape[1])
    labels = [str(x) for x in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    if len(labels) > point.shape[1]:
        raise ValueError("BAD_REAL_C3D_MARKER_LABEL_COUNT")
    return {
        "c3d_sha256": sha256(path), "frames": frames, "rate_hz": fps,
        "raw_marker_slots": int(point.shape[1]), "labeled_marker_slots": len(labels),
        "analog_channel_count": analog_channels,
        "analog_channel_data_observed": analog_channels > 0,
        "no_analog_channels_means_no_direct_C3D_forceplate_recording": analog_channels == 0,
        "synchronized_forceplate_force_truth_from_this_file": False,
        "true_human_ball_contact_from_this_file": False,
    }


def audit_quality_reference(qa, clip):
    rows = [x for x in qa["motions"] if x.get("id") == clip and x.get("subject") == "10"]
    if len(rows) != 1:
        raise ValueError("MISSING_OR_DUPLICATE_HUMAN_QUALITY_AUDIT_ROW")
    row = rows[0]
    required = ("original_C3D_frames", "original_3D_rate_hz",
                "original_human_mocap_qualified_for_pose_priors",
                "real_instestep_force_and_contact_truth_qualified")
    if any(k not in row for k in required):
        raise ValueError("MISSING_HUMAN_SOURCE_QUALIFICATION")
    if (row["original_3D_rate_hz"] != 120.0
            or row["original_human_mocap_qualified_for_pose_priors"] is not True
            or row["real_instestep_force_and_contact_truth_qualified"] is not False):
        raise ValueError("DRIFTED_HUMAN_QUALIFICATION")
    return row


def review_assets(root=DATA_ROOT, clips=CLIPS):
    frozen()
    root = Path(root)
    qa_path = root / "processed_v050/quality_audit.json"
    manifest_path = root / "original_source_manifest.json"
    qa = json.loads(qa_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if qa["version"] != "0.1.50" or manifest["version"] != "0.1.50":
        raise ValueError("HUMAN_DATA_VERSION_CHANGED")
    if (qa["accurate_verified_human_impact_time_truth_sample_count"] != 0
            or qa["accurate_instestep_contact_force_sample_count"] != 0
            or qa["elite_professional_biomechanics_groundtruth_count"] != 0
            or manifest["source_is_certified_professional_elite_player"] is not False
            or manifest["original_source_has_real_physical_ball_contact_or_impulse_truth"] is not False):
        raise ValueError("HUMAN_SOURCE_CLAIMS_HAVE_CHANGED_REQUALIFICATION_REQUIRED")
    result = {}
    subjects = set()
    for clip in clips:
        if clip not in CLIPS or clip in result:
            raise ValueError("UNDECLARED_OR_DUPLICATE_HUMAN_CLIP")
        path = root / "raw/10" / (clip + ".c3d")
        row = audit_quality_reference(qa, clip)
        parsed = metadata_c3d(path)
        if parsed["frames"] != row["original_C3D_frames"]:
            raise ValueError("C3D_AND_EXISTING_AUDIT_FRAME_COUNT_MISMATCH")
        subjects.add("10")
        result[clip] = {
            **parsed,
            "subject_id": "10",
            "raw_ball_marker_triplet_valid_frame_fraction": row.get("original_three_ball_markers_all_valid_fraction"),
            "ball_track_quality": row.get("observed_ball_kinematics_quality"),
            "human_pose_prior_only": True,
            "measured_instep_contact_impulse": False,
            "measured_human_grf": False,
            "elite_football_technique_certified": False,
        }
    return {
        "version": "v0.1.142", "source_audit_sha256": sha256(qa_path),
        "source_manifest_sha256": sha256(manifest_path),
        "source_research_license_claim": manifest.get("source_dataset_licensed_for_research"),
        "source_upstream_checksum_attested": not manifest.get("local_SHA256_not_equivalent_to_signed_upstream_checksum", True),
        "source_transport_TLS_chain_verified": not manifest.get("source_TLS_issuer_chain_verify_failed_on_device", True),
        "clips": result, "clip_count": len(result),
        "distinct_subject_count": len(subjects),
        "subject_disjoint_train_test_possible_with_these_two_clips": len(subjects) >= 2,
        "measured_human_impact_events": 0,
        "measured_human_instep_forces": 0,
        "synchronized_real_ground_reaction_force_samples": 0,
        "elite_professional_biomechanics_groundtruth_samples": 0,
        "can_supervise_unqualified_human_pose_priors": True,
        "can_supervise_true_pro_human_power_chain": False,
        "can_estimate_foot_ball_true_contact_timing": False,
        "can_truthfully_train_end_to_end_powerful_instep_kick": False,
        "data_mutations": 0,
        "new_G1_training_steps": 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = review_assets()
    s = safe_write_once(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": s, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
