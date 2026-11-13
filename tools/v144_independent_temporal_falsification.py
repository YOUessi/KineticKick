#!/usr/bin/env python3
"""Independent v144 C3D time-order falsification; does NOT endorse impact truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import v144_human_temporal_source_and_physical_audit as source


def independent_rate(v, fps=120, span=2):
    """Recompute with atan2(|cross|, dot), not v144 arccos(dot)."""
    v = np.asarray(v, dtype=float)
    norms = np.linalg.norm(v, axis=1)
    valid = np.isfinite(v).all(axis=1) & (norms > 0.05)
    u = np.zeros_like(v)
    u[valid] = v[valid] / norms[valid, None]
    a, b = u[:-2 * span], u[2 * span:]
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    dot = np.einsum("ij,ij->i", a, b)
    speed = np.full(len(v), np.nan, dtype=float)
    ok = valid[:-2 * span] & valid[2 * span:]
    core = speed[span:-span]
    core[ok] = np.arctan2(cross[ok], dot[ok]) * fps / (2 * span)
    return speed


def independently_verify_report(report, cfg, *, root=source.DATA):
    if report["new_real_PhysX_episodes"] != 0 or report["new_motor_training_or_actuation"] != 0:
        raise ValueError("SYNTHETIC_REVIEW_PRETENDS_LIVE_PHYSX")
    if report["professional_human_power_chain_learned"] or report["strict_physical_shot_success_claimed"]:
        raise ValueError("HUMAN_MOTION_OR_SHOT_MISREPRESENTED")
    h = report["human_real_C3D_marker_temporal_pilot"]
    if (set(h["per_clip"]) != set(cfg["source_clips"]) or
            h["raw_source_clips"] != 6 or h["raw_original_c3d_frames"] != 3358
            or h["qualified_weak_ball_marker_launch_events"] != 2
            or h["distinct_subjects_with_weak_event"] != 1 or
            h["subject_disjoint_temporal_event_model_evaluation_eligible"]):
        raise ValueError("BAD_DENOMINATOR_OR_FALSE_HELDOUT_GENERALIZATION")
    physical = report["historical_v53_strict_physics"]
    if (physical["historical_real_PhysX_trials_sha_verified"] != 12 or
            physical["new_physx_episodes"] != 0 or
            any(v != 0 for v in physical["strict_success_by_policy"].values())):
        raise ValueError("FAKE_PHYSICAL_SUCCESS")
    if report["historical_v54_requalification"]["historical_physics_5ms_frames"] != 6628:
        raise ValueError("HISTORICAL_PHYSX_DRIFT")
    max_error = 0.0
    event_records = {}
    for clip in cfg["source_clips"]:
        recorded = h["per_clip"][clip]
        if (recorded["measured_human_impact"] or recorded["measured_human_GRF"] or
                recorded["professional_footballer_certified"]):
            raise ValueError("FALSE_HUMAN_FORCE_OR_IMPACT_PROVENANCE")
        if recorded["subject_id"] != clip.split("_", 1)[0]:
            raise ValueError("WRONG_SUBJECT_FOR_SOURCE")
        anchor = recorded["unverified_ball_marker_launch_anchor_frame"]
        if (anchor is None) != (clip not in cfg["strong_launch_anchor_clips"]):
            raise ValueError("UNFROZEN_EVENT_INCLUSION")
        labels, xyz = source.marker_xyz(
            root / "raw" / recorded["subject_id"] / (clip + ".c3d"),
            expected_frames=recorded["frames_original_120hz"],
            expected_fps=cfg["rate_hz"])
        pts = source.named_markers(labels, xyz, cfg["segment_vectors"])
        desc = {}
        for name, (tail, head) in cfg["segment_vectors"].items():
            vector = pts[head] - pts[tail]
            independent = independent_rate(vector)
            primary, _ = source.observed_rate(vector)
            if not np.array_equal(np.isnan(independent), np.isnan(primary)):
                raise ValueError("INDEPENDENT_SOURCE_MISSINGNESS_MISMATCH")
            all_valid = np.isfinite(independent)
            if np.any(all_valid):
                max_error = max(max_error, float(np.max(
                    np.abs(independent[all_valid] - primary[all_valid]))))
            expected = recorded["segments"][name]["pre_ball_marker_launch_descriptor"]
            if anchor is None:
                if expected is not None:
                    raise ValueError("UNQUALIFIED_EVENT_MOTION_REFERENCE")
                continue
            width = int(cfg["pre_window_s"] * cfg["rate_hz"])
            window = independent[anchor - width:anchor]
            if np.isfinite(window).sum() < cfg["min_pre_event_valid_frames"]:
                raise ValueError("BAD_RAW_SEGMENT_COVERAGE")
            ix = int(np.nanargmax(window))
            observed_lag = round((width - ix) / cfg["rate_hz"], 6)
            err = abs(float(window[ix]) - expected["max_orientation_rate_proxy_rad_per_s"])
            max_error = max(max_error, err)
            if (err > 1e-5 or
                    abs(observed_lag - expected["peak_pre_ball_marker_onset_s"]) > 1e-6):
                raise ValueError("INDEPENDENT_ATAN2_TEMPORAL_RECONSTRUCTION_MISMATCH")
            desc[name] = observed_lag
        if anchor is not None:
            event_records[clip] = desc
    if set(event_records) != set(cfg["strong_launch_anchor_clips"]):
        raise ValueError("INCORRECT_HUMAN_EVENT_DENOMINATOR")
    key = ("pelvis", "right_thigh", "right_shank")
    true_orders = [
        all(x[k1] > x[k2] for k1, k2 in zip(key[:-1], key[1:]))
        for x in event_records.values()]
    reverse_orders = [
        all((cfg["pre_window_s"] + 1 / cfg["rate_hz"] - x[k1]) >
            (cfg["pre_window_s"] + 1 / cfg["rate_hz"] - x[k2])
            for k1, k2 in zip(key[:-1], key[1:]))
        for x in event_records.values()]
    # Exploratory (not pre-frozen) deterministic phase-shift diagnostic.
    # NOT an inferential p-value: same subject, inspected examples, no new labels.
    rng = np.random.default_rng(144)
    hits = 0
    for _ in range(4096):
        both = True
        for clip in cfg["strong_launch_anchor_clips"]:
            lags = rng.integers(0, 72, size=3) / cfg["rate_hz"]
            if not (lags[0] > lags[1] > lags[2]):
                both = False
        hits += int(both)
    return {
        "version": "v0.1.144",
        "independent_numerical_method": "atan2_cross_dot_4_sample_orientation_change",
        "max_abs_segment_rate_reconstruction_difference_rad_per_s": max_error,
        "human_weak_event_clips": list(event_records),
        "descriptive_proximal_to_distal_right_segment_peak_order_by_clip": {
            clip: verdict for clip, verdict in zip(event_records, true_orders)},
        "counterfactual_reverse_pre_window_order_preserved_by_clip": {
            clip: verdict for clip, verdict in zip(event_records, reverse_orders)},
        "deterministic_4096_independent_phase_shift_both_order_rate": hits / 4096,
        "counterfactual_not_valid_human_motion": True,
        "different_subjects_in_event_cohort": 1,
        "no_valid_population_p_value_or_elite_power_conclusion": True,
        "strict_v53_success": physical["strict_success_by_policy"],
        "new_physx": 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.audit.read_text())
    result = independently_verify_report(report, source.frozen())
    digest = source.write_once(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": digest, **result},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
