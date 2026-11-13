#!/usr/bin/env python3
"""v145 source-locked, post-hoc sensitivity falsification of v144 human marker phase ordering.

No motor command, true contact labels, force inference, elite skill claim, GPU or Isaac Sim.
Raw CMU files are only read. Results are analysis-choice frequencies, not p-values.
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
import v144_human_temporal_source_and_physical_audit as v144

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "6a277845fd858a67361721b94a87f119756d68a0"
CONFIG_REL = "configs/v145_frozen_human_phase_robustness.json"


def frozen():
    locked = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", FREEZE_SHA + ":" + CONFIG_REL]
    )
    if (ROOT / CONFIG_REL).read_bytes() != locked:
        raise ValueError("V145_FROZEN_PROTOCOL_DRIFT")
    c = json.loads(locked)
    if (
        c["version"] != "v0.1.145"
        or c["parent_sha"] != "51c99a9a72816a981c47cc046f38ffebaf4c0809"
        or c["source_clips"] != v144.frozen()["source_clips"]
        or c["weak_anchor_clips"] != ["10_02", "10_05"]
        or c["rate_hz"] != 120
        or c["angle_halfspan_frames"] != [1, 2, 3, 4]
        or c["pre_ball_window_seconds"] != [0.4, 0.6, 0.8]
        or c["ball_weak_anchor_shift_frames"] != [-12, -6, 0, 6, 12]
        or c["median_rate_window_frames"] != [1, 5, 9]
        or c["strict_timing_lead_min_frames"] != 2
        or c["counts"]["all_chain_evaluations"] != 720
        or not all(c["evidence_policy"].values())
    ):
        raise ValueError("V145_FROZEN_ASSUMPTIONS_DRIFT")
    return c


def median_without_imputation(rate, width):
    """Only complete local windows contribute; don't fill absent frame/marker data."""
    x = np.asarray(rate, dtype=float)
    if x.ndim != 1 or width not in (1, 5, 9):
        raise ValueError("INVALID_SMOOTHING_WIDTH")
    if width == 1:
        return x.copy()
    out = np.full(len(x), np.nan)
    half = width // 2
    for i in range(half, len(x) - half):
        window = x[i - half:i + half + 1]
        if np.isfinite(window).all():
            out[i] = float(np.median(window))
    return out


def phase_order_test(rates, chain, anchor, pre_seconds, min_margin=2, min_fraction=.75,
                     fps=120):
    """Reports absolute observed peak indices; invalid windows cannot pass."""
    width = int(round(fps * pre_seconds))
    if (isinstance(anchor, bool) or not isinstance(anchor, int)
            or anchor - width < 0 or anchor > len(next(iter(rates.values())))):
        return {"status": "INVALID_ANCHOR_OR_WINDOW", "peak_indices": None,
                "lead_frames": None, "valid_fractions": None}
    values, peak_indices, validity = {}, {}, {}
    for segment in chain:
        r = rates[segment]
        if len(r) < anchor:
            return {"status": "INVALID_SOURCE_LENGTH", "peak_indices": None,
                    "lead_frames": None, "valid_fractions": None}
        window = r[anchor - width:anchor]
        good = np.isfinite(window)
        fraction = float(np.mean(good))
        validity[segment] = fraction
        if fraction < min_fraction or not good.any():
            return {"status": "INSUFFICIENT_VALID_MARKERS", "peak_indices": None,
                    "lead_frames": None, "valid_fractions": validity}
        ix = int(np.nanargmax(window))
        peak_indices[segment] = anchor - width + ix
        values[segment] = float(window[ix])
    a, b, d = [peak_indices[s] for s in chain]
    leads = [b - a, d - b]
    strict = a < b < d
    margin_pass = min(leads) >= min_margin
    return {"status": "ORDER_MARGIN_PASS" if margin_pass else
            ("STRICT_ORDER_SMALL_MARGIN" if strict else "REVERSED_OR_TIED"),
            "peak_indices": peak_indices,
            "lead_frames": leads,
            "peak_rates_rad_per_s": values,
            "valid_fractions": validity}


def load_motion(clip, quality_row, segments, root=v144.DATA):
    path = root / "raw" / quality_row["subject"] / (clip + ".c3d")
    labels, xyz = v144.marker_xyz(
        path, expected_frames=quality_row["original_C3D_frames"], expected_fps=120)
    points = v144.named_markers(labels, xyz, segments)
    return {name: points[pair[1]] - points[pair[0]]
            for name, pair in segments.items()}


def compute_grid(c, quality, *, data_root=v144.DATA, rates_method=None):
    if rates_method is None:
        rates_method = v144.observed_rate
    segments = v144.frozen()["segment_vectors"]
    rows = []
    for clip in c["weak_anchor_clips"]:
        qa = quality[clip]
        if qa["observed_ball_kinematics_quality"] != v144.QUALIFIED:
            raise ValueError("UNQUALIFIED_WEAK_ANCHOR_NOT_ADMITTED")
        ev = qa["inferred_event"]
        if ev["verified_true_foot_ball_contact"] is not False or ev["exact_contact_time_accuracy_proven"] is not False:
            raise ValueError("WEAK_ANCHOR_MISREPRESENTED_AS_CONTACT")
        anchor = ev["inferred_ball_launch_onset_frame"]
        if isinstance(anchor, bool) or not isinstance(anchor, int):
            raise ValueError("NONINTEGER_WEAK_ANCHOR")
        vectors = load_motion(clip, qa, segments, data_root)
        for span in c["angle_halfspan_frames"]:
            source_rates = {}
            for name, vector in vectors.items():
                rate, valid = rates_method(vector, fps=120, span=span)
                if valid.mean() < v144.frozen()["min_valid_ratio_per_segment"]:
                    raise ValueError("SOURCE_MARKER_QUALITY_DRIFT:" + clip + ":" + name)
                source_rates[name] = rate
            for median_width in c["median_rate_window_frames"]:
                rates = {name: median_without_imputation(rate, median_width)
                         for name, rate in source_rates.items()}
                for pre_seconds in c["pre_ball_window_seconds"]:
                    for shift in c["ball_weak_anchor_shift_frames"]:
                        for side, chain in c["chains"].items():
                            outcome = phase_order_test(
                                rates, chain, anchor + shift, pre_seconds,
                                min_margin=c["strict_timing_lead_min_frames"],
                                min_fraction=c["min_valid_window_fraction"])
                            rows.append({
                                "clip": clip, "subject": qa["subject"], "chain": side,
                                "span_frames": span,
                                "median_window_frames": median_width,
                                "pre_window_seconds": pre_seconds,
                                "anchor_shift_frames": shift,
                                "frozen_uncalibrated_weak_anchor_frame": anchor,
                                **outcome,
                            })
    expected = c["counts"]["all_chain_evaluations"]
    if len(rows) != expected:
        raise ValueError("GRID_DENOMINATOR_DRIFT")
    return rows


def summarize(rows, c):
    result = {}
    for side in c["chains"]:
        result[side] = {}
        for clip in c["weak_anchor_clips"]:
            subset = [r for r in rows if r["chain"] == side and r["clip"] == clip]
            statuses = Counter(r["status"] for r in subset)
            positive = [r for r in subset if r["status"] == "ORDER_MARGIN_PASS"]
            original = [r for r in subset if
                        r["span_frames"] == 2 and
                        r["median_window_frames"] == 1 and
                        r["pre_window_seconds"] == .6 and
                        r["anchor_shift_frames"] == 0]
            if len(subset) != 180 or len(original) != 1:
                raise ValueError("GRID_MISSING_BASELINE_OR_CELLS")
            result[side][clip] = {
                "denominator_all_registered_choices": len(subset),
                "status_counts": dict(statuses),
                "margin_pass_count": len(positive),
                "margin_pass_fraction_all_choices": len(positive) / len(subset),
                "baseline_v144_choice": original[0],
                "never_promote_grid_rate_to_population_p_value": True,
            }
    paired = {}
    for clip in c["weak_anchor_clips"]:
        a = {(r["span_frames"], r["median_window_frames"], r["pre_window_seconds"],
              r["anchor_shift_frames"]): r for r in rows
             if r["clip"] == clip and r["chain"] == "right"}
        b = {(r["span_frames"], r["median_window_frames"], r["pre_window_seconds"],
              r["anchor_shift_frames"]): r for r in rows
             if r["clip"] == clip and r["chain"] == "left"}
        if set(a) != set(b) or len(a) != 180:
            raise ValueError("CONTROL_PAIRING_DRIFT")
        paired[clip] = {
            "right_only_pass": sum(a[k]["status"] == "ORDER_MARGIN_PASS" and
                                   b[k]["status"] != "ORDER_MARGIN_PASS" for k in a),
            "left_only_pass": sum(b[k]["status"] == "ORDER_MARGIN_PASS" and
                                  a[k]["status"] != "ORDER_MARGIN_PASS" for k in a),
            "both_pass": sum(a[k]["status"] == b[k]["status"] == "ORDER_MARGIN_PASS" for k in a),
            "neither_pass": sum(a[k]["status"] != "ORDER_MARGIN_PASS" and
                                b[k]["status"] != "ORDER_MARGIN_PASS" for k in a),
        }
        if sum(paired[clip].values()) != 180:
            raise ValueError("PAIR_CONTROL_DENOMINATOR")
    return {"per_chain_event": result, "bilateral_specificity_control": paired}


def run(output):
    c = frozen()
    old = v144.frozen()
    quality, manifest = v144.fetch_qualified_sources(old)
    rows = compute_grid(c, quality)
    # Fail closed against accidentally treating 4 other clips as event-grounded.
    if set(c["weak_anchor_clips"]) != {x for x in old["source_clips"]
            if quality[x]["observed_ball_kinematics_quality"] == v144.QUALIFIED}:
        raise ValueError("SIX_CLIP_EVENT_DENOMINATOR_CHANGED")
    historical = v144.audit_historical_v53(old)
    import v142_native_pair_ledger as native
    old_v54 = native.historical_recheck()
    if (old_v54["historical_physics_5ms_frames"] != 6628 or
            historical["historical_real_PhysX_trials_sha_verified"] != 12):
        raise ValueError("HISTORICAL_G1_PHYSX_DENOMINATOR_CHANGED")
    report = {
        "version": c["version"],
        "protocol_commit_sha": FREEZE_SHA,
        "source_frame_count_unmodified": sum(x["original_C3D_frames"] for x in quality.values()),
        "original_clips_count": len(quality),
        "original_subjects_count": len({r["subject"] for r in quality.values()}),
        "weak_event_clips": list(c["weak_anchor_clips"]),
        "weak_event_subjects": list({quality[x]["subject"] for x in c["weak_anchor_clips"]}),
        "grid_all_registered_choices": len(rows),
        "description": "POSTHOC_SENSITIVITY_NOT_CONFIRMED_ELITE_BIOMECHANICS",
        "summary": summarize(rows, c),
        "raw_grid": rows,
        "historical_physics": {"v53": historical, "v54_actual_5ms_states":
                               old_v54["historical_physics_5ms_frames"]},
        "new_physical_rollouts": 0,
        "new_human_contact_or_grf_labels": 0,
        "g1_native_left_wrist_hip_pair_manifold": "UNKNOWN",
        "g1_motor_action_authorized": False,
    }
    if report["source_frame_count_unmodified"] != 3358:
        raise ValueError("SOURCE_FRAME_DENOMINATOR_DRIFT")
    import v142_native_pair_ledger as native_gate
    digest = native_gate.safe_write_once(output, report)
    return report, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, digest = run(args.output)
    summary = data["summary"]
    print(json.dumps({"output": str(args.output), "sha256": digest,
                      "by_side_and_clip": {
                          side: {clip: {"margin_pass": d["margin_pass_count"],
                                        "denominator": d["denominator_all_registered_choices"],
                                        "baseline": d["baseline_v144_choice"]["status"]}
                                 for clip,d in values.items()}
                          for side,values in summary["per_chain_event"].items()},
                      "contralateral": summary["bilateral_specificity_control"],
                      "new_physx": data["new_physical_rollouts"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
