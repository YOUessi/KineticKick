#!/usr/bin/env python3
"""Independent atan2 (cross,dot) audit of every frozen v145 full-grid cell.

No real contact truth and no native PhysX execution. This code runs anew on
SHA-verified original C3D, not on the primary computed angular-rate arrays.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import v144_human_temporal_source_and_physical_audit as source
import v145_phase_order_robustness as primary


def atan2_observed_rate(vector, *, fps=120, span=2):
    if vector.ndim != 2 or vector.shape[1] != 3 or len(vector) <= 2*span:
        raise ValueError("BAD_ORIGINAL_3D_VECTOR")
    norms = np.linalg.norm(vector, axis=1)
    good = np.isfinite(vector).all(axis=1) & (norms > .05)
    unit = np.zeros_like(vector)
    np.divide(vector, norms[:, None], where=good[:, None], out=unit)
    out = np.full(len(vector), np.nan)
    for i in range(span, len(vector)-span):
        if good[i-span] and good[i+span]:
            left,right = unit[i-span],unit[i+span]
            cross = float(np.linalg.norm(np.cross(left,right)))
            dot = float(np.dot(left,right))
            out[i] = float(np.arctan2(cross,dot)) * fps / (2*span)
    return out,good


def review(input_path, output_path):
    c = primary.frozen()
    old = source.frozen()
    quality,manifest = source.fetch_qualified_sources(old)
    from_disk = json.loads(Path(input_path).read_text())
    if (from_disk["version"] != "v0.1.145"
            or from_disk["protocol_commit_sha"] != primary.FREEZE_SHA
            or from_disk["source_frame_count_unmodified"] != 3358
            or from_disk["new_physical_rollouts"] != 0
            or from_disk["g1_native_left_wrist_hip_pair_manifold"] != "UNKNOWN"):
        raise ValueError("PRIMARY_EVIDENCE_PROVENANCE_REJECTED")
    reconstructed=primary.compute_grid(c,quality,rates_method=atan2_observed_rate)
    observed=from_disk["raw_grid"]
    if len(reconstructed) != 720 or len(observed) != 720:
        raise ValueError("REGISTERED_CELLS_MISSING")
    rates = 0
    exact_status=0
    same_indices=0
    deviations=[]
    for i,(a,b) in enumerate(zip(observed,reconstructed)):
        metadata=("clip","chain","span_frames","median_window_frames","pre_window_seconds","anchor_shift_frames","frozen_uncalibrated_weak_anchor_frame")
        if any(a[k]!=b[k] for k in metadata):
            raise ValueError("GRID_ORDER_OR_METADATA_CHANGED")
        if a["status"]==b["status"]:
            exact_status+=1
        else:
            deviations.append({"index":i,"kind":"STATUS","original":a["status"],"independent":b["status"]})
        if a["peak_indices"]==b["peak_indices"]:
            same_indices+=1
        else:
            deviations.append({"index":i,"kind":"PEAK_INDEX","original":a["peak_indices"],"independent":b["peak_indices"]})
        if a.get("peak_rates_rad_per_s") and b.get("peak_rates_rad_per_s"):
            for segment in a["peak_rates_rad_per_s"]:
                rates=max(rates,abs(a["peak_rates_rad_per_s"][segment]-b["peak_rates_rad_per_s"][segment]))
    independent_summary=primary.summarize(reconstructed,c)
    if from_disk["summary"] != independent_summary:
        # The rates are floats, so baseline peak rate fields can differ at 1e-10.
        for side, values in independent_summary["per_chain_event"].items():
            for clip, item in values.items():
                old_item=from_disk["summary"]["per_chain_event"][side][clip]
                for key in ("status_counts","margin_pass_count","denominator_all_registered_choices"):
                    if old_item[key] != item[key]:
                        raise ValueError("UNMATCHED_INDEPENDENT_SUMMARY")
    # Zero tolerance for decision and frame-index disagreement in main result.
    if exact_status!=720 or same_indices!=720 or rates>1e-7:
        raise ValueError("INDEPENDENT_ATAN2_DISAGREES_WITH_MAIN")
    result={
        "independent_implementation":"atan2(norm(cross(u- ,u+)),dot(u-,u+))",
        "original_source_sha256_reverified":True,
        "register_total_cells":720,"exact_matched_decisions":exact_status,
        "exact_matched_3_body_peak_frame_indices":same_indices,
        "maximum_absolute_peak_rate_numerical_difference_rad_per_s":rates,
        "contralateral_independent":independent_summary["bilateral_specificity_control"],
        "discrepancies":deviations,
        "human_true_contact_truth":False,"new_physx_rollouts":0,
        "statistical_p_value_or_subject_disjoint_claim":False,
    }
    from v142_native_pair_ledger import safe_write_once
    sha=safe_write_once(output_path,result)
    return result,sha


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    r,sha=review(a.input,a.output)
    print(json.dumps({"sha256":sha,**r},ensure_ascii=False,indent=2))
