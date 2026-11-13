#!/usr/bin/env python3
"""Plot only SHA-reviewed original CMU marker timing and original-v53 failed gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--stem", type=Path, required=True)
    args = ap.parse_args()
    a = json.loads(args.audit.read_text())
    b = json.loads(args.independent.read_text())
    if (a["new_real_PhysX_episodes"] != 0 or b["new_physx"] != 0 or
            b["max_abs_segment_rate_reconstruction_difference_rad_per_s"] > 1e-5):
        raise ValueError("PLOT_INPUT_NOT_INDEPENDENTLY_AUDITED")

    names = ("pelvis", "right_thigh", "right_shank", "trunk", "right_foot",
             "left_forearm")
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.34
    for idx, clip in enumerate(("10_02", "10_05")):
        seg = a["human_real_C3D_marker_temporal_pilot"]["per_clip"][clip]["segments"]
        values = [seg[k]["pre_ball_marker_launch_descriptor"][
            "peak_pre_ball_marker_onset_s"] for k in names]
        ax.bar(x + (idx - 0.5) * width, values, width, label=clip)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("Time BEFORE marker-only fast-ball onset (s)")
    ax.set_title("Original real 120Hz CMU markers | unverified impact / same subject 10")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    image = args.stem.with_name(args.stem.name + "_real_human_segment_lags.png")
    if image.exists():
        raise FileExistsError("DO_NOT_OVERWRITE_PLOT")
    fig.savefig(image, dpi=155)
    plt.close(fig)

    gates = ("legal_right_instep_contact", "outgoing_ball_speed_at_least_3_5",
             "entire_ball_goal", "genuine_runup_and_shot", "recovered_and_stable",
             "actual_qdot_under_original_limits")
    failure = a["historical_v53_strict_physics"]["failed_gates_by_policy"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pols = ("frozen_v32", "cmu_10_02", "cmu_10_05")
    x = np.arange(len(gates))
    for ix, pol in enumerate(pols):
        ax.bar(x + (ix - 1) * .26,
               [failure[pol].get(k, 0) for k in gates], .26,
               label=pol)
    ax.set_xticks(x, [
        "Instep", "Ball speed", "Goal", "Runup + shot", "Recovery", "29 qdot"
    ])
    ax.set_ylim(0, 4.2)
    ax.set_ylabel("Original 4 ball-condition trials failing gate")
    ax.set_title("12 original G1 PhysX shots | strict PASS = 0/4 for all 3 controllers")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fail_png = args.stem.with_name(args.stem.name + "_historical_physx_gate_failures.png")
    if fail_png.exists():
        raise FileExistsError("DO_NOT_OVERWRITE_PLOT")
    fig.savefig(fail_png, dpi=155)
    plt.close(fig)
    print(json.dumps({"source_human_plot": str(image),
                      "historical_original_physx_fail_plot": str(fail_png),
                      "new_physx": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
