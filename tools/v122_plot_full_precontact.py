#!/usr/bin/env python3
"""Render only SHA-certified historical v122 robot physics, never pseudo-human force."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["waist", "right_hip", "right_knee", "right_ankle"]
DISPLAY = ["腰部", "右髋", "右膝", "右踝"]


def collect(doc):
    trials = doc["independent_trials"]
    if (doc["independent_conditions"] != 24 or len(trials) != 24
            or doc["new_PhysX_runs"] != 0):
        raise ValueError("UNEXPECTED_HISTORICAL_COHORT")
    matrix20, matrix40, fractions, support = [], [], [], []
    for row in trials:
        a = row["analysis"]
        g = a["estimated_robot_motor_power_groups"]
        matrix20.append([g[k]["full_primary20"]["onset_s"] for k in GROUPS])
        matrix40.append([g[k]["full_sensitivity40"]["onset_s"] for k in GROUPS])
        fractions.append([
            g[k]["early_pre400_work_J"] / g[k]["full_work_J"]
            if g[k]["full_work_J"] > 1e-9 else np.nan for k in GROUPS])
        support.append(a["left_support_full_precontact"]["sustained_50N_onset_s"])
    return np.asarray(matrix20, float), np.asarray(matrix40, float), np.asarray(fractions,float), np.asarray(support,float)


def plots(doc, dest):
    a, b, fractions, support = collect(doc)
    # Figure 1: observed onset since simulation start, not athlete excitation.
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    x = np.arange(len(GROUPS), dtype=float)
    for i in range(len(a)):
        ax.plot(x-0.12, a[i], "o", ms=3, alpha=.21, color="C0")
        ax.plot(x+0.12, b[i], "x", ms=4, alpha=.21, color="C1")
    ax.plot(x-.12, np.median(a,axis=0), "o-", color="C0", label="20% threshold median")
    ax.plot(x+.12, np.median(b,axis=0), "x--", color="C1", label="40% fixed sensitivity median")
    ax.axhline(float(np.median(support)), color="C2", linestyle=":",
               label="left support 50N onset median")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel("Observed onset after simulation start (s)")
    ax.set_title("Historical G1 200Hz: robot motor positive-power onsets (n=24)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(dest / "v122_measured_full_precontact_onsets.png", dpi=145)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.boxplot([fractions[:, i] for i in range(len(GROUPS))],
               tick_labels=GROUPS, showmeans=True)
    ax.set_ylim(-.025, 1.025)
    ax.set_ylabel("Fraction of precontact robot motor work before last 0.4 s")
    ax.set_title("SHA-verified historical G1 motor work, not human athlete GRF")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(dest / "v122_robot_early_motor_work_fraction.png", dpi=145)
    plt.close(fig)
    return {
        "onset_20_medians_s": [float(x) for x in np.median(a, axis=0)],
        "onset_40_medians_s": [float(x) for x in np.median(b, axis=0)],
        "left_support_50N_median_s": float(np.median(support)),
        "early_motor_work_fraction_medians": [float(x) for x in np.median(fractions, axis=0)],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audit",type=Path,
                   default=ROOT/"reports/v122_complete_precontact_verified.audit.json")
    p.add_argument("--out-dir",type=Path,default=ROOT/"reports")
    args=p.parse_args()
    result=plots(json.loads(args.audit.read_text()), args.out_dir)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
