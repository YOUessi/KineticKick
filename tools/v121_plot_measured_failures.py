#!/usr/bin/env python3
"""Figures strictly from the SHA-audited 29DoF CPU result; never inferred samples."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/v121_leftcensor_corrected_29dof.audit.json"


def main():
    data = json.loads(SOURCE.read_text())
    if data["original_PhysX_rollouts_REUSED_not_NEW"] != 29:
        raise ValueError("UNQUALIFIED_AUDIT_FIGURE_SOURCE")
    rows = data["independent_trials"]
    scenes = list(dict.fromkeys((r["source_version"], r["scene"]) for r in rows))
    controls = ("v32", "cmu02", "cmu05")
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    xlocations, heights, labels, edge = [], [], [], []
    for i, (ver, scene) in enumerate(scenes):
        for j, ctrl in enumerate(controls):
            candidates = [r for r in rows if (r["source_version"],r["scene"],r["control"]) == (ver,scene,ctrl)]
            if len(candidates) != 1:
                raise ValueError("MISSING_OR_DUPLICATE_29DOF_TRIAL")
            r = candidates[0]
            xlocations.append(i*3.7+j)
            heights.append(r["observed"]["full_29dof_qdot"]["maximum_ratio"])
            labels.append("PASS" if r["hard_gate_pass"] else "FAIL")
            edge.append(r["observed"]["full_29dof_qdot"]["first_hard_limit_violation_phase"] or "other")
    bars = ax.bar(xlocations, heights, alpha=.78)
    for bar, gate, phase in zip(bars, labels, edge):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02,
                gate + (("\n" + phase) if phase != "other" else ""),
                ha="center", va="bottom", fontsize=6, rotation=90)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.1, label="Original 29DoF qdot limit")
    ax.set_xticks([i*3.7+1 for i in range(len(scenes))],
                  [f"v{ver}/{scene}" for ver,scene in scenes])
    ax.set_ylim(0, max(1.65,max(heights)+.4))
    ax.set_ylabel("Max measured joint speed / original joint limit")
    ax.set_title("Historical original G1 PhysX, n=24 independent cells; 5 OFF/ON repeats excluded")
    ax.legend(loc="upper right")
    fig.tight_layout()
    target = ROOT / "reports/v121_original29_safety_failures.png"
    if target.exists():
        raise FileExistsError("WRITE_ONCE_FIGURE")
    fig.savefig(target, dpi=160);plt.close(fig)

    baselines = [r for r in rows if r["control"] == "v32"]
    if len(baselines) != 8:
        raise ValueError("WRONG_FROZEN_BASELINE_COUNT")
    names = ["waist", "right_hip", "right_knee", "right_ankle", "left_arm"]
    fig2, ax2 = plt.subplots(figsize=(11.3, 4.5))
    xs = np.arange(len(baselines))
    for g in names:
        centroids = [r["observed"]["group_estimated_motor_positive_power"][g]
                     ["positive_power_centroid_relative_impact_s"] for r in baselines]
        ax2.plot(xs, centroids, marker="o", label=g)
    ax2.set_xticks(xs, [r["scene"] for r in baselines])
    ax2.set_ylim(-.405,.025)
    ax2.set_xlabel("Frozen scene (baseline v32)")
    ax2.set_ylabel("Robot estimated-positive-power centroid relative to impact (s)")
    ax2.set_title("Descriptive robot motor-power centroids; no certified human transfer order")
    ax2.legend(ncol=3, fontsize=9)
    ax2.grid(alpha=.2)
    fig2.tight_layout()
    target2=ROOT/"reports/v121_robot_power_centroids.png"
    if target2.exists():
        raise FileExistsError("WRITE_ONCE_FIGURE")
    fig2.savefig(target2,dpi=160);plt.close(fig2)
    print("V121_FIGURES",str(target),str(target2),flush=True)


if __name__ == "__main__":
    main()
