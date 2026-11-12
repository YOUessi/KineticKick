#!/usr/bin/env python3
"""Render measured historical-run initial-state evidence; images stay local, not Git."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", required=True)
    p.add_argument("--prefix", required=True)
    a = p.parse_args()
    data = json.loads(Path(a.audit).read_text())
    rows = [x for x in data["runs"] if not x["observer"]]
    label = [x["scene"] + "/" + x["control"] for x in rows]
    x = np.arange(len(label))
    fig, ax = plt.subplots(figsize=(15, 5))
    speed = [r["measures"]["first_frame_root_xy_speed_mps"] for r in rows]
    ax.plot(x, speed, "o", label="True first 200Hz root XY speed (m/s)")
    ax.axhline(0.05, linestyle="--", linewidth=1.2,
               label="Frozen illustrative rest-like root speed bound")
    ax.set_xticks(x, label, rotation=70)
    ax.set_ylabel("Root XY speed (m/s)")
    ax.set_title("v123: all frozen independent original-G1 trials already moving at t=0.005s")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(a.prefix + "_first_frame_motion.png", dpi=160)
    plt.close(fig)

    # True physical trajectory; source file is SHA-certified by v123 audit.
    raw_path = Path("/home/you/projects/kinetickick-wt-v116/reports/v116_AG_v32_off.json")
    raw = json.loads(raw_path.read_text())
    r = raw["traces"][0]
    subset = [row for row in r if row["time_s"] <= 1.8]
    t = np.array([row["time_s"] for row in subset])
    root_speed = np.array([row["root_speed_xy"] for row in subset])
    left = np.array([row["support_foot_force_N"]["left"] for row in subset])
    right = np.array([row["support_foot_force_N"]["right"] for row in subset])
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t, root_speed, label="Measured G1 root XY speed")
    ax2 = ax.twinx()
    ax2.plot(t, left, label="Left foot normal-force proxy", linewidth=.75, alpha=.65)
    ax2.plot(t, right, label="Right foot normal-force proxy", linewidth=.75, alpha=.65)
    ax.set_xlim(0, 1.8)
    ax.axvline(0.005, linestyle=":", label="First physical sample")
    ax.axvline(1.405, linestyle="--", label="First measurable CMU-v32 motion difference")
    ax.axvline(raw["records"][0]["contact_time_s"], linestyle="--", label="Actual instep contact")
    ax.set_xlabel("Physical time (s)")
    ax.set_ylabel("Root XY speed (m/s)")
    ax2.set_ylabel("Foot normal-force proxy (N), not calibrated athlete GRF")
    ax.grid(alpha=.25)
    l1, t1 = ax.get_legend_handles_labels()
    l2, t2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, t1+t2, loc="upper right", fontsize=8)
    fig.suptitle("Source v116 AG/v32 OFF, real 200Hz PhysX: approach is measured; onset before initial sample is not")
    fig.tight_layout()
    fig.savefig(a.prefix + "_AG_measured_runup.png", dpi=160)
    plt.close(fig)
    print("WROTE", a.prefix + "_first_frame_motion.png",
          a.prefix + "_AG_measured_runup.png")


if __name__ == "__main__":
    main()
