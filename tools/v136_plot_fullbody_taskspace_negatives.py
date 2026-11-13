#!/usr/bin/env python3
"""Plot honest v136 CPU-only original-G1-inertial and task-space negatives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(data: dict, output: Path):
    rows = data["support_kinematics_rows"]
    factors = [str(r["multiplier"]) for r in rows]
    xpos = np.arange(len(rows))
    plt.figure(figsize=(9, 4.6))
    plt.plot(xpos,[r["fixed_root_left_ankle_origin_travel_max_m"] for r in rows],
             marker="o",label="Fixed pelvis: left-ankle displacement")
    plt.plot(xpos,[r["fixed_root_inertial_com_travel_max_m"] for r in rows],
             marker="s",label="Fixed pelvis: URDF mass COM displacement")
    plt.plot(xpos,[r["left_ankle_translation_locked_inertial_com_travel_max_m"] for r in rows],
             marker="^",label="Left-ankle-origin locked: mass COM displacement")
    plt.xticks(xpos,factors)
    plt.xlabel("Frozen synthetic multiplier (not a physical control trial)")
    plt.ylabel("Maximum root-frame displacement under stated hypothesis (m)")
    plt.title("Original G1 29DoF kinematic support compensation: CPU only")
    plt.grid(alpha=.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output/"v136_original_g1_support_com_negative.png",dpi=160)
    plt.close()

    tasks=data["frozen_taskspace_rows"]
    plt.figure(figsize=(9,4.6))
    for scale in [0,1,2,4,-1]:
        subset=[r for r in tasks if r["multiplier"]==scale]
        plt.plot([r["target_ankle_origin_root_plus_x_mps"] for r in subset],
                 [r["swing_vector_residual_mps"] for r in subset],
                 marker="o",label=f"synthetic x{scale}")
    plt.axhline(.2,linestyle="--",linewidth=1,label="Predeclared warning threshold")
    plt.xlabel("Requested right ankle-link-origin +X velocity (m/s), NOT ball speed")
    plt.ylabel("Task-space root-frame vector residual (m/s)")
    plt.title("Original 29DoF bounded least-squares task residual: no real kick")
    plt.grid(alpha=.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output/"v136_taskspace_velocity_negative.png",dpi=160)
    plt.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    if not args.out_dir.is_dir():
        raise ValueError("EXISTING_LOCAL_OUTPUT_DIR_REQUIRED")
    data=json.loads(args.input.read_text())
    if (data["version"]!="v0.1.136" or data["new_true_physx_rollouts"]!=0
            or len(data["support_kinematics_rows"])!=5
            or len(data["frozen_taskspace_rows"])!=20):
        raise ValueError("NOT_FROZEN_V136_DATA")
    plot(data,args.out_dir)


if __name__=="__main__":
    main()
