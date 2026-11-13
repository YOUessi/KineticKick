#!/usr/bin/env python3
"""Non-evidentiary plots of v137 frozen CPU LP reachability. Local PNG only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(data: dict, out: Path):
    if data["version"]!="v0.1.137" or len(data["cases"])!=60:
        raise ValueError("NOT_COMPLETE_FROZEN_V137")
    out.mkdir(parents=True,exist_ok=True)
    rows=data["cases"]
    x=[0.,1.,2.,4.,-1.]
    fig,ax=plt.subplots(figsize=(10,5))
    for com,arm,label in [
        (None,None,"free 29DoF box"),
        (None,.25,"arms within 0.25 rad/s of synthetic ref"),
        (.1,None,"COM relative speed <= 0.1 m/s"),
        (.1,.25,"COM + arm bound")]:
        y=[next(r["max_attainable_relative_x_mps"] for r in rows
                if r["synthetic_time_s"]==2.95 and r["multiplier"]==m
                and r["com_velocity_cap_mps"]==com
                and r["arm_joint_velocity_deviation_cap_rad_s"]==arm) for m in x]
        ax.plot(range(5),y,marker="o",label=label)
    ax.axhline(3,linestyle="--",linewidth=1,label="3 m/s artificial ankle-origin target")
    ax.set_xticks(range(5),[str(m) for m in x])
    ax.set_xlabel("Synthetic v135 wholebody amplitude multiplier")
    ax.set_ylabel("LP mathematical max right/left ankle origin +x relative speed (m/s)")
    ax.set_title("v137 ORIGINAL G1 29DoF kinematic LP only — not a kick")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"v137_hard_relative_ankle_envelope.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    z=[t for r in rows for t in r["targets"] if t["target_ankle_origin_relative_x_mps"]==3]
    axes[0].hist([v["max_qdot_limit_ratio"] for v in z],bins=np.linspace(0,1.05,22))
    axes[0].set_xlabel("Max original G1 joint speed / URDF limit")
    axes[0].set_ylabel("Count of 3m/s LP feasibility witnesses")
    axes[1].hist([v["hypothetical_5ms_qddot_rad_s2"] for v in z],bins=20)
    axes[1].set_xlabel("Hypothetical one-step |qdot-LP - qdot-ref|/5ms (rad/s²)")
    axes[1].set_ylabel("Count")
    fig.suptitle("LP feasibility is NOT a smooth torque-safe whole-body action")
    fig.tight_layout();fig.savefig(out/"v137_lp_extreme_witness_negative.png",dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    plot(json.loads(a.input.read_text()),a.out_dir)
    print("created",a.out_dir/"v137_hard_relative_ankle_envelope.png",
          a.out_dir/"v137_lp_extreme_witness_negative.png")


if __name__=="__main__":
    main()
