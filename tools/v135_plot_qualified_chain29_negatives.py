#!/usr/bin/env python3
"""Read-only plotter: *synthetic* proposals vs *historical* physics, distinct plots."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sources(synthetic: Path, historical: Path, prefix: Path) -> tuple[str,str]:
    p = json.loads(synthetic.read_text())
    h = json.loads(historical.read_text())
    if p["version"] != "v0.1.135" or h["version"] != "v0.1.134":
        raise ValueError("SOURCE_VERSION_MISMATCH")
    if p["actual_new_physx_episodes"] or h["new_real_physx_episodes"]:
        raise ValueError("MISLABELED_NEW_PHYSICS")
    rows = p["trials"]
    if [x["multiplier"] for x in rows] != [0., 1., 2., 4., -1.]:
        raise ValueError("FROZEN_ABLATION_ORDER_CHANGED")

    labels = ["left_leg", "right_leg", "waist_trunk", "left_arm", "right_arm"]
    fig,ax=plt.subplots(figsize=(10,5))
    x=list(range(5))
    for row in rows:
        y=[row["group_excursion_rad"][k] for k in labels]
        ax.plot(x,y,marker="o",label=f"multiplier {row['multiplier']:+g}")
    ax.set_xticks(x,labels)
    ax.set_ylabel("Max joint angle excursion (rad)")
    ax.set_title("OFFLINE SYNTHETIC 29DoF shared-chain proposals (not human-trained)")
    ax.grid(alpha=.25)
    ax.legend(loc="upper left",fontsize=8,ncol=2)
    fig.tight_layout()
    first=Path(str(prefix)+"_synthetic_group_excursions.png")
    fig.savefig(first,dpi=130)
    plt.close(fig)

    names=["frozen_v32","cmu_10_02","cmu_10_05"]
    cases=[h["cases"][x] for x in names]
    fig,ax=plt.subplots(figsize=(9,5))
    peaks=[x["pre_0p4s_max_nonfoot_net_force_N"] for x in cases]
    bars=ax.bar(names,peaks)
    ax.axhline(20,linestyle="--",label="20N historic non-foot gate")
    ax.set_ylabel("Non-foot net-force peak, 0.4s before named event (N)")
    ax.set_title("HISTORICAL SHA-VERIFIED PhysX v54 (3 episodes; 0 new physics trials)")
    for bar,case in zip(bars,cases):
        ax.annotate(f"post-1.25s: {case['real_post_1p25s_frames']}/251 frames",
                    (bar.get_x()+bar.get_width()/2,max(bar.get_height(),.1)),
                    textcoords="offset points",xytext=(0,8),ha="center",fontsize=8)
    ax.set_ylim(0,max(peaks)*1.3+1)
    ax.legend()
    fig.tight_layout()
    second=Path(str(prefix)+"_historical_nonfoot_failure.png")
    fig.savefig(second,dpi=130)
    plt.close(fig)
    return str(first),str(second)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--synthetic",required=True,type=Path)
    ap.add_argument("--historical",required=True,type=Path)
    ap.add_argument("--output-prefix",required=True,type=Path)
    args=ap.parse_args()
    if any(Path(str(args.output_prefix)+s).exists() for s in
           ("_synthetic_group_excursions.png","_historical_nonfoot_failure.png")):
        raise FileExistsError("NEVER_OVERWRITE_EXISTING_RESEARCH_FIGURE")
    for path in plot_sources(args.synthetic,args.historical,args.output_prefix):
        print(path)
