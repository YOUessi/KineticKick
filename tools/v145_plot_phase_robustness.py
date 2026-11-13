#!/usr/bin/env python3
"""Two original-source v145 diagnostic plots, write-once local only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_once(fig, path):
    p=Path(path)
    if p.is_symlink() or p.exists():
        raise FileExistsError("DO_NOT_OVERWRITE_EVIDENCE:"+str(path))
    with p.open("xb") as f:
        fig.savefig(f,format="png",dpi=150,bbox_inches="tight")
    plt.close(fig)


def plot(data_path, chart1, chart2):
    x=json.loads(Path(data_path).read_text())
    if (x["version"]!="v0.1.145" or len(x["raw_grid"])!=720
            or x["new_physical_rollouts"]!=0):
        raise ValueError("BAD_PHASE_GRID_SOURCE")
    raw=x["raw_grid"]
    subject_results=x["summary"]["per_chain_event"]
    fig,ax=plt.subplots(figsize=(9,5))
    locations=np.arange(2)
    width=.35
    for side,offset in (("right",-width/2),("left",width/2)):
        value=[subject_results[side][clip]["margin_pass_count"] for clip in ("10_02","10_05")]
        ax.bar(locations+offset,value,width,label=side.capitalize()+" body-side chain")
        for pos,n in zip(locations+offset,value):
            ax.text(pos,n+3,f"{n}/180",ha="center",fontsize=10)
    ax.set_xticks(locations,["CMU 10_02","CMU 10_05"])
    ax.set_ylim(0,190)
    ax.set_ylabel("Frozen sensitivity choices meeting >=2-frame ordered peaks")
    ax.set_title("Unverified ball-marker launch, NOT true instep impact\nBoth selected clips are the SAME subject; post-hoc analysis")
    ax.legend()
    ax.grid(axis="y",alpha=.2)
    save_once(fig,chart1)
    fig,ax=plt.subplots(figsize=(9,5))
    offsets=(-12,-6,0,6,12)
    for clip in ("10_02","10_05"):
        values=[]
        for shift in offsets:
            rows=[r for r in raw if r["clip"]==clip and r["chain"]=="right"
                  and r["anchor_shift_frames"]==shift]
            values.append(sum(r["status"]=="ORDER_MARGIN_PASS" for r in rows)/len(rows))
        ax.plot([z/120 for z in offsets],values,marker="o",linewidth=2,label=clip)
        for pos,v in zip([z/120 for z in offsets],values):
            ax.annotate(f"{v:.0%}",(pos,v),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8)
    ax.set_ylim(-.1,1.1)
    ax.set_xlabel("Weak ball marker launch anchor shift (s), not true contact")
    ax.set_ylabel("Right-side order fraction over 36 choices per anchor shift")
    ax.set_title("Original-source right pelvis→thigh→shank ordering is anchor-sensitive\nNot a causal power transfer or generalization estimate")
    ax.legend()
    ax.grid(alpha=.2)
    save_once(fig,chart2)
    return [str(chart1),str(chart2)]


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--out-order",required=True,type=Path)
    p.add_argument("--out-anchor",required=True,type=Path)
    a=p.parse_args()
    print(json.dumps({"plots":plot(a.input,a.out_order,a.out_anchor)}))
