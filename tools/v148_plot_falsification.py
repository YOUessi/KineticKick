#!/usr/bin/env python3
"""Plot v148 actual C3D analysis-choice sensitivity, never a population CI."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create(source, output):
    if Path(output).exists():
        raise FileExistsError(output)
    r=json.loads(Path(source).read_text())
    b=r["analysis"]["by_clip_and_anatomical_side"]
    labels=["-100","-50","0","+50","+100"]
    shifts=("-12","-6","0","6","12")
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(5,dtype=float)
    for idx,(clip,side) in enumerate((("10_02","right"),("10_05","right"),
                                     ("10_02","left"),("10_05","left"))):
        metric=b[side+"_"+clip]["anchor_shift_control"]
        y=[metric[s]["pass_fraction"]*100 for s in shifts]
        ax.plot(x,y,marker=("o" if side=="right" else "x"),linewidth=2,
                label=f"CMU{clip} {side}")
    ax.axhline(60,linestyle="--",linewidth=1,color="gray",
               label="frozen per-shift 60% gate")
    ax.set_xticks(x, labels)
    ax.set_ylim(-3,103)
    ax.set_xlabel("Offset of UNCALIBRATED ball-launch proxy (ms)")
    ax.set_ylabel("% of 36 correlated analysis choices passing")
    ax.set_title("Original 120 Hz CMU: anatomical phase-order proxy vs anchor shift")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output,dpi=160)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    create(a.source,a.output)
    print(a.output)
