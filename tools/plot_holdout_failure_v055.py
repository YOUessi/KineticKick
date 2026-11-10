#!/usr/bin/env python3
"""Reproduce v55 failed holdout evidence figures from SHA-pinned real PhysX.

Plots only. No synthetic dynamics, no contact manifold reconstruction, and no
overwriting existing media. Raw 30MB JSON stays ignored on Tang.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PROGRESS=ROOT/"reports/v055_pair_batch_progress.json"
OUTPUT_FORCE=ROOT/"media/v055_holdoutB_real_nonfoot_force.png"
OUTPUT_SUPPORT=ROOT/"media/v055_holdoutB_original_support_normal.png"


def verified_load(case_id):
    progress=json.loads(PROGRESS.read_text())
    case=next(x for x in progress["runs"] if x["id"]==case_id)
    path=ROOT/case["out"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=case["sha256"]:
        raise ValueError("Frozen real PhysX evidence SHA mismatch")
    return json.loads(path.read_text())


def main():
    for output in (OUTPUT_FORCE,OUTPUT_SUPPORT):
        if output.exists():
            raise FileExistsError("No overwriting figures: "+str(output))
    rows={}
    for policy in ("frozen_v32","cmu_10_02","cmu_10_05"):
        d=verified_load(f"holdoutB_seed1201_{policy}")
        tr=d["traces"][0]
        rows[policy]=tr
    fig,ax=plt.subplots(figsize=(9,4.3))
    for policy,tr in rows.items():
        ax.plot([r["time_s"] for r in tr],[r["nonfoot_contact_N"] for r in tr],
                label=policy,linewidth=1.6)
    ax.axvline(1.57,color="black",linestyle="--",alpha=.6,label="measured instep time")
    ax.axhline(20,color="gray",linestyle=":",alpha=.8,label="original nonfoot stop threshold")
    ax.set(xlabel="Original PhysX elapsed time (s)",ylabel="Body-level nonfoot net sensor force (N)",
           title="Real PhysX holdoutB: 1.64s nonfoot stop, not certified collision pair",
           xlim=(1.45,1.66))
    ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUTPUT_FORCE,dpi=150);plt.close(fig)
    tr=rows["frozen_v32"]
    fig,ax=plt.subplots(figsize=(9,4.3))
    t=[r["time_s"] for r in tr]
    for side in ("left","right"):
        ax.plot(t,[r["support_foot_force_N"][side] for r in tr],label=side+" foot",linewidth=1.2)
    ax.axvline(1.57,color="black",linestyle="--",alpha=.6,label="measured instep time")
    ax.set(xlabel="Original PhysX elapsed time (s)",ylabel="Foot-ground filtered normal force (N)",
           title="Real PhysX holdoutB (v32): body-weight support before early termination",
           xlim=(1.1,1.66))
    ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUTPUT_SUPPORT,dpi=150);plt.close(fig)
    print("V055_FIGURES",OUTPUT_FORCE,OUTPUT_SUPPORT)

if __name__=="__main__":main()
