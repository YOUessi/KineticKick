#!/usr/bin/env python3
"""Render v127 SHA-audited original 200Hz evidence; no artificial trajectories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(audit: Path, outdir: Path):
    data=json.loads(audit.read_text())
    if (data["version"]!="v0.1.127" or data["historical_true_physx_episodes_reaudited"]!=29
            or not data["observer_geometry_exact_parity"]
            or data["native_physx_actor_collider_manifold_status"]!="UNKNOWN"):
        raise ValueError("UNQUALIFIED_GEOMETRY_SOURCE")
    records=sorted((r for r in data["episodes"] if not r["observer_ON_repeat"]),
                   key=lambda r:(r["scene"],{"v32":0,"cmu02":1,"cmu05":2}[r["control"]]))
    if len(records)!=24:
        raise ValueError("NO_INDEPENDENT_COHORT")
    keys=list(records[0]["full14s_min_clearance_by_pair_m"])
    labels=["L wrist / L hip","R wrist / R hip",
            "L wrist / R hip","R wrist / L hip"]
    x=np.asarray([[r["full14s_min_clearance_by_pair_m"][k] for k in keys]
                  for r in records])
    fig,ax=plt.subplots(figsize=(10,9))
    im=ax.imshow(x,aspect="auto",cmap="viridis",vmin=0,vmax=.28)
    ax.set_yticks(range(len(records)),[r["scene"]+"/"+r["control"]+
                        (" PASS" if r["original_strict_pass"] else " FAIL")
                        for r in records],fontsize=8)
    ax.set_xticks(range(4),labels,rotation=15)
    for i in range(24):
        for j in range(4):
            ax.text(j,i,f"{x[i,j]*100:.1f}",ha="center",va="center",fontsize=7,
                    color="white" if x[i,j]<.12 else "black",
                    fontweight="bold" if x[i,j]<.02 else "normal")
    cb=fig.colorbar(im,ax=ax,shrink=.7)
    cb.set_label("Minimum capsule lower-bound clearance (m) over 14s")
    ax.set_title("Original G1 physical 200Hz trajectories: URDF geometry only\n"
                 "Each value in cm; positive excludes those URDF cylinders touching")
    fig.tight_layout()
    first=outdir/"v127_physics_full14s_geometry_lower_bound.png"
    fig.savefig(first,dpi=165)
    plt.close(fig)

    phases=["precontact_400ms","postcontact_1250ms","late_recovery"]
    fractions=np.asarray([[np.mean([
        r["per_phase"][ph]["pair_geometry"][key]["under_20mm_alert_frame_fraction"]
        for r in records]) for key in keys] for ph in phases])
    fig,ax=plt.subplots(figsize=(10,5))
    for i,phase in enumerate(phases):
        shift=(i-1)*.23
        ax.bar(np.arange(4)+shift,fractions[i],width=.23,label=phase)
    ax.set_xticks(range(4),labels,rotation=10)
    ax.set_ylabel("Mean fraction of true 200Hz frames with capsule gap ≤ 2 cm")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title("Proximity in frozen contact-relative windows (24 paired conditions)\n"
                 "Alert is not a PhysX collision-pair event; no GRF inference")
    fig.tight_layout()
    second=outdir/"v127_physics_phase_proximity_alerts.png"
    fig.savefig(second,dpi=165)
    plt.close(fig)
    return [str(first),str(second)]


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--audit",type=Path,required=True)
    a.add_argument("--outdir",type=Path,required=True)
    x=a.parse_args()
    for out in run(x.audit,x.outdir):
        print(out)
