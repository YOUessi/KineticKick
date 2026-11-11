#!/usr/bin/env python3
"""Separate real PhysX measured-censoring and support-planning plots.

Use already-frozen original nine 200Hz G1 traces, no invented post-termination
trajectories. These are not proof of learned human-style force transfer.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"reports/v072_measured_200hz_wholebody_coupling.json"
MEDIA=ROOT/"media"
def run():
    d=json.loads(SRC.read_text())
    cases=d["cases"]
    labels=[c["filename"].split("new_stance_",1)[1].removesuffix(".json") for c in cases]
    if len(cases)!=9:raise ValueError("Only original frozen nine CUDA G1 runs")

    MEDIA.mkdir(exist_ok=True)
    # Plot only observed episodes: early stops remain at 1.64 or 3.70, no 14s fill.
    fig,ax=plt.subplots(figsize=(11,5))
    ends=[c["actual_end_s"] for c in cases]
    ax.barh(labels,ends)
    ax.axvline(14.,linestyle="--",label="14 s protocol horizon")
    for i,(end,c) in enumerate(zip(ends,cases)):
        ax.text(end+.05,i,f"{end:.3f}s" + (" CENSORED" if c["physical_nonfoot_early_censored"] else ""),va="center",fontsize=8)
    ax.set_xlim(0,16.5);ax.set_xlabel("Observed original GPU G1 physical time (s)")
    ax.set_title("v72 forensic — nine historical G1 PhysX episodes; nonfoot early termination not imputed")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out=MEDIA/"v072_original_g1_early_termination.png"
    if out.exists():raise FileExistsError(out)
    fig.savefig(out,dpi=150);plt.close(fig)

    complete=[(name,c) for name,c in zip(labels,cases) if c["pre_impact"] is not None]
    fig,ax=plt.subplots(figsize=(11,5))
    bars=ax.bar([name for name,_ in complete],
                [c["pre_impact"]["left_support_fraction_mean"] for _,c in complete])
    for b,(_,c) in zip(bars,complete):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.02,
                f'{c["pre_impact"]["measured_rms_qdot_rad_s"]["pelvis_waist"]:.2f} rad/s waist RMS',
                rotation=90,ha="center",va="bottom",fontsize=8)
    ax.set_ylim(0,1.12);ax.set_ylabel("Mean left-foot normal support share, last 0.4s pre-impact")
    ax.set_title("v72 forensic — observed physical support, not a human GRF or contact polygon")
    ax.tick_params(axis="x",labelrotation=20);fig.tight_layout()
    out2=MEDIA/"v072_original_support_and_waist_coupling.png"
    if out2.exists():raise FileExistsError(out2)
    fig.savefig(out2,dpi=150);plt.close(fig)
    print("V072_PLOTS",out,out2)
if __name__=="__main__":run()
