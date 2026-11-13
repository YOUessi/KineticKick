#!/usr/bin/env python3
"""Plots from SHA-frozen v138 audit summaries, no new simulation or fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make(audit: Path, stem: Path):
    obj=json.loads(audit.read_text())
    if obj["version"]!="v0.1.138" or obj["protocol_sha"]!="c032c6f2de72ef0ace3d4e00834dd265164b98f0":
        raise ValueError("NOT_V138_FROZEN_AUDIT")
    if len(obj["cases"])!=15 or obj["summary_counts"]["temporal_attempts"]!=45:
        raise ValueError("INCOMPLETE_FROZEN_45")
    rows=[dict(mult=c["multiplier"],target=c["target_right_ankle_link_origin_x_mps"],**s)
          for c in obj["cases"] for s in c["temporal"]]
    for suffix in ("_analytic_accel.png","_joint_and_collision_negatives.png"):
        if Path(str(stem)+suffix).exists():
            raise FileExistsError("WOULD_OVERWRITE_EXISTING_RESEARCH_FIGURE")
    widths=[.1,.2,.4]
    targets=[0.,1.,3.]
    fig,ax=plt.subplots(figsize=(8,5))
    for target in targets:
        means=[np.mean([r["max_absolute_analytic_qddot_rad_s2"] for r in rows
                        if r["target"]==target and r["halfwidth_s"]==width])
               for width in widths]
        ax.plot(widths,means,marker="o",label=f"ankle-origin x target={target:g}m/s")
    ax.axhline(40.,linestyle="--",label="40rad/s² descriptive screen")
    ax.set(xlabel="Compact C2 halfwidth (s)",ylabel="Mean over five synthetic seeds of peak |qddot| (rad/s²)",
           title="v138: genuine G1 URDF limits, synthetic joint motion (not PhysX)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(stem)+"_analytic_accel.png",dpi=160)
    plt.close(fig)
    values=np.zeros((3,3))
    for i,w in enumerate(widths):
        for j,t in enumerate(targets):
            subset=[r for r in rows if r["halfwidth_s"]==w and r["target"]==t]
            if len(subset)!=5:raise ValueError("FROZEN_CELL_INCOMPLETE")
            values[i,j]=sum(int(r["position_limit_violating_200hz_frames"]>0
                                or r["descriptive_accel_violating_200hz_frames"]>0
                                or r["urdf_four_pair_warning_frames"]>0) for r in subset)
    fig,ax=plt.subplots(figsize=(7,4.7))
    im=ax.imshow(values,vmin=0,vmax=5,cmap="Greys")
    ax.set_xticks(range(3),[str(t) for t in targets])
    ax.set_yticks(range(3),[str(w) for w in widths])
    ax.set(xlabel="Relative right-ankle link-origin x target (m/s)",
           ylabel="C2 halfwidth (s)",title="Frozen CPU negative screen counts / 5 synthetic seeds")
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{int(values[i,j])}/5",ha="center",va="center",
                    color="white" if values[i,j]>2.5 else "black")
    fig.colorbar(im,ax=ax,label="Any position/acceleration/proximity warning")
    fig.tight_layout()
    fig.savefig(str(stem)+"_joint_and_collision_negatives.png",dpi=160)
    plt.close(fig)
    return values.tolist()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--stem",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps({"failure_count_grid":make(a.audit,a.stem)}))


if __name__=="__main__":
    main()
