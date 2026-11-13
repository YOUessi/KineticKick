#!/usr/bin/env python3
"""Render independently audited negative results; no invented PhysX evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot(a, stem:Path):
    d=json.loads(Path(a).read_text())
    if (d["origin"]!="INDEPENDENT_FROZEN_JSON_AUDIT_NOT_NEW_PHYSICS" or
        d["complete_cells"]!=45 or d["new_physx_rollouts"]!=0):
        raise ValueError("NOT_VERIFIED_CPU_GRID")
    widths=[.4,.8,1.2]
    velocities=[0.,1.,3.]
    grid=np.zeros((len(velocities),len(widths)),dtype=int)
    for row in d["per_target_width"]:
        i=velocities.index(row["target_ankle_origin_mps"])
        j=widths.index(row["halfwidth_s"])
        if row["attempts"]!=5:
            raise ValueError("GRID_DENOMINATOR_DRIFT")
        grid[i,j]=row["kinematic_admitted"]
    fig,ax=plt.subplots(figsize=(7,4.5))
    im=ax.imshow(grid,cmap="Greys",vmin=0,vmax=5)
    ax.set_xticks(range(3),[str(x)+"s" for x in widths])
    ax.set_yticks(range(3),[str(x)+"m/s" for x in velocities])
    ax.set_title("Original G1 29DoF: frozen CPU window QP (not PhysX)")
    ax.set_xlabel("Compact halfwidth")
    ax.set_ylabel("Synthetic ankle-origin relative x target")
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{grid[i,j]}/5",ha="center",va="center",color=("white" if grid[i,j]>2 else "black"))
    fig.colorbar(im,ax=ax,label="QP admitted / 5")
    fig.tight_layout()
    fig.savefig(str(stem)+"_qp_grid.png",dpi=180)
    plt.close(fig)
    # Report rejected statuses, keeping unresolved solver statuses distinct from infeasible.
    categories=["QP_PRIMAL_INFEASIBLE","QP_PRIMAL_INFEASIBLE_INACCURATE",
                "QP_MAXIMUM_ITERATIONS_REACHED","POSTSOLVE_HARD_CONSTRAINT_VIOLATION"]
    counts=[d["reasons"].get(k,0) for k in categories]
    fig,ax=plt.subplots(figsize=(9,4.2))
    bars=ax.barh(np.arange(len(categories)),counts)
    ax.set_yticks(np.arange(len(categories)),["Primal infeasible","Infeasible inaccurate",
                                              "Iterations exhausted","Postsolve constraint veto"])
    ax.set_xlabel("Frozen grid cells (all unsuccessful)")
    ax.set_title("Rejections are retained, not counted as successes")
    for bar,count in zip(bars,counts):
        ax.text(count+.1,bar.get_y()+bar.get_height()/2,str(count),va="center")
    ax.set_xlim(0,max(counts+[1])*1.25)
    fig.tight_layout()
    fig.savefig(str(stem)+"_qp_rejections.png",dpi=180)
    plt.close(fig)
    return {"grid":grid.tolist(),"categories":dict(zip(categories,counts))}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--independent",type=Path,required=True)
    p.add_argument("--stem",type=Path,required=True)
    args=p.parse_args()
    for suffix in ("_qp_grid.png","_qp_rejections.png"):
        if Path(str(args.stem)+suffix).exists():
            raise FileExistsError("REFUSE_OVERWRITE_PLOT_EVIDENCE")
    args.stem.parent.mkdir(parents=True,exist_ok=True)
    print(plot(args.independent,args.stem))
