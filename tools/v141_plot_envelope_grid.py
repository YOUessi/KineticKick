#!/usr/bin/env python3
"""Two CPU charts strictly from independently reviewed v141 synthetic trials."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import v141_independent_envelope_audit as audit

def plots(raw, reviewed, stem):
    validated=audit.audit(raw)
    if validated!=reviewed:
        raise ValueError("INDEPENDENT_REVIEW_AND_TRIALS_DISAGREE")
    stem=Path(stem)
    targets=(0.,1.,3.)
    baseline=np.array([sum(r["parent_solved"] for r in raw["rows"]
                           if r["target_mps"]==v) for v in targets])
    dense=np.array([sum(r["v140_pass"] for r in raw["rows"]
                           if r["target_mps"]==v) for v in targets])
    envelope=np.array([sum(r["v141_admitted"] for r in raw["rows"]
                           if r["target_mps"]==v) for v in targets])
    x=np.arange(len(targets))
    fig,ax=plt.subplots(figsize=(8,4))
    width=.24
    ax.bar(x-width,baseline,width,label="v139 200Hz QP")
    ax.bar(x,dense,width,label="v140 1ms + exact center")
    ax.bar(x+width,envelope,width,label="v141 all-time polynomial hull")
    ax.set_xticks(x,[f"{t:g} m/s" for t in targets])
    ax.set_ylim(0,16)
    ax.set_ylabel("Kinematically admitted / 15, NOT physical shots")
    ax.set_xlabel("Synthetic relative ankle-origin target")
    ax.set_title("Frozen original29 QP: strict re-audits do not add shot success")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out1=stem.with_name(stem.name+"_admissions.png")
    if out1.exists():raise FileExistsError("NO_PLOT_OVERWRITE:"+str(out1))
    fig.savefig(out1,dpi=140);plt.close(fig)

    # 5 amplitudes x 9 frozen target/width columns; deliberately distinguish
    # unresolved QP from numeric failing envelope.
    mult=(0.,1.,2.,4.,-1.)
    col=[(t,w) for t in targets for w in (.4,.8,1.2)]
    categories={"PARENT_QP_REJECTED":0,"FAIL":1,"UNRESOLVED":2,"PASS":3}
    mat=np.full((5,9),-1,int)
    for r in raw["rows"]:
        i=mult.index(r["multiplier"])
        j=col.index((r["target_mps"],r["halfwidth_s"]))
        mat[i,j]=categories[r["numeric_envelope_state"]]
    if np.any(mat<0):raise ValueError("INCOMPLETE_45_GRID")
    fig,ax=plt.subplots(figsize=(11,4))
    from matplotlib.colors import ListedColormap
    colors=plt.get_cmap("tab10")
    cmap=ListedColormap([colors(7),colors(3),colors(1),colors(2)])
    im=ax.imshow(mat,aspect="auto",cmap=cmap,vmin=-.5,vmax=3.5)
    ax.set_xticks(range(9),[f"{t:g}m/s\n{w:g}s" for t,w in col])
    ax.set_yticks(range(5),[str(m) for m in mult])
    ax.set_ylabel("Synthetic full-body amplitude")
    ax.set_title("v141 numeric envelope state by frozen cell (NOT PhysX)")
    for (i,j),value in np.ndenumerate(mat):
        ax.text(j,i,["QP","FAIL","?","PASS"][value],ha="center",va="center",fontsize=8)
    fig.tight_layout()
    out2=stem.with_name(stem.name+"_cells.png")
    if out2.exists():raise FileExistsError("NO_PLOT_OVERWRITE:"+str(out2))
    fig.savefig(out2,dpi=140);plt.close(fig)
    return [str(out1),str(out2)]

def main():
    p=argparse.ArgumentParser();p.add_argument("--trials",type=Path,required=True)
    p.add_argument("--independent",type=Path,required=True)
    p.add_argument("--stem",type=Path,required=True)
    a=p.parse_args()
    x=plots(json.loads(a.trials.read_text()),json.loads(a.independent.read_text()),a.stem)
    for path in x:print("SAVED",path)
if __name__=="__main__":main()
