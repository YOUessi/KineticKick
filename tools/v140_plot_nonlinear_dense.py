#!/usr/bin/env python3
"""Two local failure figures based on independent JSON audit, never a physics plot."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--independent",type=Path,required=True)
    p.add_argument("--stem",type=Path,required=True)
    a=p.parse_args()
    d=json.loads(a.independent.read_text())
    if d["origin"]!="INDEPENDENT_CPU_REVIEW_NO_NEW_PHYSICS" or d["new_real_physx_shots"]!=0:
        raise ValueError("REFUSE_NON_ATTESTED_FIGURE")
    for suffix in ("_passes.png","_veto.png"):
        if a.stem.with_name(a.stem.name+suffix).exists():
            raise FileExistsError("NO_OVERWRITE_FAILURE_FIGURES")
    targets=[row["target_mps"] for row in d["per_target"]]
    before=[row["parent_pass"] for row in d["per_target"]]
    after=[row["nonlinear_dense_pass"] for row in d["per_target"]]
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    x=range(len(targets))
    ax.bar([a-.19 for a in x],before,width=.38,label="v139 200Hz sampled QP")
    ax.bar([a+.19 for a in x],after,width=.38,label="v140 1ms + exact-pose re-audit")
    ax.set_xticks(list(x),[f"{t:g} m/s" for t in targets])
    ax.set_ylabel("CPU synthetic kinematic admissions / 15")
    ax.set_title("Original G1 29DoF: stricter synthetic re-audit (NOT physical shots)")
    ax.legend()
    fig.tight_layout();fig.savefig(str(a.stem)+"_passes.png",dpi=160);plt.close(fig)
    categories=[k for k,v in d["veto_counts"].items() if v]
    counts=[d["veto_counts"][k] for k in categories]
    fig,ax=plt.subplots(figsize=(7.5,3.6))
    ax.barh(categories or ["No additional veto"],counts or [0])
    ax.set_xlabel("Additional rejections among v139 admissions")
    ax.set_title("1ms sub-frame audit: NOT real PhysX / GRF")
    fig.tight_layout();fig.savefig(str(a.stem)+"_veto.png",dpi=160);plt.close(fig)
    print("WROTE",str(a.stem)+"_passes.png",str(a.stem)+"_veto.png")

if __name__=="__main__":
    main()
