#!/usr/bin/env python3
"""SHA-verified original 29DoF G1 physics qdot safety plot; non-GPU."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from v116_qdot_causality import sha256, physical_arrays

ROOT=Path(__file__).resolve().parents[1]

def plots(root=ROOT):
    batch=json.loads((root/"reports/v116_original_g1_batch.audit.json").read_text())
    if batch["version"]!="v0.1.116":raise ValueError("WRONG_PHYSICAL_BATCH")
    generated=[]
    for scene in ("AG","AH","AI"):
        fig,ax=plt.subplots(figsize=(9,4.4))
        n=0
        for r in batch["runs"]:
            if r["scene"]!=scene or r["observer"] or not r["certified_original_G1_cuda_200Hz"]:continue
            p=root/"reports"/("v116_"+r["name"]+".json")
            if sha256(p)!=r["raw_sha256"]:raise ValueError("SOURCE_SHA_DRIFT")
            d=physical_arrays(json.loads(p.read_text()))
            ratio=np.max(np.abs(d["v"])/d["lim"][None,:],axis=1)
            ax.plot(d["t"],ratio,label=r["control"]+" "+("PASS" if r["hard_gates"]["hard_gate_pass"] else "FAIL"))
            ax.axvline(d["contact"],color="gray",alpha=0.2,linewidth=1)
            n+=1
        if not n:
            plt.close(fig);continue
        ax.axhline(1,color="black",linestyle="--",linewidth=1.25,label="original 29-joint qdot limit")
        ax.set_xlabel("Actual PhysX time (s), 200Hz");ax.set_ylabel("Peak |measured joint qdot| / original limit")
        ax.set_title(f"{scene}: original free-root G1 all-29-joint speed safety, SHA verified")
        ax.legend(fontsize=8)
        ax.set_xlim(0,14);ax.grid(alpha=0.25)
        fig.tight_layout()
        path=root/"reports"/("v116_"+scene+"_actual_29qdot.png")
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=160);plt.close(fig)
        generated.append(path.name)
    return generated

if __name__=="__main__":print("WRITE_ONCE_SHA_VERIFIED_FAILURE_PLOTS",plots())
