#!/usr/bin/env python3
"""Physical failure plots from SHA-certified PhysX and measured C3D sources only."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ezc3d

from v113_phase_biomechanics import load_physical_arrays
from v117_causal_safety_statistics import ROOT,sha
from v117_human_marker_temporal_eligibility import REL

def g1_failures(root=ROOT):
    batch=json.loads((root/"reports/v117_original_g1_batch.audit.json").read_text())
    paths=[]
    for scene in ("AJ","AK","AL"):
        items=[x for x in batch["runs"] if x["scene"]==scene
               and not x["observer"] and x["certified_original_G1_cuda_200Hz"]]
        if not items:continue
        fig,axs=plt.subplots(3,1,figsize=(10,10),sharex=True)
        for row in items:
            p=root/"reports"/("v117_"+row["name"]+".json")
            if sha(p)!=row["raw_sha256"]:raise ValueError("PLOT_SOURCE_SHA_MISMATCH")
            raw=json.loads(p.read_text())
            raw["joint_velocity_limits"]=np.asarray(raw["joint_velocity_limits"],float).reshape(-1).tolist()
            a=load_physical_arrays(raw)
            axs[0].plot(a["t"],np.max(np.abs(a["vel"])/a["lim"],axis=1),linewidth=.9,
                        label=row["control"]+(" PASS" if row["hard_gates"]["hard_gate_pass"] else " FAIL"))
            axs[1].plot(a["t"],a["support"][:,0],alpha=.75,linewidth=.8,
                        label=row["control"]+" left foot force proxy")
            axs[1].plot(a["t"],a["support"][:,1],alpha=.45,linewidth=.8,linestyle="--",
                        label=row["control"]+" right foot force proxy")
            axs[2].plot(a["t"],a["com"][:,2],linewidth=.9,label=row["control"]+" COM z")
            for ax in axs:ax.axvline(a["impact"],color="gray",linestyle=":",linewidth=.5)
        axs[0].axhline(1,color="crimson",linestyle="--",label="original qdot hard limit")
        axs[0].set_ylabel("Max |qdot| / original 29 limits")
        axs[1].set_ylabel("Foot support net-force proxy (N)")
        axs[2].set_ylabel("COM height (m)")
        axs[2].set_xlabel("Original measured PhysX time (s)")
        for ax in axs:
            ax.grid(alpha=.25)
            ax.legend(loc="best",fontsize=7,ncol=2)
        fig.suptitle("G1 "+scene+": true 200Hz old controller / source SHA verified\nNot elite human kicking and not native collider manifold")
        fig.tight_layout()
        path=root/"reports"/("v117_"+scene+"_actual_qdot_support_com.png")
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=135)
        plt.close(fig)
        paths.append(str(path))
    return paths

def human_marker_figure(root=ROOT):
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=False)
    for i,(subj,stem) in enumerate((("10","10_02"),("10","10_05"),("11","11_01"))):
        path=root/REL/subj/(stem+".c3d")
        c=ezc3d.c3d(str(path))
        labels=c["parameters"]["POINT"]["LABELS"]["value"]
        pts=np.asarray(c["data"]["points"][:3],float)
        for suffix,label in (("RTOE","right toe"),("LTOE","left toe"),("LWRA","left wrist"),("RWRA","right wrist")):
            m=[j for j,s in enumerate(labels) if s.split(":")[-1]==suffix]
            if len(m)!=1:raise ValueError("AMBIGUOUS_REAL_HUMAN_MARKER")
            v=np.linalg.norm(np.diff(pts[:,m[0],:].T/1000,axis=0)*120,axis=1)
            axs[i].plot((np.arange(len(v))+1)/120,v,label=label,linewidth=1)
        axs[i].set_title("CMU "+stem+" 120Hz measured kinematic marker speeds only")
        axs[i].set_ylabel("Marker speed (m/s)")
        axs[i].set_xlabel("Source capture time (s): no verified kick-impact time")
        axs[i].grid(alpha=.25);axs[i].legend(fontsize=8)
    fig.tight_layout()
    out=root/"reports/v117_real_human_c3d_no_contact_truth.png"
    if out.exists():raise FileExistsError(out)
    fig.savefig(out,dpi=135);plt.close(fig)
    return str(out)

if __name__=="__main__":
    print("V117_REAL_G1_FIGURES",g1_failures(),flush=True)
    print("V117_REAL_C3D_FIGURE",human_marker_figure(),flush=True)
