#!/usr/bin/env python3
"""Physical failure plots from SHA-certified PhysX and measured C3D sources only."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from v113_phase_biomechanics import load_physical_arrays
from v117_causal_safety_statistics import ROOT,sha

def g1_failures(root=ROOT):
    batch=json.loads((root/"reports/v118_original_g1_batch.audit.json").read_text())
    paths=[]
    for scene in ("AM","AN"):
        items=[x for x in batch["runs"] if x["scene"]==scene
               and not x["observer"] and x["certified_original_G1_cuda_200Hz"]]
        if not items:continue
        fig,axs=plt.subplots(3,1,figsize=(10,10),sharex=True)
        for row in items:
            p=root/"reports"/("v118_"+row["name"]+".json")
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
        path=root/"reports"/("v118_"+scene+"_actual_qdot_support_com.png")
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=135)
        plt.close(fig)
        paths.append(str(path))
    return paths


if __name__=="__main__":
    print("V118_REAL_G1_FIGURES",g1_failures(),flush=True)

def ball_goal_paths(root=ROOT):
    """Observed ball XY and original goal aperture, no future trajectory generated."""
    batch=json.loads((root/"reports/v118_original_g1_batch.audit.json").read_text())
    dest=[]
    for scene in ("AM","AN"):
        fig,ax=plt.subplots(figsize=(9,7))
        goal_xy=None;goal_width=None
        for row in batch["runs"]:
            if row["scene"]!=scene or row["observer"]:continue
            f=root/"reports"/("v118_"+row["name"]+".json")
            if sha(f)!=row["raw_sha256"]:raise ValueError("PLOT_REAL_GOAL_SOURCE_SHA_MISMATCH")
            raw=json.loads(f.read_text())
            goal_xy=raw["goals"][0];goal_width=float(raw["goal_width_m"])
            trace=raw["traces"][0]
            pos=np.asarray([r["ball_xyz"] for r in trace],float)
            mask=pos[:,1]>=goal_xy[1]-2
            ax.plot(pos[mask,0],pos[mask,1],linewidth=1.2,
               label=row["control"]+(" PASS" if row["hard_gates"]["hard_gate_pass"] else " FAIL"))
            ax.scatter([pos[0,0]],[pos[0,1]],s=9)
        gx,gy=goal_xy
        ax.plot([gx-goal_width/2,gx+goal_width/2],[gy,gy],linewidth=4,
                label="Original goal mouth (center, nominal width)")
        ax.axvline(gx-goal_width/2,linestyle=":",linewidth=.8)
        ax.axvline(gx+goal_width/2,linestyle=":",linewidth=.8)
        ax.set_xlabel("Real ball center x (m)");ax.set_ylabel("Real ball center y (m)")
        ax.set_title("G1 "+scene+" 200Hz real ball XY path / original goal aperture\n"
                     "Goal pass/fail taken from engine, not this visualization")
        ax.grid(alpha=.25);ax.legend(loc="best")
        fig.tight_layout()
        target=root/"reports"/("v118_"+scene+"_real_ball_goal_path.png")
        if target.exists():raise FileExistsError("WRITE_ONCE_FIGURE:"+str(target))
        fig.savefig(target,dpi=140);plt.close(fig);dest.append(str(target))
    return dest
