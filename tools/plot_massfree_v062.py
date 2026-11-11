#!/usr/bin/env python3
"""Real original 200Hz PhysX failure/safety figures; no inferred human labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from audit_massfree_v062 import real_200hz,sha

CASES=("cuda_v32_safe_full","cuda_cmu05_safe_full","cpu_v32_safe_full","cpu_cmu05_safe_full")

def plot(root,out):
    out.mkdir(exist_ok=True,parents=True)
    raw={k:json.loads((root/"reports"/("v062_"+k+".json")).read_text()) for k in CASES}
    generated=[]
    for k in CASES:
        x=raw[k];frames,_=real_200hz(x["traces"][0]);o=x["safe_body_v062"]["rows"]
        assert len(frames)==len(o) and len(x["joint_names"])==29
        t=np.array([r["time_s"] for r in frames])
        d=np.array([r["left_wrist_hip_link_origin_distance_m"] for r in o])
        force=np.array([r["nonfoot_contact_N"] for r in frames])
        qdot=np.array([r["joint_vel"] for r in frames])
        lim=np.array(x["joint_velocity_limits"][0])
        left=np.array([r["support_foot_force_N"]["left"] for r in frames])
        right=np.array([r["support_foot_force_N"]["right"] for r in frames])
        impact=x["records"][0]["contact_time_s"]
        fig,axes=plt.subplots(4,1,figsize=(10,11),sharex=True)
        axes[0].plot(t,d);axes[0].set_ylabel("left wrist/hip\nLINK origins [m]")
        axes[1].plot(t,force);axes[1].set_ylabel("nonfoot net force [N]")
        axes[2].plot(t,np.abs(qdot/lim).max(axis=1));axes[2].axhline(1,ls=":",alpha=.5)
        axes[2].set_ylabel("peak of 29 qdot /\noriginal per-joint limit")
        axes[3].plot(t,left,label="left foot filtered");axes[3].plot(t,right,label="right foot filtered")
        axes[3].set_ylabel("foot support proxy [N]");axes[3].legend(fontsize=8)
        for a in axes:
            a.axvline(impact,ls="--",alpha=.6)
            a.grid(alpha=.25)
        axes[3].set_xlabel("Measured original 200Hz PhysX frame time [s]")
        fig.suptitle(f"{k}: Original G1, strict={x['stride_shot_success']};\n"
                     "link-origin distance is NOT collision shape clearance. "
                     "Observer's clock +10ms is not used.")
        fig.tight_layout()
        path=out/("v062_real_"+k+"_failure.png")
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=130);plt.close(fig);generated.append(str(path))
    return generated

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-dir",type=Path,default=Path("media"))
    a=p.parse_args()
    for s in plot(a.root.resolve(),a.output_dir.resolve()):print(s)
