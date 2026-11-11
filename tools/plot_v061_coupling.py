#!/usr/bin/env python3
"""Plot genuine 200Hz original G1 support, collision and joint-group timing."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_v061_coupling import trimmed_200hz,GROUPS

def read(path):
    raw=json.loads(path.read_text())
    frames=trimmed_200hz(raw)
    return raw,frames,np.asarray([f["time_s"] for f in frames])
def plot(paths,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    if len(paths)!=3:raise ValueError("Need registered 3 original physical cases")
    fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for path in paths:
        raw,frames,time=read(path)
        label=path.stem.replace("v060_","").replace("v061_","")
        mask=(time>=1.16)&(time<=2.86)
        names={n:i for i,n in enumerate(raw["joint_names"])}
        dq=np.array([f["joint_vel"] for f in frames])
        for a,(group,group_names) in zip(axs,[(k,v) for k,v in GROUPS.items()][:3]):
            ind=[names[n] for n in group_names]
            vals=np.sqrt(np.mean(dq[:,ind]**2,axis=1))
            a.plot(time[mask],vals[mask],lw=1.3,label=label)
            a.set_ylabel(group+" RMS qdot")
            a.axvline(raw["records"][0]["contact_time_s"],color="k",ls="--",alpha=.3)
    for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
    axs[-1].set_xlabel("Real PhysX 200Hz clock [s]")
    fig.suptitle("Observed pre/post impact full-body timing; not a human naturalness label")
    fig.tight_layout()
    path=output_dir/"v061_200hz_human_proxy_caution_joint_coupling.png"
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=155);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for path in paths:
        raw,frames,t=read(path)
        label=path.stem
        mask=(t>=1.55)&(t<=5.3)
        if not mask.any():continue
        n=np.array([f["nonfoot_contact_N"] for f in frames])
        l=np.array([f["support_foot_force_N"]["left"] for f in frames])
        r=np.array([f["support_foot_force_N"]["right"] for f in frames])
        com=np.array([f["whole_body_com_xyz"][2] for f in frames])
        axs[0].plot(t[mask],n[mask],label=label)
        axs[1].plot(t[mask],l[mask],label=label+" left")
        axs[1].plot(t[mask],r[mask],ls=":",label=label+" right")
        axs[2].plot(t[mask],com[mask],label=label)
    for ax,y in zip(axs,["Nonfoot net N","Filtered foot support N","Whole-body COM z [m]"]):
        ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend(fontsize=7)
    axs[-1].set_xlabel("Actual physical time s (censored at early termination)")
    fig.suptitle("Original G1 baseline and observed nonfoot stop; CPU/GPU not physically interchangeable")
    fig.tight_layout()
    path2=output_dir/"v061_real_nonfoot_failure_and_support.png"
    if path2.exists():raise FileExistsError(path2)
    fig.savefig(path2,dpi=155);plt.close(fig)
    return [str(path),str(path2)]
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,nargs=3,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    print(*plot(a.inputs,a.out_dir),sep="\n")
