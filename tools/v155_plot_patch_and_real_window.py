#!/usr/bin/env python3
"""Pure CPU figures from immutable v155 results; never overwrite old plots."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build(cpu:Path,hist:Path,prefix:Path):
    c=json.loads(cpu.read_text())
    h=json.loads(hist.read_text())
    allcop=[point for case in c["nominal"] for point in case["audit"]["cop_xy_m"]]
    xy=np.asarray(allcop)
    f1=Path(str(prefix)+"_fixture_COP.png")
    f2=Path(str(prefix)+"_point_vs_patch_angular_failure.png")
    f3=Path(str(prefix)+"_original200hz_window.png")
    if any(p.exists() for p in (f1,f2,f3)):
        raise FileExistsError("V155_PLOT_WRITE_ONCE_REFUSE")
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(xy[:,0],xy[:,1],s=12,alpha=.5)
    ax.plot([-.12,.12,.12,-.12,-.12],[-.045,-.045,.045,.045,-.045],linewidth=2)
    ax.set_xlabel("x: CoP coordinate (m), synthetic")
    ax.set_ylabel("y: CoP coordinate (m), synthetic")
    ax.set_title("v155: 6D sole patch / 48 CPU fixture QPs (NOT actual G1 GRF)")
    ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(f1,dpi=170);plt.close(fig)
    ctr=c["point_vs_patch_counterexample"]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(np.arange(len(ctr)),[x["point_base_roll_acceleration_rad_s2"] for x in ctr])
    ax.axhline(0,linewidth=1)
    ax.set_xticks(np.arange(len(ctr)),[x["stance"][0]+str(x["seed"]) for x in ctr])
    ax.set_xlabel("support / seed (synthetic)")
    ax.set_ylabel("Old point-foot base roll acceleration (rad/s²)")
    ax.set_title("2 Nm roll challenge: old 3D point-foot solves; v155 6D patch rejects 8/8")
    fig.tight_layout();fig.savefig(f2,dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for label,key in (("v32","frozen_v32"),("CMU10_02","cmu_10_02"),("CMU10_05","cmu_10_05")):
        v=h["runs"][key]
        ax.scatter([v["max_root_tilt_deg"]],[v["max_nonfoot_net_contact_N"]],s=90,label=label)
    ax.set_xlabel("max root tilt in original 1.170-2.820s window (deg)")
    ax.set_ylabel("max nonfoot NET force in window (N)")
    ax.set_title("v54 actual 200Hz window, 993 original states; NO native manifold truth")
    ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(f3,dpi=170);plt.close(fig)
    return [str(p) for p in (f1,f2,f3)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cpu",type=Path,required=True)
    p.add_argument("--history",type=Path,required=True)
    p.add_argument("--prefix",type=Path,required=True)
    x=p.parse_args()
    print(build(x.cpu,x.history,x.prefix))
if __name__=="__main__":
    main()
