#!/usr/bin/env python3
"""Create separate evidence plots from v130 SHA-qualified measurements.

Plots are local generated reports only; never a native manifold or GRF proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v130_real_com_support_work as primary
import v130_paired_controller_counterfactual as paired


def save_new(fig,path:Path):
    if path.exists():
        raise FileExistsError("REFUSE_OVERWRITE_OLD_EVIDENCE:"+str(path))
    fig.savefig(path,dpi=140,bbox_inches="tight")
    plt.close(fig)


def make_plots(audit_path:Path,out:Path):
    x=json.loads(audit_path.read_text())
    if x["protocol_commit"]!=primary.FROZEN_SHA or x["physical_200hz_frames"]!=6628:
        raise ValueError("UNQUALIFIED_200HZ_PLOT_SOURCE")
    cases=x["evidence"]
    labels=[c["case"] for c in cases]
    xs=np.arange(len(labels))
    fig,ax=plt.subplots(figsize=(8,4))
    pre=[c["phases"]["pre_400ms"]["total_signed_joint_motor_work_proxy_J"] for c in cases]
    post=[c["phases"]["post_1250ms"]["total_signed_joint_motor_work_proxy_J"] for c in cases]
    ax.bar(xs-.19,pre,width=.38,label="Before impact, 400 ms")
    ax.bar(xs+.19,post,width=.38,label="After impact, 1250 ms")
    ax.set_xticks(xs,labels)
    ax.set_ylabel("Signed estimated robot joint work (J)")
    ax.set_title("Original G1 PhysX: no human force supervision")
    ax.legend()
    save_new(fig,out/"v130_real_impulse_phase_joint_work.png")

    cfg=primary.frozen()
    src=paired.previous.frozen_protocol()
    fail=paired.measured("cmu_10_05",cfg,src,Path(src["data_root"]))
    t=fail["t"]
    mask=t>=4.90
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(t[mask],fail["support"][mask,0],label="left foot force proxy (N)")
    ax.plot(t[mask],fail["support"][mask,1],label="right foot force proxy (N)")
    ax.plot(t[mask],fail["nonfoot"][mask],label="non-foot net force proxy (N)")
    ax.axvline(5.135,linestyle="--",label="25.89 N peak in measured trace")
    ax.set_xlim(4.90,5.15)
    ax.set_ylabel("Logged net forces / force proxies (N)")
    ax.set_xlabel("Physical simulation time (s); samples end at 5.140 s")
    ax.set_title("Original v54 failed CMU10_05: no real q at 5.145 s")
    ax.legend(loc="upper left")
    save_new(fig,out/"v130_real_v54_failure_support_nonfoot.png")

    fig,ax=plt.subplots(figsize=(9,4))
    for case in ("frozen_v32","cmu_10_02","cmu_10_05"):
        measured=paired.measured(case,cfg,src,Path(src["data_root"]))
        a=measured["t"]<=5.14
        ax.plot(measured["t"][a],measured["com"][a,2],label=case)
    ax.set_xlabel("Time from original PhysX (s)")
    ax.set_ylabel("Logged whole-body COM world z (m)")
    ax.set_title("G1 measured COM height, truncated to common real frames")
    ax.legend()
    save_new(fig,out/"v130_real_three_controller_common_com_z.png")
    print("THREE_SAVED",str(out))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--audit",type=Path,
                    default=primary.ROOT/"reports/v130_real_com_support_work.audit.json")
    ap.add_argument("--out",type=Path,default=primary.ROOT/"reports")
    args=ap.parse_args()
    make_plots(args.audit,args.out)
