#!/usr/bin/env python3
"""Reproduce 3 uncensored/censored real G1 event figures; CPU only.

The named 5.145-second event belongs to failed CMU10_05; successful
controller traces use that clock as alignment reference only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v128_v54_event_geometry as v128
from v134_real_29dof_coupled_window import CASES, CONTACT_ANCHOR, DT
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

def _original_physics():
    frozen=v128.frozen_protocol()
    out={}
    for case in CASES:
        name=f"v054_{case}_contact_diagnostics.json"
        path=Path(frozen["data_root"]) / name
        v128.require_sha(path,frozen["input_sha256"][name])
        raw=json.loads(path.read_text())
        count=1028 if case=="cmu_10_05" else 2800
        if tuple(raw["joint_names"])!=tuple(G1_JOINT_ORDER):
            raise ValueError("NOT_ORIGINAL_29_DOF")
        out[case]=v128.validate_rows(raw,count+int(case=="cmu_10_05"),DT)
    return out

def make_plots(outprefix:Path):
    raw=_original_physics()
    outprefix=Path(outprefix)
    outputs=[]
    series=[
        ("29dof_qdot_ratio", lambda v: np.max(np.abs(v["dq"])/v["limits"][None,:],axis=1),
         "All 29 joints / original qdot limits (maximum)",1.0),
        ("nonfoot_net_force",lambda v:v["nonfoot"],"Logged non-foot rigid-body net force [N]",20.0),
        ("foot_support_proxy",lambda v:v["support"][:,0],
         "Logged left-foot support force proxy [N]",None),
        ("com_horizontal_displacement",lambda v: np.linalg.norm(
             v["com"][:,:2]-v["com"][0,:2],axis=1),
         "Logged COM horizontal displacement from first frame [m]",None),
    ]
    for suffix,extract,label,threshold in series:
        fig,ax=plt.subplots(figsize=(9,4.5))
        for case in CASES:
            data=raw[case]
            t=data["t"]
            selected=(t>=CONTACT_ANCHOR-.4-1e-8)&(t<=CONTACT_ANCHOR+1.25+1e-8)
            y=extract(data)
            ax.plot(t[selected],y[selected],label=case)
        ax.axvline(CONTACT_ANCHOR,ls="--",lw=1.1,label="5.145s named event (failed case)")
        if threshold is not None:
            ax.axhline(threshold,ls=":",lw=1.1,label="frozen threshold")
        ax.set_xlim(4.70,6.45)
        ax.set_xlabel("Physics sample clock [s], no fabricated post-failure data")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        dest=outprefix.with_name(outprefix.name+"_"+suffix+".png")
        # Exclusive creation; never silently overwrite evidence.
        with dest.open("xb") as f:
            fig.savefig(f,format="png",dpi=155)
        plt.close(fig)
        outputs.append(str(dest))
    return outputs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-prefix",type=Path,required=True)
    args=p.parse_args()
    for f in make_plots(args.output_prefix):
        print(f)

if __name__ == "__main__":
    main()
