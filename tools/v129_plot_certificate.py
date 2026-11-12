#!/usr/bin/env python3
"""Plot only actual v54 sampled kinematics; never imply Native manifold truth."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    audit=json.loads(args.input.read_text())
    if audit["native_physx_pair_certified"] or audit["new_physx_rollouts"]:
        raise ValueError("UNTRUSTED_NATIVE_PROVENANCE")
    fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
    for ax,(name,curves) in zip(axes,audit["last_250ms_real_physics_curves"].items()):
        t=curves["time_s"]
        cap=[v*1000 for v in curves["capsule_lower_bound_m"]]
        fin=[v*1000 for v in curves["exact_finite_cylinder_lower_bound_m"]]
        ax.plot(t,cap,label="URDF capsule lower bound")
        ax.plot(t,fin,label="URDF finite cylinder support-plane lower bound")
        ax.plot([t[0],t[-1]],[20,20],ls=":",label="20 mm old alert")
        ax.set_ylabel(f"{name}\nclearance lower bound (mm)")
        ax.legend(fontsize=7,loc="upper right")
    axes[-1].set_xlabel("Original observed simulation time (s)")
    fig.suptitle("v54 original 29DoF 200Hz URDF bounding vs finite-cylinder certificate\nNOT a runtime USD/PhysX collider-pair or manifold measurement")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    fig.savefig(args.output,dpi=150)
    plt.close(fig)
    print(args.output)

if __name__=="__main__":
    main()
