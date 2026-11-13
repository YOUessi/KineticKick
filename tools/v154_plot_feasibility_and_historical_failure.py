#!/usr/bin/env python3
"""Two evidence-class separated plots: synthetic WBC QP vs old PhysX failure.

Never conflate CPU numerical feasibility with actual shooting success.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from v146_real_200hz_coupled_audit import verified_sources,split_real_5ms_rows

def plot_synthetic(data,destination):
    phases=("approach","plant","coil","swing","followthrough","recover")
    fig,ax=plt.subplots(figsize=(10,5))
    for support in ("left","double"):
        vals=[]
        for phase in phases:
            rows=[c["independent"]["max_tau_ratio"] for c in data["nominal"]
                  if c["phase"]==phase and c["support"]==support]
            if len(rows)!=4:
                raise ValueError("V154_INCOMPLETE_CPU_MATRIX")
            vals.append(float(np.mean(rows)))
        ax.plot(range(len(phases)),vals,marker="o",label=support+" support")
    ax.set_xticks(range(len(phases)),phases,rotation=20)
    ax.set_ylabel("maximum synthetic actuator torque / synthetic limit")
    ax.set_title("v154 CPU QP feasibility — toy dynamics, NOT G1 PhysX")
    ax.legend()
    fig.tight_layout();fig.savefig(destination,dpi=130);plt.close(fig)

def plot_old_physics(destination):
    source=verified_sources()["cmu_10_05"]
    record=json.loads(Path(source).read_text())
    rows,annotations=split_real_5ms_rows(record["traces"][0],1028)
    if len(annotations)!=1:
        raise ValueError("V154_EARLY_TERMINATION_ANNOTATION_MISSING")
    recent=[r for r in rows if r["time_s"]>=4.7]
    t=[r["time_s"] for r in recent]
    nonfoot=[r["nonfoot_contact_N"] for r in recent]
    left=[r["support_foot_force_N"]["left"] for r in recent]
    right=[r["support_foot_force_N"]["right"] for r in recent]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(t,nonfoot,label="nonfoot net N",linewidth=2)
    ax.plot(t,left,label="left foot sensor NET N",alpha=.8)
    ax.plot(t,right,label="right foot sensor NET N",alpha=.8)
    ax.axvline(5.140,color="black",linestyle="--",label="last real 5ms state")
    ax.set_xlabel("original v54 PhysX time (s)")
    ax.set_ylabel("net contact sensor force magnitude (N)")
    ax.set_title("Historical CMU10_05 early termination — NOT native pair / GRF")
    ax.legend()
    fig.tight_layout();fig.savefig(destination,dpi=130);plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--matrix",required=True,type=Path)
    p.add_argument("--out-prefix",required=True,type=Path)
    args=p.parse_args()
    data=json.loads(args.matrix.read_text())
    if data["nominal_cpu_synthetic_count"]!=48 or data["new_real_physx_runs"]!=0:
        raise ValueError("V154_INCORRECT_EVIDENCE_CLASS")
    first=args.out_prefix.with_name(args.out_prefix.name+"_CPU_synthetic_QP.png")
    second=args.out_prefix.with_name(args.out_prefix.name+"_historical_real_nonfoot.png")
    for dest in (first,second):
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(str(dest))
    plot_synthetic(data,first)
    plot_old_physics(second)
    print(first,second,sep="\n")

if __name__=="__main__":
    main()
