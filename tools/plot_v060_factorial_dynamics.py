#!/usr/bin/env python3
"""Plot frozen v060 original-G1 observer factorial and failure dynamics.

Requires local write-once PhysX raw reports on Tang; source alone has no samples.
No GPU/CPU physical equivalence or post-termination extrapolation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def save_new(fig,path:Path):
    if path.exists():
        raise FileExistsError(f"Old scientific figure protected: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=170,bbox_inches="tight")
    plt.close(fig)
    print("V060_FIGURE",path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,default=ROOT/"media")
    a=p.parse_args()
    audited=json.loads(a.audit.read_text())
    if audited["version"]!="v0.1.60" or audited["physics_attempts"]!=10:
        raise ValueError("Not a complete frozen v060 audit")
    rows=audited["runs"]
    fig,ax=plt.subplots(figsize=(11,5))
    for i,r in enumerate(rows):
        valid=r["valid"]
        # Censored missing raw is deliberately different from strict FAIL.
        ax.bar(i,1 if valid else .25,hatch="" if valid else "xx")
        ax.text(i,.50 if valid else .28,
                ("PASS" if r["strict_success"] else "FAIL") if valid else "NO RAW",
                ha="center",va="bottom",fontsize=8,rotation=90)
    ax.set_xticks(range(len(rows)),[r["id"].replace("_","\n") for r in rows],
                  fontsize=8)
    ax.set_yticks([0,.25,1],["", "censored", "validated"])
    ax.set_ylim(0,1.35)
    ax.set_title("v0.1.60: frozen original 29DoF G1 observer factorial (10 PhysX starts)")
    ax.set_ylabel("200Hz original G1 sample qualification")
    save_new(fig,a.out_dir/"v060_observer_factorial_validity.png")

    selected=[("CUDA v32 / native only","cuda_v32_native_only"),
              ("CPU v32 / native only","cpu_v32_native_only")]
    physics=[]
    for label,name in selected:
        file=ROOT/"reports"/f"v060_{name}.json"
        if not file.is_file():
            raise FileNotFoundError(f"Original physics evidence unavailable: {file}")
        payload=json.loads(file.read_text())
        physics.append((label,payload))
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for label,r in physics:
        f=r["traces"][0]
        t=[x["time_s"] for x in f]
        ratio=[max(abs(v)/lim for v,lim in zip(x["joint_vel"],r["joint_velocity_limits"][0])) for x in f]
        feet=[sum(x["support_foot_force_N"].values()) for x in f]
        contact=[x["nonfoot_contact_N"] for x in f]
        axes[0].plot(t,ratio,label=label,linewidth=1)
        axes[1].plot(t,feet,label=label,linewidth=1)
        axes[2].plot(t,contact,label=label,linewidth=1)
        c=r["records"][0]["contact_time_s"]
        if c is not None:
            for ax in axes:
                ax.axvline(c,linestyle="--",alpha=.35)
    axes[0].axhline(1,linestyle=":",label="original qdot limit")
    axes[0].set_ylabel("max |qdot| / original limit")
    axes[1].set_ylabel("sum support-foot forces [N]")
    axes[2].set_ylabel("measured non-foot force [N]")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc="upper right",fontsize=8)
    axes[2].set_xlabel("actual physics time [s]; missing CPU recovery is NOT inferred")
    axes[2].set_xlim(.8,3.0)
    fig.suptitle("v0.1.60: same seed, DIFFERENT CPU/GPU backends (not physics parity)")
    save_new(fig,a.out_dir/"v060_measured_200hz_nonfoot_and_support.png")

if __name__=="__main__":
    main()
