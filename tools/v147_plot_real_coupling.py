#!/usr/bin/env python3
"""v147 figures from SHA-verified, actually executed historical G1 physics.

Creates images only under gitignored reports/; no fabricated 5.145s state.
All torque×qdot values are estimated physical telemetry proxies, not GRF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v146_real_200hz_coupled_audit import verified_sources


def figures(snapshot, prefix):
    prefix=Path(prefix)
    if not prefix.parent.is_dir():
        raise ValueError("V147_PLOT_PARENT_MISSING")
    out1=prefix.with_name(prefix.name+"_body_proxy.png")
    out2=prefix.with_name(prefix.name+"_nonfoot_censor.png")
    if out1.exists() or out2.exists() or out1.is_symlink() or out2.is_symlink():
        raise FileExistsError("V147_PLOTS_WRITE_ONCE_NO_OLD_EVIDENCE_OVERWRITE")
    policies=list(snapshot["cases"])
    groups=["left_leg","right_leg","waist","left_arm","right_arm"]
    fig, axs=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    for ax,phase in zip(axs,("pre","post")):
        totals=[0.]*len(policies)
        for group in groups:
            vals=[snapshot["cases"][name]["all_window_metrics"][phase][
                "group_absolute_sim_torque_speed_proxy_integrals_J"][group]
                for name in policies]
            ax.bar(policies,vals,bottom=totals,label=group)
            totals=[x+y for x,y in zip(totals,vals)]
        ax.set_title(("Before","After")[phase=="post"]+
                     " 1.570s legacy contact proxy")
        ax.set_ylabel("Integrated |simulated tau × qdot| (proxy J)")
        ax.tick_params(axis="x",rotation=17)
        ax.grid(alpha=.2,axis="y")
    axs[1].legend(loc="upper right",fontsize=8)
    fig.suptitle("Historical PhysX: segment-resolved original29 mechanical proxy; NOT measured motor output")
    fig.savefig(out1,dpi=160)
    plt.close(fig)

    original=verified_sources()
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True,layout="constrained")
    for name,path in original.items():
        obj=json.loads(Path(path).read_text())
        trace=obj["traces"][0]
        samples=[r for r in trace if 4.9<=r.get("time_s",0)<=5.145
                 and "nonfoot_contact_N" in r and "support_foot_force_N" in r]
        if not samples:
            raise ValueError("V147_SOURCE_MISSING_FAILURE_WINDOW")
        xs=[r["time_s"] for r in samples]
        axs[0].plot(xs,[r["nonfoot_contact_N"] for r in samples],label=name)
        axs[1].plot(xs,[r["support_foot_force_N"]["left"] for r in samples],
                    label=name+" left-net",linestyle="-")
        axs[1].plot(xs,[r["support_foot_force_N"]["right"] for r in samples],
                    label=name+" right-net",linestyle="--")
    for ax in axs:
        ax.axvline(5.14,linestyle=":",alpha=.7)
        ax.axvline(5.145,linestyle="--",alpha=.7)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
        ax.set_ylabel("Recorded link net force (N); not isolated GRF")
    axs[1].set_xlabel("Physics time (s), original v54")
    fig.suptitle("5.145s named event is NOT an aligned 29DoF physics state; collider pair UNKNOWN")
    fig.savefig(out2,dpi=160)
    plt.close(fig)
    return [str(out1),str(out2)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--prefix",type=Path,required=True)
    args=p.parse_args()
    files=figures(json.loads(args.input.read_text()),args.prefix)
    print(json.dumps({"local_figures":files, "source_is_historical_real_physx":True,
                      "native_wrist_hip_truth":"UNKNOWN"}))


if __name__=="__main__":
    main()
