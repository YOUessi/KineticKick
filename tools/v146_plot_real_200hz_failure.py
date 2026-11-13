#!/usr/bin/env python3
"""Real-data plots from previously SHA-audited v54 original G1 physical states.

All charts depict *historical* PhysX measurements; not new physical runs,
qualified ground reaction forces or native pair manifold truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def draw(input_file, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data=json.loads(Path(input_file).read_text())
    if data["historical_5ms_frames"]!=6628 or data["new_physx_episodes"]!=0:
        raise ValueError("ONLY_SHA_LOCKED_HISTORICAL_REAL_DATA_ALLOWED")
    names=list(data["cases"])
    fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for name in names:
        r=data["cases"][name]["full_real_200hz_coupled_rows"]
        t=[x["time_s"] for x in r]
        q=[max(x["qdot_ratio29"]) for x in r]
        non=[x["nonfoot_net_contact_N"] for x in r]
        ax[0].plot(t,q,label=name,linewidth=1.0)
        ax[1].plot(t,non,label=name,linewidth=1.0)
    ax[0].axhline(1,linestyle="--",label="original max qdot / limit = 1")
    ax[0].set_ylabel("Max |qdot| / original 29 limits")
    ax[0].legend()
    ax[1].set_ylabel("Non-foot net sensor force (N)")
    ax[1].set_xlabel("Original real PhysX time (s)")
    ax[1].axvline(1.57,linestyle=":",label="recorded foot-ball contact at 1.57s")
    ax[1].legend()
    fig.suptitle("Original v54 historical G1 200Hz: all29 qdot and nonfoot net force")
    fig.tight_layout()
    a=Path(str(prefix)+"_full14_dynamics.png")
    if a.exists():raise FileExistsError(a)
    fig.savefig(a,dpi=145)
    plt.close(fig)

    fail=data["cases"]["cmu_10_05"]["full_real_200hz_coupled_rows"]
    tail=[x for x in fail if x["time_s"]>=4.9]
    fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True)
    t=[x["time_s"] for x in tail]
    ax[0].plot(t,[x["nonfoot_net_contact_N"] for x in tail],label="nonfoot net force")
    ax[0].set_ylabel("N, net ContactSensor")
    ax[0].legend()
    ax[1].plot(t,[max(x["qdot_ratio29"]) for x in tail],label="all29 qdot ratio")
    ax[1].set_ylabel("Max original29 qdot ratio")
    ax[1].set_xlabel("Original real PhysX time (s)")
    for x in ax:
        x.axvline(5.135,linestyle=":",label="old nonfoot net peak 5.135s")
        x.axvline(5.140,linestyle="-.",label="last original complete state 5.140s")
        x.axvline(5.145,linestyle="--",label="named sensor event 5.145s, NO same-step original29")
        x.legend(loc="upper left",fontsize=8)
    fig.suptitle("CMU10_05 early termination: native left wrist-hip collider pair UNKNOWN")
    fig.tight_layout()
    b=Path(str(prefix)+"_last_contact_evidence_gap.png")
    if b.exists():raise FileExistsError(b)
    fig.savefig(b,dpi=145)
    plt.close(fig)

    paired=data["interpolicy_original_center_ball_comparison"]
    key1=paired["frozen_v32__cmu_10_02"]["pre_contact_original29_body_group_differences"]
    key2=paired["frozen_v32__cmu_10_05"]["pre_contact_original29_body_group_differences"]
    groups=("legs","waist","arms")
    fig,ax=plt.subplots(figsize=(9,5))
    positions=[0,1,2]
    width=.32
    ax.bar([x-width/2 for x in positions],
           [key1[g]["max_abs_29q_difference_rad"] for g in groups],
           width,label="CMU10_02 vs v32")
    ax.bar([x+width/2 for x in positions],
           [key2[g]["max_abs_29q_difference_rad"] for g in groups],
           width,label="CMU10_05 vs v32")
    ax.set_xticks(positions,list(groups))
    ax.set_ylabel("Largest actual original29 joint-position difference (rad)")
    ax.set_title("Real PhysX pre-contact activation: 1.405 to 1.565s, 33 ticks\n"
                 "Joint differences only; NOT learned human force transfer")
    ax.legend()
    fig.tight_layout()
    c=Path(str(prefix)+"_precontact_body_activation.png")
    if c.exists():raise FileExistsError(c)
    fig.savefig(c,dpi=145)
    plt.close(fig)
    return [str(a),str(b),str(c)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--prefix",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps({"generated_local_original_physx_figures":
                      draw(args.input,args.prefix),"not_new_physx":True}))


if __name__=="__main__":
    main()
