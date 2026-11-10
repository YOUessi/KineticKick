#!/usr/bin/env python3
"""v47 true G1/PhysX learned recovery PPO vs original frozen v32 overview.

No extra generation of robot trajectories: all scores are from saved original
14-second PhysX free-root 29-DoF robot and soccer-ball contact.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA="reports/v047_independent_original_physx_audit.json"

def main():
    r=json.loads((ROOT/DATA).read_text())
    if len(r["records"])!=6:raise ValueError("Exactly the 6 predeclared one-shot tasks")
    xs=np.arange(6)
    old=np.asarray([t["old"]["actual_qdot_peak_over_original_limit_ratio"] for t in r["records"]])
    trained=np.asarray([t["learned"]["actual_qdot_peak_over_original_limit_ratio"] for t in r["records"]])
    fig,ax=plt.subplots(figsize=(11.4,5.8))
    a=ax.bar(xs-.18,old,width=.36,label="Frozen v32 PPO (original G1)")
    b=ax.bar(xs+.18,trained,width=.36,label="v47 safety-margin RL PPO")
    ax.axhline(1.,linestyle="--",linewidth=1.2,color="black",label="Original 100% speed limit")
    for i,row in enumerate(r["records"]):
        y=max(old[i],trained[i])
        ax.text(i,y+.04,
            f"v32 {'PASS' if row['old']['strict_original_full_task_pass'] else 'FAIL'} / "
            f"v47 {'PASS' if row['learned']['strict_original_full_task_pass'] else 'FAIL'}",
            ha="center",fontsize=8)
    ax.set_xticks(xs,[z["id"] for z in r["records"]],rotation=12)
    ax.set_ylim(0,max(1.25,float(max(old.max(),trained.max())+.20)))
    ax.set_ylabel("Max original 5ms measured |G1 qdot| / unchanged physical limit")
    ax.set_title("KineticKick v0.1.47 | 72-step warm-start real recovery PPO vs original v32")
    ax.legend(loc="upper left",fontsize=9)
    ax.grid(axis="y",alpha=.25)
    fig.text(.012,.007,
      "Six new exact ball and root-velocity parameter triples from the same robot/ball physical environment; "
      "one 14s real PhysX run each. Trained from 18 existing development cases; no professional human instep-motion "
      "truth, multi-seed reliability or hardware safety established.",
      fontsize=8)
    fig.tight_layout(rect=(0,.077,1,1))
    out=ROOT/"media/v047_safety_margin_ppo_paired_ball_physx.png"
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160)
    plt.close(fig)
    print("V047_PAIRED_ORIGINAL_PHYSX_FIGURE",out,out.stat().st_size,flush=True)
if __name__=="__main__":main()
