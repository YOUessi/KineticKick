#!/usr/bin/env python3
"""Plot v41 strictly measured physics on known development conditions only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def main():
    k=json.loads((ROOT/"reports/v041_coordination_style_comparison.json").read_text())
    if not k["zero_pass"] or not k["all_evaluated_physically_pass"]:
        raise ValueError("Original G1 physics gates did not all pass")
    data=k["records"]
    if len(data)!=3:raise ValueError("Only the three frozen cases")
    labels=["Nominal","Near / slow B","Left-near / slow A"]
    x=np.arange(len(data))
    speeds0=[v["original_outgoing_speed_xy_mps"] for v in data]
    speeds1=[v["new_outgoing_speed_xy_mps"] for v in data]
    fig,ax=plt.subplots(figsize=(10.1,4.9))
    bars=ax.bar(x-.19,speeds0,.38,label="Original v32 speed")
    bars2=ax.bar(x+.19,speeds1,.38,label="v41 style reference + original v32")
    for ii,(v0,v1) in enumerate(zip(speeds0,speeds1)):
       ax.text(ii-.19,v0+.045,f"{v0:.3f}",ha="center",fontsize=9)
       ax.text(ii+.19,v1+.045,f"{v1:.3f}",ha="center",fontsize=9)
    for i,r in enumerate(data):
       label=f"old goal {'yes' if r['original_goal'] else 'NO'} / new goal {'yes' if r['candidate_goal'] else 'NO'}"
       ax.text(i,3.7,label,ha="center",fontsize=9)
    ax.set_xticks(x,labels)
    ax.set_ylim(0,5.35)
    ax.set_ylabel("Measured outgoing ball horizontal speed (m/s)")
    ax.set_title("KineticKick v0.1.41 — true PhysX known-task instep shots")
    ax.grid(axis="y",alpha=.3)
    ax.legend()
    fig.text(.02,.012,"All 3 tasks were previous development conditions; 1 train-only video-estimated 3DSP clip, not verified skilled-instestep biomechanics. Exactly one deterministic rollout each. No unseen generalization or naturalness claim.",fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.08,1,1))
    dest=ROOT/"media/v041_three_known_case_physical_shots.png"
    dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=160)
    plt.close(fig)
    print("V041_SCIENCE_FIGURE_SAVED",dest,dest.stat().st_size)
if __name__=="__main__":main()
