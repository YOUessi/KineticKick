#!/usr/bin/env python3
"""Plot v37 observed normal, combined foot impulse and target directions.

Descriptive same-window physical evidence; never a friction-off counterfactual.
"""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_ball_impulse import heading

ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="reports/v037_paired_contact_mechanics.json")
    ap.add_argument("--output",default="media/v037_measured_contact_directions.png")
    a=ap.parse_args()
    src=ROOT/a.input
    j=json.loads(src.read_text())
    if not j["all_momentum_balances_verified"] or not j["all_readonly_parity"]:
        raise ValueError("Cannot plot unverified physical contact")
    records=j["recorded_cases"]
    if len(records)!=4:raise ValueError("Exactly the four declared paired outcomes required")
    labels=["A old (miss)","A learned (goal)","B old (miss)","B learned (miss)"]
    normal=[heading(v["aggregate_Ns"]["normal"]) if "aggregate_Ns" in v else v["normal_heading_deg"] for v in records]
    total=[v["total_heading_deg"] if "total_heading_deg" in v else v["combined_heading_deg"] for v in records]
    a_t=j["matched_pairs"]["left_near_slow"]["target_deg"]
    b_t=j["matched_pairs"]["near_slow_17"]["target_deg"]
    target=[a_t,a_t,b_t,b_t]
    x=np.arange(4,dtype=float)
    fig,ax=plt.subplots(figsize=(11,5.4))
    width=.24
    ax.bar(x-width,normal,width,label="Foot normal impulse")
    ax.bar(x,total,width,label="Normal + tangential impulse")
    ax.scatter(x+width,target,marker="D",s=55,label="Ball-to-goal center")
    ax.axhline(0,linewidth=.7)
    ax.set_xticks(x,labels)
    ax.set_ylabel("Horizontal direction from world -Y (degrees)")
    ax.set_title("KineticKick v0.1.37 | Measured PhysX foot-ball impulse direction")
    ax.grid(axis="y",alpha=.25)
    ax.legend()
    for xs,ys in ((x-width,normal),(x,total),(x+width,target)):
        for xi,yi in zip(xs,ys):
            ax.annotate(f"{yi:+.2f}°",(xi,yi),textcoords="offset points",xytext=(0,7),
                        ha="center",fontsize=8)
    fig.text(.03,.005,"Old A uses the independently signed-audited v0.1.33 capture. New v36 and old B captures are byte-identical in physics to their saved original controls; registered metadata exceptions are documented. Not a causal friction-free simulation or a human-naturalness score.",
             fontsize=7,wrap=True)
    fig.tight_layout(rect=(0,.07,1,1))
    dest=ROOT/a.output
    dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=175)
    plt.close(fig)
    print("V037_FIGURE",str(dest),"bytes",dest.stat().st_size,flush=True)
if __name__=="__main__":main()
