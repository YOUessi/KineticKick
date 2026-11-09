#!/usr/bin/env python3
"""v46 strict real PhysX joint-speed and shot task gates per NEW ball condition."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/"reports/v046_landing_transfer_final_audit.json"
    a=json.loads(p.read_text())
    samples=a["results"]
    if len(samples)!=6 or a["zero_parity"]["all_shared_original_physics_identical"] is not True:
        raise ValueError("Full paired original G1 PhysX evidence required")
    x=np.arange(6)
    values0=np.asarray([r["original_v41"]["original_qdot_ratio_peak"] for r in samples])
    values1=np.asarray([r["landing_v46"]["original_qdot_ratio_peak"] for r in samples])
    fig,ax=plt.subplots(figsize=(11.6,5.9))
    ax.bar(x-.18,values0,.36,label="Frozen v41 original G1 qdot")
    ax.bar(x+.18,values1,.36,label="v46 pre-landing 29DoF wholebody correction")
    ax.axhline(1.,linestyle="--",linewidth=1.2,color="black",
               label="Original joint-speed hard limit")
    for i,r in enumerate(samples):
        old=r["original_v41"]
        new=r["landing_v46"]
        ax.text(i,max(values0[i],values1[i])+.035,
              f"old {'PASS' if old['original_strict_pass'] else 'FAIL'} / "
              f"v46 {'PASS' if new['original_strict_pass'] else 'FAIL'}"
              f"\nreal v46 added {new['landing_actual_nonzero_control_steps']} steps",
              ha="center",fontsize=8)
        ax.text(i-.18,values0[i]+.005,f"{values0[i]:.3f}",ha="center",fontsize=8)
        ax.text(i+.18,values1[i]+.005,f"{values1[i]:.3f}",ha="center",fontsize=8)
    ax.set_xticks(x,[r["id"] for r in samples])
    ax.set_ylim(0,max(1.32,float(max(values0.max(),values1.max())+.23)))
    ax.set_ylabel("Maximum measured original G1 joint |qdot| / unchanged limit, full 14s")
    ax.set_title("KineticKick v0.1.46 | Six fixed new ball-parameter paired true PhysX tasks")
    ax.grid(axis="y",alpha=.22)
    ax.legend(loc="upper left",fontsize=8)
    fig.text(.01,.014,
       "All values are actual 5ms G1 PhysX body qdot. Exact new ball/velocity triples, one run per method. "
       "No certified professional human kicking biomechanics, hardware deployment or general population success rate.",
       fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.085,1,1))
    dest=ROOT/"media/v046_six_ball_cases_physical_qdot.png"
    dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=170)
    plt.close(fig)
    print("V046_STRICT_PHYSX_FIGURE",dest,dest.stat().st_size,flush=True)

if __name__=="__main__":main()
