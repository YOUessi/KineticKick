#!/usr/bin/env python3
"""Visualize actual v41/v46 original 5ms joint-speed near known landing hazard.

No generated animation, no unsafe qdot limit relaxation, no extrapolated
mechanical impulse model. Real joint speed ratio from original PhysX traces.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def physical_qdot(path):
    r=json.loads((ROOT/path).read_text())
    raw=r["traces"][0]
    t=np.asarray([v["time_s"] for v in raw])
    v=np.asarray([x["joint_vel"] for x in raw])
    limit=np.asarray(r["joint_velocity_limits"][0])
    if v.shape!=(len(raw),29) or limit.shape!=(29,):
        raise ValueError("Original G1 velocity/limit schema changed")
    return t,np.max(abs(v)/limit,axis=-1),r["records"][0]["contact_time_s"]

def main():
    before="reports/v042_trial01_style_v41.json"
    after="reports/v046_known_trial01_landing.json"
    t0,q0,c0=physical_qdot(before)
    t1,q1,c1=physical_qdot(after)
    if not np.array_equal(t0,t1) or c0!=c1:
        raise ValueError("Cannot compare different time/first physical contact")
    window=(t0>=c0-.1)&(t0<=c0+1.3)
    fig,ax=plt.subplots(figsize=(11.2,5))
    ax.plot(t0[window]-c0,q0[window],label="Frozen v41 style only, original G1 qdot ratio",lw=1.7)
    ax.plot(t1[window]-c1,q1[window],label="v46 real pre-landing whole-body reference",lw=1.7)
    ax.axhline(1.,linestyle="--",color="black",lw=1.,label="Original physical G1 joint-speed hard limit")
    ax.axvline(0,linestyle=":",color="grey",lw=1.,label="Real ball-foot first impact")
    oldpeak=int(np.argmax(q0[window]))
    t_peak=t0[window][oldpeak]-c0
    y_peak=q0[window][oldpeak]
    ax.scatter([t_peak],[y_peak],s=38,marker="x")
    ax.annotate(f"Original overspeed {y_peak:.4f}x at +{t_peak:.3f}s",
        (t_peak,y_peak),xytext=(t_peak-.18,1.08),fontsize=9,
        arrowprops=dict(arrowstyle="->",lw=.8))
    ax.grid(alpha=.25)
    ax.set_xlabel("Seconds after true original PhysX right-foot/ball impact")
    ax.set_ylabel("Worst measured 29-joint |qdot| / original G1 joint speed limit")
    ax.set_title("KineticKick v0.1.46 — trial01: actual post-kick whole-body landing safety")
    ax.legend(loc="upper right",fontsize=8)
    ax.set_xlim(-.1,1.3)
    ax.set_ylim(0,max(1.2,float(max(q0[window].max(),q1[window].max())+.11)))
    fig.text(.02,.015,
       "Known development condition: not unseen generalization. Original G1 physics at 5ms samples, same original motor limits. Relative human reference source is broadcast-estimated pose without verified professional instep kinetics.",
       fontsize=7.5)
    fig.tight_layout(rect=(0,.065,1,1))
    out=ROOT/"media/v046_known_trial01_real_landing_qdot.png"
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160)
    plt.close(fig)
    print("V046_PHYSX_QDOT_MECHANISM_FIGURE",out,out.stat().st_size,flush=True)

if __name__=="__main__":main()
