#!/usr/bin/env python3
"""Local-only charts from SHA checked original measured G1 PhysX, never diagrams presented as experiments."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def plot():
    a=json.loads((ROOT/"reports/v063_independent_recheck.json").read_text())
    print("CERTIFIED",a["scorable"],"/",a["real_PhysX_attempts"])
    for case in ("center","hold_a","hold_b","hold_c"):
        fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        found=0
        for label in ("v32","synth"):
            path=ROOT/f"reports/v063_{case}_{label}.json"
            if not path.exists():continue
            z=json.loads(path.read_text());trace=z["traces"][0]
            t=np.array([v["time_s"] for v in trace])
            vel=np.array([v["joint_vel"] for v in trace])
            names=z["joint_names"]
            support=np.array([[v["support_foot_force_N"]["left"],v["support_foot_force_N"]["right"]] for v in trace])
            nonfoot=np.array([v["nonfoot_contact_N"] for v in trace])
            j=[names.index(s) for s in ("right_hip_pitch_joint","waist_yaw_joint","right_knee_joint","left_shoulder_pitch_joint")]
            axes[0].plot(t,np.linalg.norm(vel[:,j],axis=-1),label=label+" 4-joint speed proxy")
            axes[1].plot(t,support[:,0],label=label+" left support")
            axes[2].plot(t,nonfoot,label=label+" original non-foot force")
            ct=z["records"][0].get("contact_time_s")
            if ct is not None:
                for ax in axes:ax.axvline(ct,linestyle=":",alpha=.4)
            found+=1
        for ax,y in zip(axes,("Joint speed norm, rad/s (not naturalness)","Measured left foot support, N","Original non-foot force, N")):
            ax.set_ylabel(y);ax.legend(fontsize=8);ax.grid(alpha=.2)
        axes[-1].set_xlabel("Original 200 Hz measured physical time, s")
        fig.suptitle(f"v0.1.63 {case}: real G1 physics; unscorable runs never interpolated")
        fig.tight_layout()
        dest=ROOT/f"media/v063_{case}_real_dynamics.png"
        dest.parent.mkdir(exist_ok=True)
        fig.savefig(dest,dpi=145)
        plt.close(fig)
        print(dest.name,found)
if __name__=="__main__":plot()
