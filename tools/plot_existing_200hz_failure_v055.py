#!/usr/bin/env python3
"""Plot real 200Hz foot load, joint velocity, COM, wrist/hip event against time."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def make():
    p=ROOT/"reports/v054_cmu_10_05_contact_diagnostics.json"
    s=json.loads(p.read_text());tr=s["traces"][0]
    t=np.array([x["time_s"] for x in tr])
    qd=np.array([x["joint_vel"] for x in tr])
    limit=np.asarray(s["joint_velocity_limits"][0]);peak=(np.abs(qd)/limit).max(axis=1)
    com=np.array([x["whole_body_com_xyz"] for x in tr])
    support=np.array([[x["support_foot_force_N"][f] for f in ("left","right")] for x in tr])
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    axes[0].plot(t,support[:,0],label="Left foot normal [N]");axes[0].plot(t,support[:,1],label="Right foot normal [N]")
    axes[0].set_ylabel("Ground normal (N)");axes[0].legend()
    axes[1].plot(t,peak,label="Max |qdot| / original joint limit");axes[1].axhline(1,linestyle="--",label="Original limit");axes[1].set_ylabel("Max speed ratio");axes[1].legend()
    axes[2].plot(t,com[:,2],label="Observed full-body COM height");axes[2].plot(t,[x["root_xyz"][2] for x in tr],label="Root height")
    axes[2].set_ylabel("Height [m]");axes[2].legend()
    axes[3].plot(t,[x["root_tilt_deg"] for x in tr],label="Actual root tilt [deg]")
    axes[3].set_ylabel("Tilt [deg]");axes[3].set_xlabel("Real PhysX time [s]");axes[3].legend()
    for ax in axes:
        ax.axvline(s["records"][0]["contact_time_s"],alpha=.5,linestyle=":",label="instep impact")
        ax.axvline(t[-1],alpha=.7,linestyle="-.",label="physical_nonfoot stop")
    named=s["nonfoot_contact_diagnostics"]
    if named:
        axes[0].set_title("v54 10_05: 25.890 N opposing left wrist / hip at 5.145 s (one 5ms beyond final trace)")
    else:axes[0].set_title("v54 10_05 observed trace")
    fig.tight_layout()
    out=ROOT/"media/v055_existing_physx_wrist_hip_failure_dynamics.png"
    fig.savefig(out,dpi=140);plt.close(fig)
    print("V055_PLOT",out,out.stat().st_size)
if __name__=="__main__":make()
