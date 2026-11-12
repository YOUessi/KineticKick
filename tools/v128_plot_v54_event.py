#!/usr/bin/env python3
"""Read-only v128 diagnostic failure plots from SHA-certified CPU audit JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(audit: dict, prefix: Path):
    event=audit["v54_equal_opposite_net_forces"]["sensor_event_s"]
    prefix.parent.mkdir(parents=True,exist_ok=True)

    fig,ax=plt.subplots(figsize=(11,5))
    for case,rows in audit["measured_pre_event_curves"].items():
        x=np.asarray(rows["time_s"])
        y=np.asarray(rows["primary_pair_URDF_capsule_gap_m"])*1000
        ax.plot(x,y,label=case,linewidth=2)
    ax.axhline(20,linestyle="--",label="frozen 20 mm proximity alert")
    ax.axhline(0,linestyle=":",label="URDF capsule overlap not proven contact")
    ax.axvline(event,linestyle="--",alpha=.7,label="5.145 s named sensor timestamp (no q)")
    ax.set(xlabel="Original G1 PhysX trace time (s)",ylabel="URDF capsule clearance lower bound (mm)",
           title="Original v54 G1: wrist/hip geometry before the net-force event")
    ax.set_xlim(event-.5,event+.015)
    ax.legend(fontsize=8)
    fig.tight_layout()
    first=Path(str(prefix)+"_wrist_hip_original_event.png")
    if first.exists():raise FileExistsError(first)
    fig.savefig(first,dpi=180);plt.close(fig)

    rows=audit["measured_pre_event_curves"]["cmu_10_05"]
    t=np.asarray(rows["time_s"])
    fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
    ax[0].plot(t,np.asarray(rows["primary_pair_URDF_capsule_gap_m"])*1000)
    ax[0].axhline(0,linestyle=":")
    ax[0].set_ylabel("URDF gap lower bound (mm)")
    ax[0].set_title("Failed CMU10_05: same-time sensor/trace observations are not native manifold truth")
    ax[1].plot(t,rows["measured_nonfoot_body_net_force_peak_N"],label="logged nonfoot net force")
    ax[1].plot(t,rows["original29_peak_qdot_ratio_per_frame"],label="max original 29DoF qdot ratio")
    ax[1].legend(fontsize=9)
    ax[1].set_ylabel("Net force (N) / qdot ratio")
    ax[2].plot(t,rows["ground_support_left_N"],label="left ground force proxy")
    ax[2].plot(t,rows["ground_support_right_N"],label="right ground force proxy")
    ax[2].plot(t,rows["COM_z_m"],label="COM z (m)")
    ax[2].legend(fontsize=9)
    ax[2].set_ylabel("Foot net force (N) / COM z (m)")
    for a in ax:
        a.axvline(5.135,linestyle=":",alpha=.8)
        a.axvline(event,linestyle="--",alpha=.8)
    ax[-1].set_xlabel("Original G1 200 Hz logged time (s), terminal marker excluded")
    fig.tight_layout()
    second=Path(str(prefix)+"_failed_G1_coupled_real_physics.png")
    if second.exists():raise FileExistsError(second)
    fig.savefig(second,dpi=180);plt.close(fig)
    return [str(first),str(second)]


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--prefix",type=Path,required=True)
    args=p.parse_args()
    print("\n".join(run(json.loads(args.audit.read_text()),args.prefix)))
