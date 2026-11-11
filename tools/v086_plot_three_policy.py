#!/usr/bin/env python3
"""Plot real v85 200Hz G1 preimpact coordination: unique policies, not repeats."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run(base,out):
    prior=json.loads((base/"reports/v085_g1_physics_final.audit.json").read_text())
    sha={x["name"]:x["raw_sha256"] for x in prior["runs"]}
    names={"v32":"center_v32_off","CMU02":"center_cmu02_off",
           "CMU05(nonfoot_FAIL)":"center_cmu05_off"}
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for label,case in names.items():
        p=base/"reports"/("v085_"+case+".json")
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha[case]
        raw=json.loads(p.read_text())
        states=raw["traces"][0]
        t=np.asarray([r["time_s"] for r in states if r["time_s"]<=2.82+1e-6])
        states=states[:len(t)]
        qdot=np.asarray([r["joint_vel"] for r in states])
        qnames=raw["joint_names"]
        t=t-float(raw["records"][0]["contact_time_s"])
        mask=(t>=-.4)&(t<=1.25)
        for ax,group,keys in zip(axes,
                 ("Waist yaw/roll/pitch","Swing right leg","Opposite left arm"),
                 (["waist_yaw_joint","waist_roll_joint","waist_pitch_joint"],
                  ["right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
                   "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"],
                  ["left_shoulder_pitch_joint","left_shoulder_roll_joint",
                   "left_shoulder_yaw_joint","left_elbow_joint",
                   "left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint"])):
            idx=[qnames.index(n) for n in keys]
            energy=np.linalg.norm(qdot[:,idx],axis=1)
            ax.plot(t[mask],energy[mask],label=label,linewidth=1.15)
            ax.set_ylabel(group+"\n|qdot| (rad/s)")
            ax.axvline(0,color="gray",linestyle="--",linewidth=.8)
            ax.grid(alpha=.25)
    axes[0].legend(loc="upper right",fontsize=8)
    axes[-1].set_xlabel("Seconds relative to ORIGINAL G1 observed foot-ball impact")
    fig.suptitle("Historical v85 original G1 measured 200Hz phase curves; descriptive, not human naturalness")
    fig.tight_layout()
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=150)
    print("V086_LOCAL_PLOT",str(out),hashlib.sha256(out.read_bytes()).hexdigest())
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--v85-root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.v85_root,a.out)
