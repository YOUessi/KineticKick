#!/usr/bin/env python3
"""v49 original G1 joint qdot temporal sequence near REAL measured instep.

Only original G1 joint coordinate rates. Do NOT label hip-knee-ankle ordering
as a validated human proximal/distal segmental energy transfer phenomenon.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/"data/research_human_kicks/kinetic_chain_v049/kinetic_chain_physx_v049.npz"
def main():
    with np.load(FILE,allow_pickle=False) as z:
        qdot=z["qdot"].astype(float)
        time=z["relative_real_contact_s"].astype(float)
        names=z["original_G1_joint_names"].tolist()
        case=z["source_case"].tolist()
        controller=z["controller"].tolist()
    target=("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint")
    indexes=[names.index(j) for j in target]
    xx=np.flatnonzero(np.asarray(case)=="preimpact_new01")
    if len(xx)!=2 or [controller[i] for i in xx]!=["original_v32","learned_preimpact_v48"]:
        raise ValueError("Original paired G1 joint trace alignment mismatch")
    fig,ax=plt.subplots(figsize=(12.2,5.5))
    for joint,j in zip(target,indexes):
        y=np.abs(qdot[xx[0],:,j])
        mask=time<=0.
        peak=int(np.argmax(y[mask]))
        ax.plot(time,y,label=joint+" (original v32)",linewidth=1.5)
        ax.scatter([time[mask][peak]],[y[mask][peak]],s=25)
    ax.axvline(0,linestyle="--",color="black",label="Measured first foot/ball impact")
    ax.set_xlabel("Seconds relative to real right-foot / ball contact (actual original 5ms physics)")
    ax.set_ylabel("Absolute joint angular velocity in each G1 DOF (rad/s)")
    ax.set_title("KineticKick v0.1.49 | Right hip/knee/ankle joint-speed peak timing (one previously known G1 shot)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.text(.025,.01,
       "G1 joint qdot in robot-specific coordinates is NOT a human anatomical segment speed/torque measurement. "
       "A qdot-peak order is not proof of musculoskeletal power transfer or visual naturalness.",
       fontsize=8)
    fig.tight_layout(rect=(0,.065,1,1))
    target_file=ROOT/"media/v049_original_right_leg_joint_speed_timing.png"
    target_file.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target_file,dpi=160)
    plt.close(fig)
    print("V049_CONTACT_ALIGNED_REAL_G1_QDOT_TIMING",target_file,target_file.stat().st_size)
if __name__=="__main__":main()
