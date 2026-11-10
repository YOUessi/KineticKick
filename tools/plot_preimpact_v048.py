#!/usr/bin/env python3
"""v48 visualize actual G1/PhysX new pre-impact PPO vs unchanged v32."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA="reports/v048_preimpact_full_physics_audit.json"

def main():
    r=json.loads((ROOT/DATA).read_text())
    if len(r["cases"])!=6 or not r["zero_physics_equals_original"]:
        raise ValueError("Full predeclared true original PhysX required")
    rows=r["cases"];names=[x["id"].replace("preimpact_","") for x in rows]
    xx=np.arange(6)
    baseline=np.asarray([q["original_v32"]["true_outgoing_ball_speed_xy_mps"] or 0. for q in rows])
    learned=np.asarray([q["learned_preimpact_v48"]["true_outgoing_ball_speed_xy_mps"] or 0. for q in rows])
    a=np.asarray([q["original_v32"]["max_original_joint_velocity_limit_ratio"] for q in rows])
    b=np.asarray([q["learned_preimpact_v48"]["max_original_joint_velocity_limit_ratio"] for q in rows])
    fig,axes=plt.subplots(2,1,figsize=(12.4,9.2))
    ax=axes[0]
    ax.bar(xx-.18,baseline,.36,label="Original v32 G1 right-instep")
    ax.bar(xx+.18,learned,.36,label="v48 new trained pre-impact 29DoF PPO")
    ax.axhline(3.5,linestyle="--",color="black",linewidth=1.,label="Original minimum ball speed")
    ax.set_ylim(0,max(6,max(baseline.max(),learned.max())+1.))
    ax.set_ylabel("Real observed ball horizontal exit speed (m/s)")
    ax.set_title("True G1/PhysX outgoing speed, identical ball parameters")
    for i,item in enumerate(rows):
        old=item["original_v32"]["strict_physical_pass"]
        new=item["learned_preimpact_v48"]["strict_physical_pass"]
        observed=item["learned_preimpact_v48"]["new_preimpact_policy_nonzero_control_steps"]
        ax.text(i,max(baseline[i],learned[i])+.08,
                f"old {'PASS' if old else 'FAIL'} / PPO {'PASS' if new else 'FAIL'}\n"
                f"real new G1 action {observed} steps",ha="center",fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y",alpha=.25)
    ax=axes[1]
    ax.bar(xx-.18,a,.36,label="Frozen v32 qdot risk")
    ax.bar(xx+.18,b,.36,label="v48 trained PPO true qdot risk")
    ax.axhline(1.,linestyle="--",color="black",linewidth=1.2,
               label="Unchanged original G1 joint-speed physical limit")
    ax.set_ylim(0,max(1.3,max(a.max(),b.max())+.2))
    ax.set_ylabel("Full 14s worst |actual joint velocity| / original actuator limit")
    ax.set_xticks(xx,names)
    ax.set_title("29-joint 5ms original PhysX physical speed, not target-reference bounds")
    ax.legend(fontsize=8)
    ax.grid(axis="y",alpha=.25)
    fig.suptitle("KineticKick v0.1.48 | Actual learned PRECONTACT whole-body PPO vs original G1")
    fig.text(.02,.006,
        "Six pre-registered new exact soccer ball/velocity triples in original IsaacLab G1 physics; one true 14s rollout each. "
        "The 3DSP weak human cue comes from broadcast-estimated 3D, not verified professional fast instep force or human naturalness.",
        fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.965))
    file=ROOT/"media/v048_learned_preimpact_true_physx_comparison.png"
    file.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(file,dpi=160)
    plt.close(fig)
    print("V048_ACTUAL_G1_PREIMPACT_PPO_PHYSX_FIGURE",file,file.stat().st_size)
if __name__=="__main__":main()
