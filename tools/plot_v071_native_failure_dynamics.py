#!/usr/bin/env python3
"""Make factual read-only 200Hz failure and physical group-motion evidence figures.

Figures are LOCAL media only: never commit raw/source media nor impute beyond
native PhysX termination. Nonfoot net force is NOT native pair manifold.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_v071_native_event_ledger import DEFAULT_SOURCE,DEFAULT_ROOT,sha256,verify_manifest,authenticate_censored_native_gpu_logs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-dir",type=Path,default=DEFAULT_SOURCE)
    ap.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    a=ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c=json.loads((a.root/"configs/g1_physx_readonly_evidence_ledger_v071.json").read_text())
    verify_manifest(a.source_dir,c)
    authenticate_censored_native_gpu_logs(a.source_dir,c)
    out=a.root/"media";out.mkdir(parents=True,exist_ok=True)
    policies=("v32","synth","reverse")
    palette=("black","#007da5","#b33a31")
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True,constrained_layout=True)
    for k,ax in zip("abc",axs):
        for policy,color in zip(policies,palette):
            p=a.source_dir/f"v067_new_stance_{k}_{policy}.json"
            raw=json.loads(p.read_text())
            ts=[r["time_s"] for r in raw["traces"][0]]
            force=[r["nonfoot_contact_N"] for r in raw["traces"][0]]
            ax.plot(ts,force,color=color,linewidth=1.25,label=f"{policy} (end {ts[-1]:.2f}s)")
            if ts[-1]<13.995:
                ax.plot([ts[-1]],[force[-1]],marker="x",color=color,markersize=8)
        ax.axhline(20,linestyle=":",color="gray",linewidth=1,label="20N original early-stop threshold")
        ax.set_title(f"original G1 v67 ball/reset context {k} — measured native nonfoot net force")
        ax.set_ylabel("net N (not pair)")
        ax.set_ylim(bottom=-5)
        ax.legend(loc="upper right",fontsize=8)
    axs[-1].set_xlabel("actual PhysX time s (nothing drawn after native early termination)")
    p=out/"v071_original_200hz_nonfoot_censored.png"
    fig.savefig(p,dpi=145);plt.close(fig)
    # The time window is physical strike-centered on native G1 measured contact.
    fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True,constrained_layout=True)
    for policy,color in zip(policies,palette):
        raw=json.loads((a.source_dir/f"v067_new_stance_b_{policy}.json").read_text())
        rec=raw["records"][0]
        ct=rec.get("contact_time_s")
        if ct is None:continue
        rows=[x for x in raw["traces"][0] if ct-.4<=x["time_s"]<=ct+1.25]
        rel=[x["time_s"]-ct for x in rows]
        a29=[sum(abs(v) for v in x["joint_vel"])/29 for x in rows]
        com=[x["whole_body_com_xyz"][2] for x in rows]
        supports=[x["support_foot_force_N"]["left"]-x["support_foot_force_N"]["right"] for x in rows]
        axs[0].plot(rel,a29,color=color,label=policy)
        axs[1].plot(rel,com,color=color,label=policy)
        axs[2].plot(rel,supports,color=color,label=policy)
    for ax,label in zip(axs,
        ("measured mean abs original 29DoF qdot rad/s (not human skill)",
         "whole body simulated COM z [m]",
         "measured left-minus-right support force [N] (not support polygon)")):
        ax.set_ylabel(label,fontsize=9)
        ax.axvline(0,linestyle="--",color="gray",linewidth=1)
        ax.grid(alpha=.25)
        ax.legend(loc="upper right",fontsize=9)
    axs[-1].set_xlabel("physical seconds relative to original G1 actual ball contact (context b)")
    p2=out/"v071_context_b_measured_29dof_com_support.png"
    fig.savefig(p2,dpi=145);plt.close(fig)
    print("V071_LOCAL_FIGURES",p,p2,sep="\n")
if __name__=="__main__":
    main()
