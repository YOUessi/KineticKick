#!/usr/bin/env python3
"""Local-only v84 genuine 200Hz original-29DoF G1 plots, never synthetic frames."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"reports/v084_independent_200hz.audit.json"

def plots():
    result=json.loads(SRC.read_text())
    created=[]
    for ep in result["episodes"]:
        name=ep["name"];p=ROOT/"reports"/("v084_"+name+".json")
        if hashlib.sha256(p.read_bytes()).hexdigest()!=ep["raw_sha256"]:
            raise ValueError("RAW_SHA_CHANGED:"+name)
        d=json.loads(p.read_text());rows=d["traces"][0]
        t=[r["time_s"] for r in rows]
        left=[r["support_foot_force_N"]["left"] for r in rows]
        right=[r["support_foot_force_N"]["right"] for r in rows]
        comz=[r["whole_body_com_xyz"][2] for r in rows]
        root=[r["root_xyz"][2] for r in rows]
        limit=d["joint_velocity_limits"][0]
        qdot=[max(abs(x)/lim for x,lim in zip(r["joint_vel"],limit)) for r in rows]
        nonfoot=[r["nonfoot_contact_N"] for r in rows]
        ball=[math.hypot(*r["ball_velocity"][:2]) for r in rows]
        fig,axes=plt.subplots(4,1,sharex=True,figsize=(12,9.5),constrained_layout=True)
        axes[0].plot(t,left,label="left ground-filtered support N")
        axes[0].plot(t,right,label="right ground-filtered support N")
        axes[0].legend(fontsize=7,loc="upper right")
        axes[0].set_ylabel("foot support (N)")
        axes[1].plot(t,comz,label="wholebody COM z m")
        axes[1].plot(t,root,label="root z m")
        axes[1].legend(fontsize=7,loc="upper right")
        axes[1].set_ylabel("height (m)")
        axes[2].plot(t,qdot,label="max real |qdot| / original 29 actuator limits")
        axes[2].axhline(1,color="crimson",ls="--",label="hard gate")
        axes[2].legend(fontsize=7,loc="upper right")
        axes[2].set_ylabel("qdot ratio")
        axes[3].plot(t,nonfoot,label="nonfoot net force N")
        axes[3].plot(t,ball,label="ball planar speed m/s")
        axes[3].axhline(20,color="crimson",ls="--",label="nonfoot threshold N")
        axes[3].legend(fontsize=7,loc="upper right")
        axes[3].set_xlabel("true PhysX time (s)")
        axes[3].set_ylabel("force / speed")
        contact=ep["contact_time_s"]
        if isinstance(contact,(float,int)):
            for ax in axes:
                ax.axvline(contact,color="gray",ls=":",lw=1.1)
        status="PASS" if ep["strict_hard_gate_pass"] else "FAIL ("+",".join(ep["failed_gates"])+")"
        fig.suptitle("v084 original G1 200Hz  •  "+name+"  •  "+status+
                     "\nNot human motion truth; no inferred actor-pair collision from net forces",fontsize=11)
        out=ROOT/"media"/("v084_"+name+"_fullbody_real_physx.png")
        if out.exists() or out.is_symlink():raise FileExistsError(out)
        with out.open("xb") as fp:fig.savefig(fp,format="png",dpi=125)
        plt.close(fig)
        created.append(str(out))
    manifest=ROOT/"reports/v084_figures_local.audit.json"
    if manifest.exists():raise FileExistsError(manifest)
    manifest.write_text(json.dumps({"local_only":True,"images":created,
        "true_original_cuda_physx_episodes":len(created),"human_supervision":False},indent=2)+"\n")
    print("V084_REAL_PLOTS",len(created))
if __name__=="__main__":plots()
