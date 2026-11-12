#!/usr/bin/env python3
"""Generate a source-hashed 200Hz G1 failure figure (data stay on Tang)."""
import hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for s in iter(lambda:f.read(1048576),b""):h.update(s)
    return h.hexdigest()
def main():
    master=ROOT/"reports/v107_original_g1_batch.audit.json"
    d=json.loads(master.read_text())
    rows=[]
    for r in d["runs"]:
        if not r["certified_original_G1_cuda_200Hz"]:continue
        f=ROOT/"reports"/("v107_"+r["name"]+".json")
        if sha(f)!=r["raw_sha256"]:raise ValueError("RAW_SHA_DRIFT")
        z=json.loads(f.read_text())
        t=z["traces"][0]
        if any(len(x["joint_vel"])!=29 for x in t):raise ValueError("NOT_SOURCE29")
        rows.append((r,z,t))
    if not rows:raise ValueError("NO_VERIFIED_G1")
    fig,ax=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for r,z,t in rows:
        name=r["name"]
        time=np.asarray([x["time_s"] for x in t],float)
        lim=np.asarray(z["joint_velocity_limits"],float).reshape(-1)
        dq=np.asarray([x["joint_vel"] for x in t],float)
        support=np.asarray([[x["support_foot_force_N"]["left"],x["support_foot_force_N"]["right"]] for x in t],float)
        nonfoot=np.asarray([x["nonfoot_contact_N"] for x in t],float)
        ball=np.asarray([x["ball_velocity"] for x in t],float)
        ax[0,0].plot(time,np.max(np.abs(dq)/lim,axis=1),label=name,linewidth=.9)
        ax[0,1].plot(time,support[:,0],label=name+" left",linewidth=.8)
        ax[0,1].plot(time,support[:,1],label=name+" right",linewidth=.8,linestyle="--")
        ax[1,0].plot(time,nonfoot,label=name,linewidth=.8)
        ax[1,1].plot(time,np.linalg.norm(ball,axis=1),label=name,linewidth=.9)
    ax[0,0].axhline(1.,color="black",linewidth=.8,linestyle=":")
    ax[1,0].axhline(20.,color="black",linewidth=.8,linestyle=":")
    ax[0,0].set_title("Measured original 29 joint |dq|/limit maximum")
    ax[0,1].set_title("Measured foot support force proxy (N)")
    ax[1,0].set_title("Measured nonfoot net force proxy (N)")
    ax[1,1].set_title("Measured ball center speed (m/s)")
    for a in ax.flat:
        a.set_xlabel("200Hz true physics time (s)")
        a.grid(alpha=.2)
        a.legend(loc="best",fontsize=6,ncol=2)
    fig.suptitle("v107 frozen original G1: actual 200Hz, no post-censor fabrication; tensor is not native manifold")
    fig.tight_layout()
    f=ROOT/"reports/v107_failure_traces.png"
    if f.exists():raise FileExistsError("WRITE_ONCE_FAILURE_FIGURE")
    fig.savefig(f,dpi=135);plt.close(fig)
    print("V107_FAILURE_FIGURE",f,"actual certified episodes",len(rows),"source audit sha",sha(master))
if __name__=="__main__":main()
