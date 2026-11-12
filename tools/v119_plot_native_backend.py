#!/usr/bin/env python3
"""Figure from six SHA-pinned original 200Hz CPU/CUDA physics trajectories."""
from __future__ import annotations
import json
from pathlib import Path
from v119_independent_backend_audit import score
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parents[1]
    master=json.loads((root/"reports/v119_native_backend_master.audit.json").read_text())
    evidence=score(root,master)
    if evidence["actual_physx_runs"]!=6:raise ValueError("NOT_ALL_PHYSX_CASES")
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for i,dev in enumerate(("cpu","cuda")):
        for typ in ("positive_off","positive_on","separated_on"):
            d=json.loads((root/"reports"/("v119_native_"+dev+"_"+typ+".json")).read_text())
            t=[frame[0]*.005 for frame in d["runtime_z_vz"]]
            z=[frame[1] for frame in d["runtime_z_vz"]]
            axs[i].plot(t,z,label=dev+" "+typ,linewidth=1.3)
        axs[i].set_ylabel("True ball center z (m)")
        axs[i].set_title(dev+" backend, 200 Hz: collided positives vs separated negative")
        axs[i].legend(loc="best");axs[i].grid(alpha=.25)
    axs[1].set_xlabel("Measured simulation time (s)")
    fig.suptitle("Native collider reports: no certified pairs despite actual collision dynamics")
    fig.tight_layout()
    path=root/"reports/v119_native_backend_real_200hz_negative.png"
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=150);plt.close(fig)
    print("SAVED_REAL_PHYSICS_FIGURE",path)
if __name__=="__main__":main()
