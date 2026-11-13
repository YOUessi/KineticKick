#!/usr/bin/env python3
"""Local-only plots of real URDF *offline numerical* outcomes, not PhysX."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def write_new(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"V158_NO_EVIDENCE_OVERWRITE: {path}")
    with path.open("xb") as stream:
        fig.savefig(stream,format="png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    print(str(path), hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--prefix",required=True,type=Path)
    a=p.parse_args()
    results=json.loads(a.input.read_text())
    phases=["approach","plant","coil","swing","followthrough","recover"]
    rows=results["case_rows"]
    nominal=[np.mean([r["tracking_rmse_rad_s2"] for r in rows if r["phase"]==ph and r["effort_condition"]=="urdf_effort"]) for ph in phases]
    relaxed=[np.mean([r["tracking_rmse_rad_s2"] for r in rows if r["phase"]==ph and r["effort_condition"]=="counterfactual_100x_effort"]) for ph in phases]
    x=np.arange(len(phases))
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(x-.19,nominal,.38,label="Actual URDF effort ceiling")
    ax.bar(x+.19,relaxed,.38,label="100x effort counterfactual",alpha=.7)
    ax.set_xticks(x,phases,rotation=20)
    ax.set_ylabel("29-DoF acceleration tracking RMSE (rad/s²)")
    ax.set_title("v158: original URDF offline 35D, assumed stance\nNot real PhysX strike success or learned human motion")
    ax.legend()
    write_new(fig,str(a.prefix)+"_phase_actuation_ablation.png")
    fig,ax=plt.subplots(figsize=(10,5))
    z=np.array([r["maximum_original_effort_ratio"] for r in rows if r["effort_condition"]=="urdf_effort"])
    ax.scatter(np.arange(len(z)),z,s=18)
    ax.axhline(1.0,color="red",linestyle="--",label="Original URDF effort cap")
    ax.set_xlabel("Frozen source fixture index (48 hypothetical states)")
    ax.set_ylabel("Max |tau| / URDF effort ceiling")
    ax.set_ylim(0,max(1.1,float(z.max())+.05))
    ax.set_title("v158: 29-joint actuation headroom of offline QP\nNeither real controller safety nor measured torque")
    ax.legend()
    write_new(fig,str(a.prefix)+"_effort_limit_headroom.png")

if __name__=="__main__":main()
