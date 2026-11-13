#!/usr/bin/env python3
"""v156 URDF collision-source negative: toy corner forces outside source envelope.

Figures are only Tang-local outputs. URDF collision box is an overapprox,
NOT the actual physical support polygon and NOT a ground contact measurement.
"""
from __future__ import annotations
import argparse
import io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from v156_urdf_collision_footprint_audit import footprint_audit
from v156_independent_runtime_and_history_gate import independent_real_history_audit

def write_new(fig, path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    mem=io.BytesIO()
    fig.savefig(mem,format="png",dpi=150,bbox_inches="tight")
    with path.open("xb") as f:f.write(mem.getvalue())
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--geometry",type=Path,required=True)
    ap.add_argument("--history",type=Path,required=True)
    args=ap.parse_args()
    src=footprint_audit()
    foot=src["feet"]["left_ankle_roll_link"]
    lo=foot["enclosing_min_xyz_m"];hi=foot["enclosing_max_xyz_m"]
    fig,ax=plt.subplots(figsize=(7,5))
    ax.add_patch(Rectangle((lo[0],lo[1]),hi[0]-lo[0],hi[1]-lo[1],
                  fill=False,linewidth=2,label="URDF collision cylinder outer XY envelope"))
    for collision in foot["sources"]:
        x,y=collision["center_xyz_m"][:2]
        half=collision["bbox_halfwidth_m"]
        ax.add_patch(Rectangle((x-half[0],y-half[1]),half[0]*2,half[1]*2,
                     fill=False,linewidth=.7,alpha=.5))
    for x,y in foot["v155_toy_corners_xy_m"]:
        ax.scatter(x,y,marker="x",s=100)
    ax.set_title("Source-data falsification: v155 toy corners vs original URDF")
    ax.set_xlabel("Ankle link local X (m)");ax.set_ylabel("Ankle link local Y (m)")
    ax.axis("equal");ax.grid(alpha=.2)
    ax.legend(loc="lower center",bbox_to_anchor=(.5,-.25))
    write_new(fig,args.geometry)
    h=independent_real_history_audit()
    fig,ax=plt.subplots(figsize=(8,4))
    names=list(h["runs"])
    rows=[h["runs"][n]["real_200hz_rows"] for n in names]
    accepted=[h["runs"][n]["admissible_full_35d_observed_runtime_steps"] for n in names]
    ax.bar(names,rows,label="Real original 200 Hz q29-only states")
    ax.bar(names,accepted,label="Accepted exact-root 35D states")
    ax.set_title("Original PhysX observations lack quaternion + 6D root twist")
    ax.set_ylabel("Observed steps, not new PhysX simulations")
    ax.legend();ax.grid(axis="y",alpha=.2)
    write_new(fig,args.history)
    print({"geometry":str(args.geometry),"history":str(args.history),
           "outside_toy_corners":foot["v155_unmeasured_corner_count_outside_urdf_collision_outer_bbox"],
           "true_original_rows":h["actual_original_200hz_states"],
           "accepted_full35_runtime_rows":h["valid_35d_historical_pin_reconstruction_steps"]})

if __name__=="__main__":
    main()
