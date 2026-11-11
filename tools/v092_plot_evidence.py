#!/usr/bin/env python3
"""Visualize frozen human geometric candidate confidence and G1 censored negatives."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot(data,out):
 clips=data["human"]
 assert len(clips)==6 and len(data["original_G1"])==4
 fig,ax=plt.subplots(figsize=(10,5))
 names=[x["file"].replace(".c3d","") for x in clips]
 distances=[x.get("candidate_right_toe_ball_marker_centroid_distance_m",float("nan")) for x in clips]
 bars=ax.bar(names,distances)
 for b,x in zip(bars,clips):
  if not x.get("phase_learning_eligible"):b.set_hatch("///")
 ax.axhline(.30,ls="--",label="Geometric candidate cut (NOT real contact)",color="black")
 ax.set_ylabel("Right toe to soccer-ball marker-centroid minimum (m)")
 ax.set_title("Original 120 Hz CMU C3D: optical proximity only, not GRF/instep truth")
 ax.legend();fig.tight_layout();fig.savefig(out[0],dpi=140);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5))
 for i,r in enumerate(data["original_G1"]):
  dur=r["real_physics_frames"]*.005
  ax.barh(i,dur)
  ax.text(dur+.05,i,"CENSORED" if not r["full_14s_recovery_observed"] else "14s",va="center")
 ax.set_yticks(list(range(4)),[x["case"] for x in data["original_G1"]])
 ax.axvline(14,color="black",ls="--",label="Required physical recovery")
 ax.set_xlim(0,16.5);ax.set_xlabel("True original 200 Hz PhysX exposure (s)")
 ax.set_title("Original v91 E/F new-ball heldout: strict 0/4, no recovery fabrication")
 ax.legend();fig.tight_layout();fig.savefig(out[1],dpi=140);plt.close(fig)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--input",type=Path,required=True)
 p.add_argument("--out-human",type=Path,required=True)
 p.add_argument("--out-g1",type=Path,required=True)
 a=p.parse_args()
 if any(x.exists() for x in (a.out_human,a.out_g1)):p.error("WRITE_ONCE_PLOT")
 plot(json.loads(a.input.read_text()),(a.out_human,a.out_g1))
 print("V092_IMAGES",a.out_human,a.out_g1,flush=True)
