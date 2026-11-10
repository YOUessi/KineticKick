#!/usr/bin/env python3
"""CMU optical 120Hz soccer vs G1 physical 29DoF semantic pose coordination.

All six panels show time RELATIVE TO DIFFERENT EVENT DEFINITIONS: human is
inferred ball-marker speed launch, robot is PhysX measured foot-ball impact.
DO NOT use apparent agreement to conclude biomechanical contact-time sync,
human/robot anatomical equivalence, naturalness or motor policy transfer.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/cmu_g1_cross_morph_v051.json"
def main():
 cfg=json.loads((ROOT/STUDY).read_text())
 base=ROOT/cfg["output_dir"]
 meta=json.loads((base/"cross_morphology_audit.json").read_text())
 with np.load(base/"cmu_g1_6d_exploratory_timeline.npz",allow_pickle=False) as x:
  time=x["relative_event_time_s"].astype(float)
  names=x["semantic_6_channels"].tolist()
  signals={k:x[k].astype(float) for k in
      ("10_02_cos","10_05_cos","v32_frozen_cos","v48_learned_cos")}
 if len(time)!=20 or len(meta["comparisons"])!=4:
  raise ValueError("Original 2x2 CMU/G1 temporal alignment incomplete")
 fig,axes=plt.subplots(3,2,figsize=(13,10.4),sharex=True)
 groups=(("10_02_cos","Human 10_02, 120Hz C3D"),
         ("10_05_cos","Human 10_05, 120Hz C3D"),
         ("v32_frozen_cos","G1 frozen v32, true PhysX"),
         ("v48_learned_cos","G1 learned v48, true PhysX"))
 for i,ax in enumerate(axes.flat):
  for k,name in groups:
   ax.plot(time,signals[k][:,i],label=name,linewidth=1.3,
      linestyle="--" if k=="v48_learned_cos" else "-")
  ax.set_ylim(-1.07,1.07)
  ax.set_ylabel("3D segment cosine (-1 to 1)")
  ax.set_title(names[i])
  ax.grid(alpha=.2)
  if i>=4:ax.set_xlabel("Seconds before human LAUNCH proxy vs G1 physical impact")
 axes[0,0].legend(loc="upper left",fontsize=7)
 fig.suptitle("KineticKick v0.1.51 | Real human optical soccer coordination vs G1 URDF joint/link proxy")
 fig.text(.01,.012,
  "CMU t=0 is inferred outgoing ball-marker movement, NOT verified instep collision. G1 t=0 is real foot-ball PhysX contact. Skeleton joint centers differ anatomically; curves do not prove professional human motion imitation, power or naturalness.",
  fontsize=8)
 fig.tight_layout(rect=(0,.055,1,.95))
 dest=ROOT/"media/v051_real_human_vs_G1_six_segment_cosines.png"
 dest.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(dest,dpi=155)
 plt.close(fig)
 print("V051_REAL_HUMAN_G1_SEMANTIC_TIMELINE_PLOT",dest,dest.stat().st_size)
 # Relative action amplitudes as research diagnostic, not expert score.
 fig,ax=plt.subplots(figsize=(8.5,4.5))
 rms=[]
 labels=[]
 for k,name in groups:
  v=signals[k]
  rms.append(float(np.sqrt(np.mean((v-v[:1])**2))))
  labels.append(name)
 ax.bar(range(4),rms)
 for i,x in enumerate(rms):
  ax.text(i,x+.006,f"{x:.3f}",ha="center",fontsize=9)
 ax.set_xticks(range(4),labels,rotation=16,ha="right")
 ax.set_ylim(0,max(rms)+.07)
 ax.set_ylabel("Within-own-body first-frame-relative 6-cosine RMS")
 ax.set_title("Temporal magnitude only: no human anatomical/collision reference alignment")
 ax.grid(axis="y",alpha=.25)
 fig.tight_layout()
 dest2=ROOT/"media/v051_human_G1_relative_motion_amplitudes.png"
 fig.savefig(dest2,dpi=150)
 plt.close(fig)
 print("V051_HUMAN_G1_RELATIVE_BODY_CHANGE_PLOT",dest2,dest2.stat().st_size)
if __name__=="__main__":main()
