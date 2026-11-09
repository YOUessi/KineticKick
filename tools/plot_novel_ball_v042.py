#!/usr/bin/env python3
"""v42 observed PhysX paired novel continuous parameter conditions.

Chart contains no inferred human-style/naturalness scores.
"""
from pathlib import Path
import json,math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"reports/v042_paired_novel_ball_audit.json"

def main():
 z=json.loads(DATA.read_text())
 if z["n_cases"]!=6:raise ValueError("Missing prespecified six cases")
 cases=z["trials"]
 labels=[c["id"] for c in cases]
 a=np.asarray([c["original_v32"]["ball_outgoing_xy_mps"] or 0. for c in cases])
 b=np.asarray([c["style_v41"]["ball_outgoing_xy_mps"] or 0. for c in cases])
 p=np.arange(6)
 fig,ax=plt.subplots(figsize=(12.2,5.8))
 ax.bar(p-.18,a,.36,label="Original v32, measured ball speed")
 ax.bar(p+.18,b,.36,label="Frozen v41 + weak human-pose reference")
 for i,c in enumerate(cases):
  old=c["original_v32"]["strict_functional_pass"]
  new=c["style_v41"]["strict_functional_pass"]
  on=c["style_v41"]["style_applied_decisions"]
  meta=f"ball: {c['ball_offset_xy_m'][0]:+.3f}, {c['ball_offset_xy_m'][1]:+.3f}m"
  meta+=f"\nΔvy: {c['root_velocity_y_delta_mps']:+.3f}m/s"
  ax.text(i,.15,meta,ha="center",va="bottom",fontsize=8)
  ax.text(i,max(a[i],b[i])+.1,
    f"old {'PASS' if old else 'FAIL'} / style {'PASS' if new else 'FAIL'}\nstyle applied {on} decisions",
    ha="center",fontsize=8)
 ax.set_xticks(p,labels)
 ax.set_ylabel("Observed outgoing ball horizontal speed (m/s), 0 if no measured exit")
 ax.set_ylim(0,max(6.4,max(a.max(),b.max())+1.4))
 ax.set_title("KineticKick v0.1.42 | Six prespecified new ball-parameter triples, paired true G1 PhysX")
 ax.grid(axis="y",alpha=.25)
 ax.legend(loc="upper left",fontsize=9)
 fig.text(.015,.006,
  "Six new exact parameter triples, fixed before outcomes; one deterministic G1 PhysX run per model and condition. "
  "Not population reliability, certified human instep biomechanics, high-power football motion or hardware naturalness.",
  fontsize=8,wrap=True)
 fig.tight_layout(rect=(0,.08,1,1))
 out=ROOT/"media/v042_novel_ball_paired_physx.png"
 out.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(out,dpi=155)
 plt.close(fig)
 print("V042_PAIRED_PLOT",out,out.stat().st_size)
if __name__=="__main__":main()
