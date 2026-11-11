#!/usr/bin/env python3
"""Post-registered physical figure renderer from SHA-audited v64 G1 physics.

No 120Hz CMU contact or elite power inference; all negative cases included.
Raw JSON/PNG media remain local by .gitignore; source files are GitHub tracked.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"reports/v064_independent_physics_audit.json"
OUT=ROOT/"media"
def main():
 audit=json.loads(SRC.read_text())
 if audit["actual_original_G1_PhysX_attempts"]!=12:raise RuntimeError("Cannot omit failed case")
 OUT.mkdir(exist_ok=True,parents=True)
 policies=["v32","human_ordered","human_reversed"]
 ctx=[x["name"] for x in json.loads((ROOT/"configs/g1_human_phase_batch_v064.json").read_text())["groups"]]
 rows=audit["all_episodes"]
 fig,ax=plt.subplots(figsize=(12,5))
 for j,p in enumerate(policies):
  values=[]
  for c in ctx:
   r=next(x for x in rows if x["context"]==c and x["policy"]==p)
   values.append(r.get("original_instep_impulse_fraction") if r.get("original_instep_impulse_fraction") is not None else np.nan)
  x=np.arange(len(ctx))+0.26*(j-1)
  ax.bar(x,values,.24,label=p)
 for i,c in enumerate(ctx):
  for j,p in enumerate(policies):
   r=next(x for x in rows if x["context"]==c and x["policy"]==p)
   ax.text(i+.26*(j-1),min(1.03,(r.get("original_instep_impulse_fraction") or 0)+.015),
    ("PASS" if r.get("strict_success") else "FAIL") if r.get("strict_success") is not None else "NO PHYS",
    ha="center",va="bottom",fontsize=8,rotation=90)
 ax.axhline(.95,ls="--",lw=1,label="strict instep impulse >=95%")
 ax.set(xlim=(-.45,3.5),ylim=(0,1.25),
  xticks=np.arange(4),xticklabels=ctx,ylabel="Original true G1 instep impulse fraction",
  title="v64 12 frozen PhysX attempts (not human strike-contact ground truth)")
 ax.legend(loc="lower left",fontsize=8)
 fig.tight_layout();one=OUT/"v064_original_g1_contact_instep_all_cases.png";fig.savefig(one,dpi=145);plt.close(fig)
 # Figure 2: actual full 29DoF physical difference, not proposed trajectory.
 valid=[a for a in audit["preregistered_pair_results"] if a.get("valid_real_PhysX_pair")]
 fig,ax=plt.subplots(figsize=(12,4.5))
 for policy in ("human_ordered","human_reversed"):
  sub=[x for x in valid if x["policy"]==policy]
  x=np.arange(len(sub))+(0 if policy=="human_ordered" else .34)
  y=[r["actual_measured_g1_29dof_joint_position_difference_max_rad"] for r in sub]
  ax.bar(x,y,.32,label=policy)
 ax.set(xticks=np.arange(len(ctx))+.17,xticklabels=ctx,
  ylabel="Max measured G1 29DoF Δq vs original v32 (rad)",
  title="Actual 200Hz articulated G1 physical divergence (zero action is a negative result)")
 ax.legend()
 fig.tight_layout();two=OUT/"v064_measured_robot_phase_effect_vs_v32.png";fig.savefig(two,dpi=145);plt.close(fig)
 print("V064_REAL_PHYS_FIGURES",one,two)
if __name__=="__main__":main()
