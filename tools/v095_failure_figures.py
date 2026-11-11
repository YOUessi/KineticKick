#!/usr/bin/env python3
"""Reproducible plots ONLY from SHA-validated read-only G1 and box physics."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v095_independent_evidence as audit
def writeplot(path,fig):
 if path.exists():raise FileExistsError("WRITE_ONCE_FIGURE "+str(path))
 fig.savefig(path,dpi=145,bbox_inches="tight")
 plt.close(fig)
 return dict(file=str(path),sha256=audit.sha(path))
def figures():
 a=json.loads((ROOT/"reports/v095_independent_evidence.audit.json").read_text())
 b=json.loads((ROOT/"reports/v095_29dof_generated_holdout.audit.json").read_text())
 outs={}
 fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
 for case in ("gpu_pose_off","gpu_tensor_on","gpu_both_on","gpu_separated_on"):
  raw=json.loads((ROOT/"reports"/("v095_"+case+".json")).read_text())
  if audit.sha(ROOT/"reports"/("v095_"+case+".json"))!=a["individual_write_once_raw_source_sha256"][case]["raw_sha"]:
   raise ValueError("CONTACT_PLOT_SOURCE_DRIFT")
  t=np.asarray(raw["runtime_z_vz"],float)
  ax[0].plot(t[:,0]*.005,t[:,1],label=case)
  points=raw["tensor_contact_points"]
  if points:
   ax[1].scatter([p["step"]*.005 for p in points],
                 [p["force_magnitude_N"] for p in points],s=12,label=case)
 ax[0].set_ylabel("Real sphere height (m)")
 ax[1].set_ylabel("CUDA filtered pair force (N)")
 ax[1].set_xlabel("True PhysX simulation time (s)")
 for x in ax:x.legend(loc="best")
 ax[0].set_title("Isolated physics positive/negative; stage paths are NOT native shape manifold")
 outs["contact_positive_negative"]=writeplot(ROOT/"reports/v095_calibration_contact.png",fig)
 fig,axs=plt.subplots(4,1,figsize=(12,10),sharex=True)
 for j,case in enumerate(("E_v32","E_cmu02","F_v32","F_cmu02")):
  entry,trace=audit.orig.inspect(case)
  t,dq,limit,loads,com,nonfoot,ff,impact=trace
  x=axs[j]
  x.plot(t,nonfoot,label="non-foot net force N")
  x.axhline(20,linestyle="--",label="nonfoot 20N hard gate")
  if impact is not None:x.axvline(impact,linestyle=":",label="true foot-ball time")
  x.set_ylabel(case+"\nForce (N)")
  x.legend(loc="upper right")
  if entry["independent_gates"]["strict_pass"]:raise ValueError("SCORER_FALSE_PASS")
 axs[-1].set_xlabel("Real original 200Hz G1 physical time (s); F ends early, no imagined recovery")
 fig.suptitle("Four original G1 heldout failures; no learned model actuated")
 outs["g1_holdout_nonfoot"]=writeplot(ROOT/"reports/v095_real_g1_nonfoot_failures.png",fig)
 groups=("state_only","state_and_support","shuffled_support_negative","freeze_arms","strike_leg_only","time_reversed")
 cases=("E_v32","E_cmu02","F_v32","F_cmu02")
 values=np.array([[b["holdout"][k]["metrics"][g]["mean_abs_29_qdot_normalized"]
                   for g in groups] for k in cases])
 fig,ax=plt.subplots(figsize=(11,6))
 xx=np.arange(4)
 for j,g in enumerate(groups):
  ax.bar(xx+(j-2.5)*.13,values[:,j],width=.125,label=g)
 ax.set_xticks(xx,cases)
 ax.set_ylabel("50Hz 0.4s generated 29DoF velocity error / original joint speed limit")
 ax.set_title("Offline robotic G1 imitation ONLY (not human motion and not new physical shots)")
 ax.legend(ncol=2,fontsize=8)
 outs["generator_ablation"]=writeplot(ROOT/"reports/v095_generator_ablation.png",fig)
 print("V095_FIGURES",json.dumps(outs),flush=True)
 return outs
if __name__=="__main__":
 figures()
