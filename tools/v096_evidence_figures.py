"""Read-only plots from write-once actual v96 PhysX and original v91 G1 200Hz.
Never interpolate past early termination or fabricate contact manifold truth."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v091_independent_dynamics as g1
g1.ROOT=Path("/home/you/projects/kinetickick-wt-v091")
def hashfile(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def plot_calibration():
 path=ROOT/"reports/v096_real_contact_calibration.png"
 if path.exists():raise FileExistsError("WRITE_ONCE_FIGURE")
 matrix=json.loads((ROOT/"configs/v096_frozen_matrix.json").read_text())["matrix"]
 fig,ax=plt.subplots(2,1,figsize=(12,9),sharex=True)
 for m in matrix:
  p=ROOT/"reports"/("v096_"+m["case"]+".json")
  raw=json.loads(p.read_text())
  t=np.array([r[0]*.005 for r in raw["runtime_z_vz"]])
  z=np.array([r[1] for r in raw["runtime_z_vz"]])
  ax[0].plot(t,z,label=m["case"])
  if raw["tensor_contact_points"]:
   ct=np.array([c["step"]*.005 for c in raw["tensor_contact_points"]])
   fn=np.array([c["force_magnitude_N"] for c in raw["tensor_contact_points"]])
   ax[1].scatter(ct,fn,s=12,label=m["case"]+" tensor force")
 ax[0].set_ylabel("Real sphere center height (m)")
 ax[1].set_ylabel("Observed tensor contact force (N)")
 ax[1].set_xlabel("Actual 200Hz PhysX time (s)")
 ax[0].legend(fontsize=7,ncol=3)
 ax[1].legend(fontsize=9)
 ax[0].set_title("v96 isolated PhysX, contact reporters do not change saved trajectory; not G1 shooting")
 ax[1].set_title("Native manifold remains uncalibrated despite positive tensor filtered-contact control")
 fig.tight_layout();fig.savefig(path,dpi=125);plt.close(fig)
 return {"local_path":str(path),"sha256":hashfile(path)}
def plot_original_g1():
 path=ROOT/"reports/v096_original_G1_hard_failures.png"
 if path.exists():raise FileExistsError("WRITE_ONCE_FIGURE")
 fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
 for case in g1.CASES:
  rec,(t,vel,lim,loads,com,nonfoot,footforce,impact)=g1.inspect(case)
  rel=t-impact
  mask=(rel>=-.4)&(rel<=1.25)
  if not np.any(mask):raise ValueError("NO_REAL_WINDOW")
  axes[0].plot(rel[mask],loads[mask,0],label=case+" left")
  axes[1].plot(rel[mask],loads[mask,1],label=case+" right")
  axes[2].plot(rel[mask],np.max(np.abs(vel[mask])/lim,axis=1),label=case)
  axes[3].plot(rel[mask],nonfoot[mask],label=case)
 axes[0].set_ylabel("Original G1 left support N")
 axes[1].set_ylabel("Original G1 right support N")
 axes[2].set_ylabel("Max 29 qdot / limit")
 axes[3].set_ylabel("Nonfoot net force N")
 axes[2].axhline(1,linestyle="--",label="qdot hard gate")
 axes[3].axhline(20,linestyle="--",label="nonfoot termination")
 for a in axes:
  a.axvline(0,linestyle=":",alpha=.7)
  a.legend(fontsize=8,ncol=2)
 axes[-1].set_xlabel("Measured original G1 seconds relative to actual ball contact")
 fig.suptitle("True original 200Hz G1 historical negative cases E/F. Truncated lines are censored, not recovered.")
 fig.tight_layout();fig.savefig(path,dpi=125);plt.close(fig)
 return {"local_path":str(path),"sha256":hashfile(path)}
def main():
 output=ROOT/"reports/v096_figure_sha.audit.json"
 if output.exists():raise FileExistsError("WRITE_ONCE_FIGURE_MANIFEST")
 obj=dict(version="v0.1.96",isolated_physx=plot_calibration(),historical_real_original_G1=plot_original_g1(),
      raw_physics_modified=False,missing_recovery_data_invented=False)
 with output.open("x") as f:json.dump(obj,f,indent=2);f.write("\n")
 print("V096_FIGURES",json.dumps(obj))
if __name__=="__main__":main()

