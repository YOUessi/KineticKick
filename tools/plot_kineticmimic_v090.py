#!/usr/bin/env python3
"""KineticMimic v90: write-one authentic G1/CMU evidence figure batch.

No new rendering/simulation. The original measured 5ms G1 q/dq, normal
support, contact force, qdot speed fraction and contact events remain true.
Historical source SHA and censors audited by independent v090 code.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from kineticmimic_physics_v090 import load_original_physics
from kineticmimic_loss_v090 import IDS
from kineticmimic_physics_v090 import sha
def plot(config,output_dir):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 c=json.loads(config.read_text())
 audit=json.loads((ROOT/"reports/v090_kineticmimic_independent_audit.json").read_text())
 output_dir.mkdir(parents=True,exist_ok=True)
 plots=[]
 fig,ax=plt.subplots(figsize=(10,5))
 keys=("center_v32","center_cmu02","center_cmu05")
 for key in keys:
  spec=c["source_physx"][key]
  p=Path(spec["path"])
  e=load_original_physics(p,spec["sha256"],key)
  rel=e.times-float(e.first_impact_s)
  mask=(rel>=-.55)&(rel<=.35)
  ax.plot(rel[mask],e.feet_normal_N[mask,0],label=key+" left support")
  # Full line only while genuine real physical 5ms rows exist, no padding.
  if key=="center_v32":ax.plot(rel[mask],e.feet_normal_N[mask,1],linestyle="--",label="v32 right support")
 ax.axvline(0,color="black",linestyle=":",label="Measured G1 foot-ball first contact")
 ax.set_xlabel("Time relative to ACTUAL G1 PhysX first ball contact (s)")
 ax.set_ylabel("Original support-foot ground-filtered normal net force (N)")
 ax.set_title("Measured G1 200Hz support transfer; NOT human GRF")
 ax.legend(loc="upper left",fontsize=8);ax.grid(alpha=.25)
 fig.tight_layout();f=output_dir/"v090_original_G1_support_and_impact.png"
 fig.savefig(f,dpi=130);plt.close(fig)
 plots.append(dict(file=str(f.relative_to(ROOT)),SHA256=sha(f),source="original G1 PhysX 200Hz true foot support"))
 fig,ax=plt.subplots(figsize=(10,5))
 for key in keys:
  spec=c["source_physx"][key]
  e=load_original_physics(Path(spec["path"]),spec["sha256"],key)
  rel=e.times-e.first_impact_s
  region=(rel>=-.4)&(rel<=.2)
  v=np.sqrt(np.mean(e.dq[:,IDS["trunk"]]**2,axis=1))
  u=np.sqrt(np.mean(e.dq[:,IDS["swing"]]**2,axis=1))
  ax.plot(rel[region],v[region],label=key+" torso speed",alpha=.8)
  if key=="center_v32":ax.plot(rel[region],u[region],label="v32 swing-leg speed",linestyle="--")
 ax.axvline(0,color="black",linestyle=":",label="G1 true impact")
 ax.set_xlabel("Time relative to original 200Hz contact (s)")
 ax.set_ylabel("Joint RMS measured angular velocity (rad/s)")
 ax.set_title("G1 temporal limb coordination; NOT measured human muscle power")
 ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
 f=output_dir/"v090_original_G1_trunk_vs_swing_29dof.png";fig.savefig(f,dpi=130);plt.close(fig)
 plots.append(dict(file=str(f.relative_to(ROOT)),SHA256=sha(f),source="original G1 all 29 measured joint rates"))
 result=ROOT/c["output_dir"]/"kineticmimic_contact_training_audit.json"
 x=json.loads(result.read_text())
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 names=("ordered","history_shuffle_negative")
 stance=[x["fitted_contact_heads"][k]["test"]["predicted_stance_accuracy"] for k in names]
 hazard=[x["fitted_contact_heads"][k]["test"]["hazard_next_100ms_F1"]["f1"] for k in names]
 axs[0].bar(range(3),[stance[0],stance[1],x["nonlearned_test_baselines"]["stance_persistence_accuracy"]])
 axs[0].set_xticks(range(3),["sequence","shuffled","last stance"]);axs[0].set_ylim(0,1)
 axs[0].set_ylabel("Next 100ms 4-state stance accuracy")
 axs[1].bar(range(3),[hazard[0],hazard[1],0.])
 axs[1].set_xticks(range(3),["sequence","shuffled","never-contact"]);axs[1].set_ylim(0,1)
 axs[1].set_ylabel("Next 100ms FIRST foot-ball impact F1")
 fig.suptitle("Hist. PhysX CENTER tests: contact event F1=0 (negative result)")
 for q in axs:q.grid(axis="y",alpha=.25)
 fig.tight_layout();f=output_dir/"v090_physx_support_head_f1_failure.png"
 fig.savefig(f,dpi=130);plt.close(fig)
 plots.append(dict(file=str(f.relative_to(ROOT)),SHA256=sha(f),source="Original 9 PhysX trajectories; 2 frozen CPU contact head checkpoints"))
 meta=ROOT/"reports/v090_figures_audit.json"
 if meta.exists():raise FileExistsError("Don't rewrite G1 source-derived plots")
 meta.write_text(json.dumps(dict(version="v0.1.90",all_original_sources_SHA_checked=True,
   plots=plots, original_sources_unchanged=True,new_PhysX=0),indent=2)+"\n")
 print("KINETICMIMIC_V90_REAL_SOURCE_PLOTS",json.dumps(plots),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--config",type=Path,default=ROOT/"configs/kineticmimic_v090_study.json")
 p.add_argument("--output-dir",type=Path,default=ROOT/"media")
 a=p.parse_args()
 plot(a.config,a.output_dir)
