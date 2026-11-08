#!/usr/bin/env python3
"""Read-only study result verification and research figure, no model retraining.

The single held-out test has just 4 football matches; intervals are descriptive
game-group bootstrap estimates, not evidence of population generalization.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

def analyze():
 config=json.loads((ROOT/"configs/professional_coordination_pilot_v039.json").read_text())
 artifact=ROOT/config["output_dir"]
 result=json.loads((artifact/"result.json").read_text())
 assert result["source_zip_sha256"]==config["original_archive_sha256"]
 assert result["source_data_sha256"]==config["data_sha256"]
 assert sha(config["output_dir"]+"/best_validation_model.pt")==result["best_model_sha256"]
 assert result["first_test_access_after_selection"] is True
 assert result["game_disjoint_train_val_test"]
 for split in ("val","test"):
  m=result["metrics"][split]
  assert set(m)=={"persistence","train_mean_curve","ridge","conditional_vae_prior_mean"}
  assert len(m["ridge"]["game_mse"])==4
 if result["model_selected_step"] not in range(100,config["training"]["steps"]+1,100):
  raise ValueError("Best checkpoint not selected on fixed validation schedule")
 rng=np.random.default_rng(3901)
 sample_metrics=result["metrics"]["test"]
 game_names=sorted(sample_metrics["ridge"]["game_mse"])
 diff=np.asarray([sample_metrics["conditional_vae_prior_mean"]["game_mse"][g]
                 -sample_metrics["ridge"]["game_mse"][g] for g in game_names])
 draws=rng.integers(0,len(diff),(2000,len(diff)))
 means=diff[draws].mean(-1)
 lo,hi=np.quantile(means,[.025,.975])
 result_numeric={
  "study_version":"0.1.39",
  "game_count_heldout":len(game_names),
  "prediction_future_frames":12,
  "observed_history_frames":8,
  "test_mse":{name:val["mse"] for name,val in sample_metrics.items()},
  "validation_mse":{name:val["mse"] for name,val in result["metrics"]["val"].items()},
  "conditional_model_beats_persistence":sample_metrics["conditional_vae_prior_mean"]["mse"]<sample_metrics["persistence"]["mse"],
  "conditional_model_beats_ridge":sample_metrics["conditional_vae_prior_mean"]["mse"]<sample_metrics["ridge"]["mse"],
  "mse_delta_model_minus_ridge":sample_metrics["conditional_vae_prior_mean"]["mse"]-sample_metrics["ridge"]["mse"],
  "game_balanced_mse_delta_model_minus_ridge":float(diff.mean()),
  "per_game_mse_deltas_model_minus_ridge":{g:float(x) for g,x in zip(game_names,diff)},
  "game_cluster_bootstrap_descriptive_95_interval":list(map(float,(lo,hi))),
  "bootstrap_sufficient_for_population_claim":False,
  "frozen_checkpoint":result["best_model_sha256"],
  "source_limits":["Only 108 train,47 validation,45 test shot clips from 12/4/4 football games",
       "20 poses from video lifting; temporal center is shot annotation not contact",
       "Model predicts normalized segment angle cosines, not true 3D robot motion",
       "No professional instep subtype/foot-ball force or valid dynamics labels",
       "Potential repeated players across separate matches; game split is not player-independent",
       "No G1 PhysX transfer or biomechanics naturalness is evaluated"],
 }
 dest=ROOT/"data/research_human_kicks/3dsp_cvpr2024/processed_v039/comparison_summary.json"
 if dest.exists():raise FileExistsError(dest)
 dest.write_text(json.dumps(result_numeric,indent=2,ensure_ascii=False)+"\n")
 title=["Last pose","Mean future","Ridge (train)","Conditional CVAE"]
 order=["persistence","train_mean_curve","ridge","conditional_vae_prior_mean"]
 x=np.arange(len(order))
 v=[result["metrics"]["val"][k]["mse"] for k in order]
 t=[result["metrics"]["test"][k]["mse"] for k in order]
 fig,ax=plt.subplots(figsize=(10.8,5.4))
 ax.bar(x-.19,v,width=.38,label="Validation (4 matches)")
 ax.bar(x+.19,t,width=.38,label="Unseen-match test (4 matches)")
 for xi,a,b in zip(x,v,t):
  ax.text(xi-.19,a+.003,f"{a:.3f}",ha="center",fontsize=9)
  ax.text(xi+.19,b+.003,f"{b:.3f}",ha="center",fontsize=9)
 ax.set_xticks(x,title)
 ax.set_ylabel("MSE on predicted pose-relation cosines (lower is better)")
 ax.set_title("KineticKick v0.1.39 | Broadcast-estimated 3D soccer coordination temporal pilot")
 ax.grid(axis="y",alpha=.3)
 ax.legend()
 fig.text(.01,.006,
   "108 train / 47 validation / 45 test from mutually exclusive 12 / 4 / 4 football matches. CVAE selected by validation only. "
   "Source is broadcast-estimated 3D joints; event labels, real instep mechanics, G1 motion and human naturalness are NOT verified.",
   fontsize=7)
 fig.tight_layout(rect=(0,.055,1,1))
 figpath=ROOT/"media/v039_coordination_prediction_baselines.png"
 figpath.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(figpath,dpi=170)
 plt.close(fig)
 print("V039_RESEARCH_COMPARISON",json.dumps({
   "test":result_numeric["test_mse"],"validation":result_numeric["validation_mse"],
   "test_model_vs_ridge_mse_delta":result_numeric["mse_delta_model_minus_ridge"],
   "model_worse_in_games":int((diff>0).sum()),"test_games":len(game_names),
   "figure":str(figpath)},ensure_ascii=False),flush=True)
 return result_numeric
if __name__=="__main__":analyze()
