#!/usr/bin/env python3
"""v44 explainable small-sample 29-joint delayed speed forecast comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def run():
    cfg=json.loads((ROOT/"configs/contact_future_forecast_v044.json").read_text())
    result=json.loads((ROOT/cfg["training_dir"]/"result.json").read_text())
    names=["Current qdot","Training mean","Train-only ridge","Planned-action MLP"]
    keys=["current_qdot_persistence","training_mean_future","training_only_ridge","planned_action_mlp"]
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for ax,kind in zip(axes,["val","test"]):
        data=result["metrics"][kind]
        y=np.asarray([data[k]["mae"] for k in keys])
        bars=ax.bar(np.arange(4),y)
        for rect,v in zip(bars,y):
            ax.text(rect.get_x()+rect.get_width()/2,v+.004,f"{v:.4f}",ha="center",fontsize=9)
        ax.set_xticks(range(4),names,rotation=17,ha="right")
        ax.set_ylabel("MAE of each future joint maximal |qdot|/unchanged limit")
        ax.set_ylim(0,.23)
        ax.set_title("Validation, 1 trial group (2 robot runs)" if kind=="val" else "Grouped test, 1 trial group (2 robot runs)")
        ax.grid(axis="y",alpha=.25)
    fig.suptitle("KineticKick v0.1.44 | Predicting actual G1 joint-speed risk through contact +1.25s")
    fig.text(.015,.006,"144 correlated pre-impact samples from 12 already inspected PhysX episodes; group split 4 training / 1 validation / 1 test ball states. On grouped test, the MLP flags all 12 actually unsafe original-v32 windows as safe; this is NOT a deployable risk controller.",fontsize=7.5)
    fig.tight_layout(rect=(0,.06,1,.93))
    output=ROOT/"media/v044_delayed_joint_risk_forecaster_baselines.png"
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150)
    plt.close(fig)
    print("V044_PLOT",output,output.stat().st_size)
if __name__=="__main__":run()
