#!/usr/bin/env python3
"""Actual six-C3D counterfactuals and 24 ORIGINAL G1 true-physics failure timelines."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def make(human,robot,out_dir):
    h=json.loads(Path(human).read_text())
    r=json.loads(Path(robot).read_text())
    if h["trial_pair_tests"]!=12 or r["original_independent_scene_controller_cells"]!=24:
        raise ValueError("UNFROZEN_REAL_COHORT")
    rows=[]
    for t in h["trials"]:
        for name,v in t["pairs"].items():
            rows.append((t["trial"]+"\n"+name[:4],v))
    x=np.arange(len(rows))
    observed=np.array([abs(v["observed"]["best"]["correlation"]) for _,v in rows])
    null=np.array([v["surrogate_peak_abs_corr_p95"] for _,v in rows])
    fig,ax=plt.subplots(figsize=(13,5.6))
    ax.scatter(x-.13,observed,s=55,label="Measured max |Pearson r|")
    ax.scatter(x+.13,null,s=55,marker="s",label="Circular-shift null p95 of max |r|")
    for i,(_,v) in enumerate(rows):
        if v["observed_lag_on_boundary"]:
            ax.text(i, max(observed[i],null[i])+.035,"EDGE",ha="center",rotation=90,fontsize=7)
    ax.set_xticks(x,[label for label,_ in rows],rotation=65)
    ax.set_ylim(0,max(1.,float(max(observed.max(),null.max())+.25)))
    ax.set_ylabel("Peak |r|, descriptive only")
    ax.set_title("6 real CMU C3Ds (2 subjects): apparent human arm–toe lags vs phase-shift null")
    ax.legend(loc="upper right");ax.grid(alpha=.25)
    fig.tight_layout()
    p=Path(out_dir)/"v125_real_human_lag_surrogates.png"
    fig.savefig(p,dpi=150);plt.close(fig)
    # First actual qdot violation relative to ball impact; excludes observer repeats.
    physical=sorted(r["per_episode"],key=lambda e:(e["version"],e["scene"],e["control"]))
    fig,ax=plt.subplots(figsize=(12,8.2))
    xpositions=[];labels=[];ypositions=[]
    for i,e in enumerate(physical):
        label=f"{e['scene']} / {e['control']}" + (" / PASS" if e["strict_hard_gate_pass"] else " / FAIL")
        labels.append(label);ypositions.append(i)
        v=e["first_qdot_violation"]
        if v["found"]:
            rel=v["time_s"]-e["impact_s"]
            ax.scatter(rel,i,marker="x",s=60)
            if rel<=1.7:
                ax.text(rel+.03,i-.10,v["joint"].replace("_joint",""),fontsize=6)
            xpositions.append(rel)
    ax.axvline(0,linestyle="--",label="First ball contact (t=0)")
    ax.axvline(1.25,linestyle=":",label="End of initial 1.25s follow-through")
    ax.set_yticks(ypositions,labels,fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("First original-29DoF qdot limit violation relative to impact (seconds)")
    ax.set_title("24 SHA-certified original G1 PhysX independent scene×controller cells; no new G1 rollout")
    ax.legend(loc="lower right");ax.grid(alpha=.25,axis="x")
    fig.tight_layout()
    q=Path(out_dir)/"v125_true_physx_qdot_failure_timeline.png"
    fig.savefig(q,dpi=150);plt.close(fig)
    return [p,q]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--human",type=Path,default=ROOT/"reports/v125_human_counterfactual_sha.audit.json")
    p.add_argument("--g1",type=Path,default=ROOT/"reports/v125_original_g1_timeline_sha.audit.json")
    p.add_argument("--output-dir",type=Path,default=ROOT/"reports")
    a=p.parse_args()
    for f in make(a.human,a.g1,a.output_dir):print("PLOTTED_REAL_MEASUREMENTS",f)
if __name__=="__main__":main()
