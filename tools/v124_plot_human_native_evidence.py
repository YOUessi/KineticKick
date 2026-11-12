#!/usr/bin/env python3
"""Plot only SHA-qualified local audits; no human force, skill or G1 motor claims."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def plot(human,native,out):
    rows=human["per_trial"]
    if human["qualified_real_human_c3d_trials"]!=6 or len(rows)!=6:
        raise ValueError("SIX_SHA_QUALIFIED_C3D_REQUIRED")
    if native["actual_isolated_physx_runs"]!=6 or native["native_pair_positive_certifications"]!=0:
        raise ValueError("NATIVE_SOURCE_EVIDENCE_DRIFT")
    labels=[r["trial"] for r in rows]
    lag_a=[];lag_b=[];corr_a=[];corr_b=[]
    for row in rows:
        p=row["exploratory_opposite_limb_radial_rate_lag"]
        aa=p["LWRA_LSHO__RTOE_PELVIS"]["original"]["best"]
        bb=p["RWRA_RSHO__LTOE_PELVIS"]["original"]["best"]
        lag_a.append(aa["lag_seconds"]);lag_b.append(bb["lag_seconds"])
        corr_a.append(aa["correlation"]);corr_b.append(bb["correlation"])
    fig,ax=plt.subplots(figsize=(9,4.5))
    ix=np.arange(len(rows))
    ax.scatter(ix-.09,lag_a,s=55,label="Left wrist ↔ right toe radial-rate")
    ax.scatter(ix+.09,lag_b,s=55,label="Right wrist ↔ left toe radial-rate")
    ax.axhline(0,color="gray",lw=.8)
    ax.axhline(.35,color="gray",lw=.7,ls="--")
    ax.axhline(-.35,color="gray",lw=.7,ls="--")
    ax.set_ylim(-.43,.43)
    ax.set_xticks(ix,labels)
    ax.set_ylabel("Exploratory best lag (s), |lag| ≤ 0.35")
    ax.set_xlabel("Original 120 Hz C3D trial (two subjects)")
    ax.set_title("Observed radial distance-change timing; not certified soccer impact")
    ax.legend(loc="upper right",fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out/"v124_human_relative_lag_measured.png"),dpi=145)
    plt.close(fig)
    cases=native["per_case"]
    fig,ax=plt.subplots(figsize=(10,4.5))
    x=np.arange(len(cases))
    ax.bar(x,[int(r["positive_dynamics"]) for r in cases],width=.33,
           label="collision dynamics observed")
    ax.bar(x+.33,[int(r["native_pair_qualified"]) for r in cases],width=.33,
           label="native collider pair certified")
    ax.set_xticks(x+.165,[r["name"].replace("_","\n") for r in cases],fontsize=8)
    ax.set_ylim(0,1.2);ax.set_yticks([0,1])
    ax.set_title("Historical isolated CPU/CUDA PhysX: no native pair witness")
    ax.set_ylabel("Boolean observed / certified")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out/"v124_native_contact_missing_truth.png"),dpi=145)
    plt.close(fig)
    return dict(left_wrist_right_toe_lags_s=lag_a,
                right_wrist_left_toe_lags_s=lag_b,
                left_wrist_right_toe_at_lag_boundary=sum(abs(x)>=.35-1e-8 for x in lag_a),
                right_wrist_left_toe_positive_lag_count=sum(x>0 for x in lag_b),
                human_trials=6,native_isolated_historical_runs=6,
                native_certified_pairs=0,figures_created=2)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--human",type=Path,
                        default=ROOT/"reports/v124_qualified_human_coordination.audit.json")
    parser.add_argument("--native",type=Path,
                        default=ROOT/"reports/v124_native_truth_gate.retry1.audit.json")
    parser.add_argument("--out",type=Path,default=ROOT/"reports")
    args=parser.parse_args()
    result=plot(json.loads(args.human.read_text()),json.loads(args.native.read_text()),args.out)
    print(json.dumps(result))
if __name__=="__main__":main()
