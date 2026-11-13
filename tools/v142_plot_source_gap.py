#!/usr/bin/env python3
"""Show actual historical 200Hz trace coverage and missing human truth.

The first panel shows ORIGINAL v54 PhysX trace lengths, not newly simulated
shooting and not the physical contact pair. No tail is reconstructed.
"""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v142_native_pair_ledger import frozen, safe_write_once


def draw_report(input_path, stem):
    frozen()
    obj=json.loads(Path(input_path).read_text())
    history=obj["historical_physics"]
    human=obj["human_data"]
    if history["historical_physics_5ms_frames"]!=6628 or human["clip_count"]!=2:
        raise ValueError("V142_REAL_SOURCE_TRACE_DRIFT")
    stem=Path(stem)
    first=stem.with_name(stem.name+"_real_physx_trace_coverage.png")
    second=stem.with_name(stem.name+"_human_truth_eligibility.png")
    if first.exists() or second.exists():
        raise FileExistsError("V142_PLOTS_WRITE_ONCE")
    dt=frozen()["physics_dt_s"]
    names=list(history["cases"])
    steps=[history["cases"][x] for x in names]
    fig,ax=plt.subplots(figsize=(9,3.9))
    for i,(name,n) in enumerate(zip(names,steps)):
        ax.barh(i,n*dt,height=.48,label="observed" if i==0 else None)
        if n<2800:
            ax.barh(i,(2800-n)*dt,left=n*dt,alpha=.2,height=.48,
                    label="not observed" if i==0 else None)
        ax.text(min(n*dt+.05,13.5),i,f"{n} real states",va="center",fontsize=10)
    ax.axvline(history["cmu_10_05_sensor_event_s"],linestyle=":",
               label="CMU10_05 named sensor event, no 29DoF state")
    ax.set_yticks(list(range(3)),names)
    ax.invert_yaxis()
    ax.set_xlim(0,14.1)
    ax.set_xlabel("Original simulated elapsed time (s); last failing q=5.140 s")
    ax.set_title("v54 original real PhysX evidence coverage (historical, not v142 reruns)")
    ax.legend(fontsize=8,loc="lower right")
    fig.tight_layout()
    fig.savefig(first,dpi=155)
    plt.close(fig)

    features=["120 Hz 3D pose","measured foot-ball instant",
              "synchronized GRF","elite human instep label"]
    matrix=[]
    for id in ("10_02","10_05"):
        meta=human["clips"][id]
        matrix.append([int(meta["human_pose_prior_only"]),
                       int(meta["measured_instep_contact_impulse"]),
                       int(meta["measured_human_grf"]),
                       int(meta["elite_football_technique_certified"])])
    fig,ax=plt.subplots(figsize=(10.1,2.9))
    ax.imshow(matrix,aspect="auto",vmin=0,vmax=1,cmap="Greys")
    ax.set_xticks(range(4),features,rotation=15,ha="right")
    ax.set_yticks([0,1],["CMU10_02","CMU10_05"])
    for i,row in enumerate(matrix):
        for j,val in enumerate(row):
            ax.text(j,i,"YES" if val else "NO",ha="center",va="center",
                    color="white" if val else "black",fontsize=11)
    ax.set_title("Directly qualified measurements; NO is not evidence of absent physiology")
    fig.tight_layout()
    fig.savefig(second,dpi=155)
    plt.close(fig)
    return [str(first),str(second)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--stem",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps({"plots":draw_report(args.input,args.stem)}))


if __name__=="__main__":
    main()
