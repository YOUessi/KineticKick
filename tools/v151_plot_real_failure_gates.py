#!/usr/bin/env python3
"""V151 two immutable figures from verified ORIGINAL G1 PhysX evidence only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from v151_native_campaign import SOURCE_ROOT,frozen
from v151_independent_historical_reaudit import independently_reaudit
from v146_real_200hz_coupled_audit import verified_sources,split_real_5ms_rows

LABELS=[
    ("legal_right_instep_contact","95% instep impulse"),
    ("outgoing_ball_speed_at_least_3_5",">=3.5 m/s ball"),
    ("entire_ball_goal","goal crossing"),
    ("actual_qdot_under_original_limits","29 qdot safe"),
    ("genuine_runup_and_shot","runup continuity"),
    ("real_physx_full_14_seconds","14 s complete"),
    ("recovered_and_stable","stable recovery")]
POLICY=["frozen_v32","cmu_10_02","cmu_10_05"]


def create(outdir):
    outdir=Path(outdir)
    if not outdir.is_dir():
        raise ValueError("V151_PLOT_DEST_NOT_DIRECTORY")
    outcomes=[outdir/"v151_real_v53_12_physics_failure_gate_matrix.png",
              outdir/"v151_real_v54_failed_nonfoot_5ms_timeline.png"]
    if any(p.exists() or p.is_symlink() for p in outcomes):
        raise FileExistsError("V151_PLOTS_WRITE_ONCE")
    source=independently_reaudit()
    counts=np.array([
       [source["historical_v53_strict_by_policy"][name]["failed_gate_counts"].get(key,0)
        for key,_ in LABELS] for name in POLICY],dtype=int)
    fig,ax=plt.subplots(figsize=(11,3.6))
    heat=ax.imshow(counts,vmin=0,vmax=4,cmap="Greys",aspect="auto")
    ax.set_yticks(range(len(POLICY)),POLICY)
    ax.set_xticks(range(len(LABELS)),[s for _,s in LABELS],rotation=24,ha="right")
    ax.set_title("Historical 12 original G1 PhysX trials: independent failed strict gates (0–4 per policy)")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j,i,str(counts[i,j]),ha="center",va="center",
                    color="white" if counts[i,j]>=3 else "black")
    fig.colorbar(heat,ax=ax,label="failed of four trials")
    fig.tight_layout()
    fig.savefig(outcomes[0],dpi=130)
    plt.close(fig)

    files=verified_sources()
    raw=json.loads(files["cmu_10_05"].read_text())
    n=frozen()["history_expected_rows"]["cmu_10_05"]
    rows,terminal=split_real_5ms_rows(raw["traces"][0],n)
    tail=[r for r in rows if r["time_s"]>=4.8]
    ts=[float(r["time_s"]) for r in tail]
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.plot(ts,[float(r["nonfoot_contact_N"]) for r in tail],
            label="non-foot *net* contact, measured sensor (N)")
    ax.plot(ts,[float(r["support_foot_force_N"]["left"]) for r in tail],
            label="left foot net contact (N)")
    ax.plot(ts,[float(r["support_foot_force_N"]["right"]) for r in tail],
            label="right foot net contact (N)")
    ax.axhline(20,linestyle="--",linewidth=1,label="historical termination threshold 20N")
    ax.axvline(rows[-1]["time_s"],linestyle=":",label="last actual 200Hz physics state")
    ax.set(xlabel="Original G1 PhysX time (s)",ylabel="Aggregate sensor force (N)",
           title="Original CMU10_05: 5ms non-foot event and support, NOT a native collider pair")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outcomes[1],dpi=130)
    plt.close(fig)
    return dict(paths=[str(x) for x in outcomes],
       count_matrix=counts.tolist(),last_physics_time_s=rows[-1]["time_s"],
       excluded_nonphysics_terminal_notes=len(terminal),
       new_physx_episodes=0,
       native_wrist_hip_pair="UNKNOWN")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir",type=Path,required=True)
    a=ap.parse_args()
    print(json.dumps(create(a.outdir),ensure_ascii=False))


if __name__=="__main__":
    main()
