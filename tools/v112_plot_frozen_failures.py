#!/usr/bin/env python3
"""Plot only SHA-authenticated v112 physical gate failures and causal-control deltas."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v112_coordination_falsification import ROOT, digest
def main():
    src=ROOT/"reports/v112_coordination_falsification.audit.json"
    a=json.loads(src.read_text())
    if a["source_master_sha256"]!=digest(ROOT/"reports/v112_original_g1_batch.audit.json"):
        raise ValueError("STALE_RAW_EVIDENCE")
    names=["W/v32","W/cmu02","X/v32","X/cmu02","Y/v32","Y/cmu02"]
    values=[a["independent_runs"][x]["instep_fraction"]*100 for x in names]
    passed=[a["independent_runs"][x]["hard_gate_pass"] for x in names]
    if sum(passed)!=2:raise ValueError("UNEXPECTED_FROZEN_STRICT_GATES")
    fig,axs=plt.subplots(2,1,figsize=(9.4,6.3),layout="constrained")
    axs[0].bar(names,values)
    axs[0].axhline(95,ls="--",c="black",lw=1,label="Frozen instep >= 95%")
    axs[0].set_ylim(0,110);axs[0].set_ylabel("Measured instep impulse share (%)")
    axs[0].legend()
    for i,v in enumerate(values):axs[0].text(i,v+1.5,f"{v:.1f}% / {'PASS' if passed[i] else 'FAIL'}",ha="center",fontsize=8)
    pairs=a["preregistered_scene_pairs"]
    axs[1].bar([x["scene"] for x in pairs],[x["cmu_minus_v32_ball_speed_proxy_mps"]*1000 for x in pairs])
    axs[1].axhline(0,c="black",lw=.8)
    axs[1].set_ylabel("CMU02 - v32 ball-speed proxy (mm/s)")
    axs[1].set_xlabel("Frozen scene (OFF-only, independent pairs)")
    fig.suptitle("v112 | Original 29DoF G1: physical failures and negative human-reference effect")
    out=ROOT/"reports/v112_frozen_g1_failure_and_control.png"
    if out.exists():raise FileExistsError("WRITE_ONCE_PLOT")
    fig.savefig(out,dpi=145)
    plt.close(fig)
    print("SAVED",out, "png_bytes",out.stat().st_size)
if __name__=="__main__":main()
