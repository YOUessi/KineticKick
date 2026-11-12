#!/usr/bin/env python3
"""Plot measured 200Hz URDF geometry, never inventing the 5.145s joint state."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

CASES=("frozen_v32","cmu_10_02","cmu_10_05")
PAIR_IDS=("left_hip_roll_link__left_wrist_yaw_link",
          "left_shoulder_yaw_link__torso_link")
LABELS=("Left wrist—left hip: URDF outer bound","Left shoulder—torso: URDF outer bound")


def plot(inp: Path,out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data=json.loads(inp.read_text())
    if data["real_200hz_states_revalidated"]!=6628:
        raise ValueError("SOURCE_REAL_PHYSICS_FRAME_DRIFT")
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for axis,pair,label in zip(axes,PAIR_IDS,LABELS):
        for case in CASES:
            row=data["cases"][case]["sampled_event_window"]
            t=np.asarray(row["time_s"],float)
            gap=1000*np.asarray(row["selected_URDF_lower_bounds_m"][pair],float)
            if len(t)!=13 or len(gap)!=13 or not np.isfinite(gap).all():
                raise ValueError("INVALID_MEASURED_REAL_STEP_CURVE")
            # Keep discontinuity beyond last observed frame. No q at sensor event.
            axis.plot(t,gap,marker=".",label=case)
        axis.axhline(20,ls="--",alpha=.7,label="20 mm warning")
        axis.axvline(5.145,ls=":",alpha=.8,label="5.145 s sensor event (no synchronized q)")
        axis.set_ylabel("Lower bound (mm)")
        axis.set_title(label)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8,loc="upper right")
    axes[-1].set_xlabel("Actual simulation time (s), 200 Hz; data end 5.140 s")
    axes[-1].set_xlim(5.075,5.153)
    fig.suptitle("KineticKick v132: original G1 URDF outer capsules — NOT native PhysX contact")
    fig.tight_layout()
    with out.open("xb") as fh:
        fig.savefig(fh,format="png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("PLOT_ALREADY_EXISTS")
    plot(args.input,args.output)
    print("V132_REAL_MEASURED_FAILURE_PLOT",args.output)
