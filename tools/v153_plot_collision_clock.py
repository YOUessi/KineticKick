#!/usr/bin/env python3
"""Plot historical real 5ms signals and unrecorded named-body event (not a contact pair)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v146_real_200hz_coupled_audit import verified_sources,split_real_5ms_rows

def figure(out):
    sources=verified_sources()
    raw=json.loads(sources["cmu_10_05"].read_text())
    rows,note=split_real_5ms_rows(raw["traces"][0],1028)
    if len(note)!=1:raise ValueError("TERMINAL_NOTE_NOT_SEPARATED")
    s=[r for r in rows if r["time_s"]>=4.7]
    t=[r["time_s"] for r in s]
    named=raw["nonfoot_contact_diagnostics"]
    if len(named)!=2:raise ValueError("MISSING_NAMED_REAL_SENSOR_EVENT")
    tn=float(named[0]["time_s"])
    plt.figure(figsize=(12,6))
    plt.plot(t,[r["support_foot_force_N"]["left"] for r in s],label="Left-foot net contact (N)")
    plt.plot(t,[r["support_foot_force_N"]["right"] for r in s],label="Right-foot net contact (N)")
    plt.plot(t,[r["nonfoot_contact_N"] for r in s],label="Non-foot net contact (N)")
    plt.axvline(5.135,linestyle="--",linewidth=1.5,label="5.135s: original 200Hz non-foot net-force spike")
    plt.axvline(5.14,linestyle="-.",linewidth=1.5,label="5.140s: last original physical state")
    plt.axvline(tn,linestyle=":",linewidth=2,label="~5.145s: named-body sensor, no same-step 29DoF row")
    plt.axvspan(5.14,5.152,alpha=.12,label="After last available physical state")
    plt.xlim(4.9,5.153)
    plt.xlabel("Original simulation time (s)")
    plt.ylabel("Net force magnitude (N) — NOT isolated ground reaction force")
    plt.title("CMU10_05 historical PhysX: censored 200Hz state vs later sensor event")
    plt.legend(fontsize=8,loc="upper left")
    plt.grid(alpha=.2)
    plt.tight_layout()
    if Path(out).exists():raise FileExistsError(out)
    plt.savefig(out,dpi=155)
    plt.close()
    return {"output":str(out),"actual_physics_steps":len(rows),
            "named_event_after_last_real_state":tn>rows[-1]["time_s"],
            "true_native_wrist_hip_pair":"UNKNOWN"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(figure(a.output)))

if __name__=="__main__":
    main()
