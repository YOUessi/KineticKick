#!/usr/bin/env python3
"""Conservative opposite-net-force pair hypotheses from frozen v55 real PhysX.

Net wrench cancellation is *not* a collision manifold. Neither body IDs nor
collider meshes nor points of contact can be inferred uniquely from it.
"""
from __future__ import annotations
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROGRESS=ROOT/"reports/v055_pair_batch_progress.json"
OUTPUT=ROOT/"reports/v055_nonfoot_net_force_pair_candidates.json"


def pair_hypotheses(events, force_tol_n=.002, min_force_n=20.):
    times=defaultdict(list)
    for row in events:
        t=round(float(row["time_s"]),3)
        if not all(math.isfinite(x) for x in row["force_w_N"]):
            raise ValueError("Nonfinite physical net-force sensor data")
        times[t].append(row)
    pairs=[]
    for time_s,rows in times.items():
        # Supports multiple contact bodies at one step, does not force pairing.
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                if a["body"]==b["body"]:
                    continue
                f=a["force_w_N"];g=b["force_w_N"]
                magnitude=math.sqrt(sum(x*x for x in f))
                cancellation=math.sqrt(sum((x+y)**2 for x,y in zip(f,g)))
                if magnitude>=min_force_n and cancellation<=force_tol_n:
                    pairs.append(dict(approx_time_s=time_s,
                                      body_names=sorted((a["body"],b["body"])),
                                      observed_force_magnitude_N=magnitude,
                                      net_force_cancellation_residual_N=cancellation,
                                      true_physx_contact_pair_verified=False,
                                      true_collider_paths_verified=False))
    return pairs


def run():
    if OUTPUT.exists():
        raise FileExistsError("Do not overwrite hypotheses/evidence from prior analysis")
    p=json.loads(PROGRESS.read_text())
    data=[]
    for run in p["runs"]:
        f=ROOT/run["out"]
        if hashlib.sha256(f.read_bytes()).hexdigest()!=run["sha256"]:
            raise ValueError("Input PhysX JSON SHA changed "+run["id"])
        episode=json.loads(f.read_text())
        candidates=pair_hypotheses(episode["nonfoot_contact_diagnostics"])
        data.append(dict(run_id=run["id"],source_sha256=run["sha256"],
                         termination=episode["termination_reasons"],
                         nonfoot_net_force_sensor_events=len(episode["nonfoot_contact_diagnostics"]),
                         oppositely_balanced_named_body_candidates=candidates))
    result=dict(version="0.1.55",independent_from_control=True,
        physical_rollouts=len(data),
        certified_contact_pair_count=0,
        caveat="Opposite per-link net forces in original sensor data are not PhysX narrow-phase contact pairs; repeated force cancellation may include several hidden simultaneous contacts.",
        results=data)
    OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    for row in data:
        if row["oppositely_balanced_named_body_candidates"]:
            p=row["oppositely_balanced_named_body_candidates"][0]
            print("V055_NET_FORCE_HYPOTHESIS",row["run_id"],p["body_names"],
                  round(p["observed_force_magnitude_N"],3),"N")
    return result

if __name__=="__main__":run()
