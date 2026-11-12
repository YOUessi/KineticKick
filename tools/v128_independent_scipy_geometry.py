#!/usr/bin/env python3
"""v128 SciPy independent bounded endpoint geometry crosscheck on real v54 G1 q.

Checks legacy v127 vectorized segment-distance algorithm, not PhysX collider truth.
No tuning: original three policies, original 5.130/5.135/5.140s samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import v127_collider_proximity as g
import v128_v54_event_geometry as a

TIMES=(5.130, 5.135, 5.140)


def independent_segment_gap(p0,p1,q0,q1,radii):
    p0,p1,q0,q1=map(lambda x:np.asarray(x,dtype=float).reshape(3),(p0,p1,q0,q1))
    d1=p1-p0;d2=q1-q0;r=p0-q0
    def square_distance(z):
        d=r+z[0]*d1-z[1]*d2
        return float(d@d)
    def gradient(z):
        v=r+z[0]*d1-z[1]*d2
        return np.array([2*v@d1,-2*v@d2])
    results=[]
    for seed in ((0,0),(1,0),(0,1),(1,1),(.5,.5),(.25,.75),(.75,.25)):
        sol=minimize(square_distance,seed,jac=gradient,method="SLSQP",
                     bounds=((0,1),(0,1)),
                     options={"maxiter":1000,"ftol":1e-16})
        results.append(float(square_distance(sol.x)))
    return float(np.sqrt(max(0,min(results)))-sum(radii))


def run(cfg):
    a.require_sha(Path(cfg["original_G1_urdf"]),cfg["original_G1_urdf_sha256"])
    tree=g.parse_tree(Path(cfg["original_G1_urdf"]))
    pair=cfg["failure_pair_names"]
    records=[]
    for spec in cfg["original_v54_cases"]:
        path=Path(cfg["data_root"])/f'v054_{spec["case"]}_contact_diagnostics.json'
        a.require_sha(path,cfg["input_sha256"][path.name])
        raw=json.loads(path.read_text())
        data=a.validate_rows(raw,spec["expected_trace_rows"],cfg["dt_s"])
        for t in TIMES:
            ind=int(np.argmin(np.abs(data["t"]-t)))
            if abs(data["t"][ind]-t)>2e-5:
                raise ValueError("NO_REAL_200HZ_MATCHING_TIME")
            q=data["q"][ind:ind+1]
            fk=g.fk_targets(tree,q,set(pair))
            p0,p1,ra=g.cylinder_axis_endpoints(tree,fk,pair[0])
            q0,q1,rb=g.cylinder_axis_endpoints(tree,fk,pair[1])
            internal=float(g.segment_distance(p0,p1,q0,q1)[0]-ra-rb)
            reference=independent_segment_gap(p0[0],p1[0],q0[0],q1[0],(ra,rb))
            if abs(internal-reference)>2e-7:
                raise ValueError("INDEPENDENT_SCIPY_GEOMETRY_MISMATCH")
            records.append(dict(case=spec["case"],real_sample_s=t,
                original_URDF_capsule_lower_bound_m=internal,
                independent_SciPy_bounded_minimum_m=reference,
                absolute_discrepancy_m=abs(internal-reference),
                native_contact_pair_certified=False))
    return dict(frozen_protocol_commit=a.FROZEN_SHA,
                original_G1_samples_crosschecked=len(records),
                independent_optimization_not_physx_manifold=True,
                maximum_difference_m=max(x["absolute_discrepancy_m"] for x in records),
                results=records)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out=run(a.frozen_protocol())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(samples=out["original_G1_samples_crosschecked"],
                          max_delta_m=out["maximum_difference_m"]),indent=2))
