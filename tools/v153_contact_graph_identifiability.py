#!/usr/bin/env python3
"""Frozen v153 algebraic countermodels for an observed 3D body net-force pair.

Graphs are NOT measured colliders, real G1 poses, native PhysX manifolds, or
dynamically feasible contact configurations. They demonstrate only that
equal/opposite net body forces do not uniquely determine contact topology.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
from v142_native_pair_ledger import safe_write_once

ROOT = Path(__file__).resolve().parents[1]
FREEZE = "08774eae66c2bddf3796da47e6df31088d9a33b6"
CONFIG = "configs/v153_frozen_collision_identifiability.json"
WRIST, HIP = "left_wrist_yaw_link", "left_hip_roll_link"
THIRD, GROUND_W, GROUND_H = "unspecified_third_body", "world_contact_w", "world_contact_h"
TOPOLOGIES = ("direct", "two_to_third", "two_to_world", "split_paths")

def frozen():
    b = subprocess.check_output(["git", "-C", str(ROOT), "show", FREEZE + ":" + CONFIG])
    if b != (ROOT / CONFIG).read_bytes():
        raise ValueError("V153_FROZEN_PROTOCOL_CHANGED")
    cfg = json.loads(b)
    if (cfg["version"] != "v0.1.153"
        or cfg["parent_sha"] != "cda3b6e62a9d99c05842d201574c83585b1f2584"
        or cfg["matrix"]["total"] != 288
        or cfg["original_g1_dof"] != 29 or cfg["physics_dt_s"] != .005
        or cfg["no_native_pair_or_manifold_truth_claim"] is not True):
        raise ValueError("V153_PROTOCOL_INVALID")
    return cfg

def vector(v):
    if not isinstance(v, (tuple, list)) or len(v) != 3:
        raise ValueError("V153_NOT_THREE_COMPONENT_VECTOR")
    if any(type(a) not in (float, int) or not math.isfinite(a) for a in v):
        raise ValueError("V153_NONFINITE_VECTOR")
    return [float(a) for a in v]

def vadd(x, y):
    return [a+b for a,b in zip(x,y)]

def scaled(x, k):
    return [a*k for a in x]

def magnitude(x):
    return math.sqrt(sum(v*v for v in x))

def graph_edges(name, f):
    f = vector(f)
    if not magnitude(f):
        raise ValueError("V153_ZERO_FORCE_NO_DIRECTION")
    if name == "direct":
        links = [(WRIST,HIP,f)]
    elif name == "two_to_third":
        links = [(WRIST,THIRD,f),(HIP,THIRD,scaled(f,-1))]
    elif name == "two_to_world":
        links = [(WRIST,GROUND_W,f),(HIP,GROUND_H,scaled(f,-1))]
    elif name == "split_paths":
        links = [(WRIST,THIRD,scaled(f,.5)),
                 (WRIST,GROUND_W,scaled(f,.5)),
                 (HIP,THIRD,scaled(f,-.5)),
                 (HIP,GROUND_H,scaled(f,-.5))]
    else:
        raise ValueError("V153_UNKNOWN_CONTACT_TOPOLOGY")
    return [{"a":a,"b":b,"force_on_a_N":x} for a,b,x in links]

def net_forces(edges):
    """Compute every node's net force with explicit Newton pair conservation."""
    out={}
    for row in edges:
        a,b=row["a"],row["b"]
        if a == b: raise ValueError("V153_SELF_EDGE")
        v=vector(row["force_on_a_N"])
        out[a]=vadd(out.get(a,[0.,0.,0.]),v)
        out[b]=vadd(out.get(b,[0.,0.,0.]),scaled(v,-1))
    return out

def create_case(name,f):
    cfg=frozen()
    if name not in cfg["matrix"]["contact_graphs"]:
        raise ValueError("V153_UNFROZEN_GRAPH")
    edges=graph_edges(name,f)
    result=net_forces(edges)
    f=vector(f)
    tolerance=cfg["max_net_force_residual_N"]
    residual=max(abs(x-y) for actual,expected in (
            (result[WRIST],f),(result[HIP],scaled(f,-1)))
            for x,y in zip(actual,expected))
    if residual > tolerance:raise ValueError("V153_GRAPH_DOES_NOT_MATCH_OBSERVATIONS")
    direct=any({e["a"],e["b"]}=={WRIST,HIP} for e in edges)
    if direct != (name=="direct"):
        raise ValueError("V153_TOPOLOGY_LABEL_MISMATCH")
    return {"hypothesis":name,"symbolic_edges_not_physx_manifold":edges,
            "net_wrist_N":result[WRIST],"net_hip_N":result[HIP],
            "has_wrist_hip_direct_edge":direct,
            "observed_force_residual_N":residual,
            "all_contacts_are_counterfactual_not_runtime_ones":True}

def directions(observed_force):
    f=vector(observed_force)
    n=magnitude(f)
    if n <= 0:raise ValueError("V153_NO_MEASURED_DIRECTION")
    out=[[x/n for x in f]]
    for i in range(3):
        elevation=(-.7,0.,.7)[i]
        for j in range(8):
            az=2*math.pi*j/8
            r=math.sqrt(1-elevation*elevation)
            out.append([r*math.cos(az),r*math.sin(az),elevation])
    return out[:24]

def matrix(observed_force):
    cfg=frozen()
    original_norm=magnitude(vector(observed_force))
    cases=[]
    for i,d in enumerate(directions(observed_force)):
        for scale in cfg["matrix"]["scales"]:
            f=scaled(d,original_norm*scale)
            for name in cfg["matrix"]["contact_graphs"]:
                result=create_case(name,f)
                cases.append({"direction_index":i,"norm_scale":scale,
                              "observation_wrist_N":f,**result})
    if len(cases)!=288:raise ValueError("V153_INCOMPLETE_FROZEN_MATRIX")
    counts=Counter(r["hypothesis"] for r in cases)
    if set(counts)!=set(TOPOLOGIES) or any(v!=72 for v in counts.values()):
        raise ValueError("V153_MATRIX_UNBALANCED")
    return {"version":"v0.1.153","kind":"ALGEBRAIC_CONTACT_GRAPH_COUNTEREXAMPLES",
            "historical_force_magnitude_N":original_norm,
            "matrix_rows":len(cases),"counts":dict(counts),
            "all_graphs_same_two_body_net_forces":True,
            "all_graphs_are_physically_certified":False,
            "actual_left_wrist_hip_physx_manifold":"UNKNOWN",
            "new_physx_runs":0,"rows":cases}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-audit",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    source=json.loads(a.source_audit.read_text())
    if source.get("version")!="v0.1.153" or source.get("named_pair_verification")!="NET_FORCE_ONLY":
        raise ValueError("V153_SOURCE_NOT_VERIFIED_NET_FORCE")
    result=matrix(source["observed_wrist_net_force_N"])
    result["historical_source_sha256"]=source["historical_failed_trial_sha256"]
    checksum=safe_write_once(a.output,result)
    print(json.dumps({"rows":len(result["rows"]),"sha256":checksum,
                      "direct_topology_supported_as_truth":False,"new_physx_runs":0}))

if __name__=="__main__":
    main()
