#!/usr/bin/env python3
"""Independent incidence-matrix re-audit; does NOT import v153 graph builder."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from v142_native_pair_ledger import safe_write_once

WRIST="left_wrist_yaw_link"
HIP="left_hip_roll_link"
EXPECTED={
    "direct":((WRIST,HIP),),
    "two_to_third":((WRIST,"unspecified_third_body"),(HIP,"unspecified_third_body")),
    "two_to_world":((WRIST,"world_contact_w"),(HIP,"world_contact_h")),
    "split_paths":((WRIST,"unspecified_third_body"),(WRIST,"world_contact_w"),
                   (HIP,"unspecified_third_body"),(HIP,"world_contact_h")),
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()

def audit_matrix(rows, source):
    if type(rows) is not dict or type(source) is not dict:
        raise ValueError("V153_BAD_INDEPENDENT_INPUT")
    if rows.get("kind")!="ALGEBRAIC_CONTACT_GRAPH_COUNTEREXAMPLES":
        raise ValueError("V153_WRONG_SOURCE_LEVEL")
    if rows.get("all_graphs_are_physically_certified") is not False:
        raise ValueError("V153_FALSE_RUNTIME_GRAPH_CERTIFICATION")
    if rows.get("actual_left_wrist_hip_physx_manifold")!="UNKNOWN":
        raise ValueError("V153_FALSE_NATIVE_MANIFOLD")
    recorded=source.get("observed_wrist_net_force_N")
    if not isinstance(recorded,list) or len(recorded)!=3:
        raise ValueError("V153_NOT_REAL_FORCE_VECTOR")
    orig_norm=math.sqrt(sum(float(t)*float(t) for t in recorded))
    if abs(orig_norm-rows["historical_force_magnitude_N"])>1e-9:
        raise ValueError("V153_HISTORICAL_VECTOR_MAGNITUDE_DRIFT")
    samples=rows.get("rows")
    if not isinstance(samples,list) or len(samples)!=288:
        raise ValueError("V153_MISSING_MATRIX_CASES")
    seen=set()
    max_error=0.
    topologies=Counter()
    fingerprints={}
    for case in samples:
        name=case["hypothesis"]
        if name not in EXPECTED:raise ValueError("V153_UNREGISTERED_TOPOLOGY")
        k=(case["direction_index"],case["norm_scale"],name)
        if k in seen:raise ValueError("V153_DUPLICATE_CASE")
        seen.add(k)
        if (type(k[0]) is not int or not 0 <= k[0] < 24
            or type(k[1]) not in (float,int) or k[1] not in (.25,1.,2.)):
            raise ValueError("V153_UNFROZEN_MATRIX_PARAMETER")
        e=case["symbolic_edges_not_physx_manifold"]
        if tuple((x["a"],x["b"]) for x in e)!=EXPECTED[name]:
            raise ValueError("V153_WRONG_CONTACT_GRAPH_EDGES")
        nodes={n for pair in EXPECTED[name] for n in pair}
        # Independent incidence reconstruction: row per vertex; column per edge.
        incidence={n:[int(pair[0]==n)-int(pair[1]==n)
                      for pair in EXPECTED[name]] for n in nodes}
        raw=[x["force_on_a_N"] for x in e]
        if any(len(x)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in x) for x in raw):
            raise ValueError("V153_BAD_INDEPENDENT_FORCE")
        computed={n:[sum(incidence[n][j]*raw[j][dim] for j in range(len(e)))
                     for dim in range(3)] for n in nodes}
        f=case["observation_wrist_N"]
        if len(f)!=3 or any(not math.isfinite(z) for z in f):
            raise ValueError("V153_NONFINITE_OBSERVATION")
        expected_pair=(f,[-x for x in f])
        for node,expected_vec,serialized in (
            (WRIST,expected_pair[0],case["net_wrist_N"]),
            (HIP,expected_pair[1],case["net_hip_N"])):
            for actual,goal,claimed in zip(computed[node],expected_vec,serialized):
                err=max(abs(actual-goal),abs(actual-claimed))
                max_error=max(max_error,err)
                if err>1e-10:raise ValueError("V153_CONTACT_GRAPH_FORCE_MISMATCH")
        if any(abs(sum(computed[n][dim] for n in computed))>1e-10 for dim in range(3)):
            raise ValueError("V153_TOTAL_ACTION_REACTION_IMBALANCE")
        direct=any({p[0],p[1]}=={WRIST,HIP} for p in EXPECTED[name])
        if direct is not case["has_wrist_hip_direct_edge"]:
            raise ValueError("V153_GRAPH_FALSE_DIRECT_EDGE_FLAG")
        fp_key=(k[0],k[1])
        existing=fingerprints.setdefault(fp_key,tuple(f))
        if any(abs(a-b)>1e-10 for a,b in zip(existing,f)):
            raise ValueError("V153_TOPOLOGIES_DIFFERENT_OBSERVATIONS")
        topologies[name]+=1
    if set(topologies)!=set(EXPECTED) or any(v!=72 for v in topologies.values()):
        raise ValueError("V153_UNBALANCED_MATRIX")
    if len(fingerprints)!=72:
        raise ValueError("V153_MISSING_DIRECTION_SCALE_COMBINATIONS")
    return {
        "status":"INDEPENDENT_INCIDENCE_AUDIT_COMPLETE",
        "cases":len(samples),"topologies":dict(topologies),
        "direction_scale_groups_with_4_indistinguishable_nets":len(fingerprints),
        "max_force_residual_N":max_error,
        "true_native_wrist_hip_collider_pair":"UNKNOWN",
        "contact_equivalence_is_algebraic_only":True,
        "no_new_physx":True,
    }

def independent_source_check(source, root=Path("/home/you/projects/kinetickick")):
    if source.get("named_pair_verification")!="NET_FORCE_ONLY":
        raise ValueError("V153_NO_HISTORICAL_NAMED_NET_FORCE")
    if source.get("original_real_physics_200hz_rows")!=6628:
        raise ValueError("V153_HISTORY_INCOMPLETE")
    if source.get("new_physx_runs")!=0:
        raise ValueError("V153_UNSUPPORTED_NEW_PHYSX_RUNS")
    h=source["historical_failed_trial_sha256"]
    path=root/"reports/v054_cmu_10_05_contact_diagnostics.json"
    if sha(path)!=h or h!="d46a48f84dbaa7a367eac3e5dc399d4378007485c369c2b57c1c67775ccbfac1":
        raise ValueError("V153_PHYSX_SOURCE_CHANGED")
    source_json=json.loads(path.read_text())
    named=source_json.get("nonfoot_contact_diagnostics",[])
    if len(named)!=2:raise ValueError("V153_NAMED_BODY_DIAGNOSTICS_CHANGED")
    wrist=next(r for r in named if r["body"]==WRIST)
    hip=next(r for r in named if r["body"]==HIP)
    if any(abs(a-b)>1e-10 for a,b in zip(wrist["force_w_N"],source["observed_wrist_net_force_N"])):
        raise ValueError("V153_INDEPENDENT_REAL_FORCE_MISMATCH")
    if any(abs(a+b)>1e-8 for a,b in zip(wrist["force_w_N"],hip["force_w_N"])):
        raise ValueError("V153_INDEPENDENT_NET_NOT_ANTISYMMETRIC")
    original=source_json["traces"][0]
    if len(original)!=1029 or abs(original[1027]["time_s"]-5.14)>1e-7:
        raise ValueError("V153_SOURCE_CENSORING_CHANGED")
    if abs(wrist["time_s"]-5.145)>1e-5 or wrist["time_s"]<=original[1027]["time_s"]:
        raise ValueError("V153_NAMED_EVENT_CLOCK_NOT_AFTER_TERMINATION")
    return {"source_sha256":h,
            "real_5ms_steps":1028,"excluded_terminal_notes":1,
            "named_net_sensor_is_later_than_final_real_state":True,
            "native_contact_manifold_certification":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-audit",type=Path,required=True)
    p.add_argument("--matrix",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    source=json.loads(a.source_audit.read_text())
    matrix=json.loads(a.matrix.read_text())
    report={"version":"v0.1.153",
            "real_source":independent_source_check(source),
            "graph_matrix":audit_matrix(matrix,source)}
    digest=safe_write_once(a.output,report)
    print(json.dumps({"independent_passed_cases":report["graph_matrix"]["cases"],
                      "source_true_pair":"UNKNOWN","sha256":digest,"new_physx":0}))

if __name__=="__main__":
    main()
