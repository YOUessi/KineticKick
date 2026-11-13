#!/usr/bin/env python3
"""Independent (source-file) comparison of two frozen CPU QP executions.

This does not pretend OSQP floating-point outputs are bitwise deterministic.
No hidden tolerance and no pass/fail reclassification; report raw differences.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

def numeric_max_diff(x,y):
    if isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)) and not isinstance(y,bool):
        return abs(float(x)-float(y))
    if isinstance(x,list) and isinstance(y,list) and len(x)==len(y):
        return max((numeric_max_diff(a,b) for a,b in zip(x,y)),default=0.)
    if isinstance(x,dict) and isinstance(y,dict) and set(x)==set(y):
        return max((numeric_max_diff(x[k],y[k]) for k in x),default=0.)
    if x==y:return 0.
    raise ValueError("V158_NONSEMANTIC_ROW_DRIFT")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a",type=Path,required=True)
    ap.add_argument("--b",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    p=ap.parse_args()
    a=json.loads(p.a.read_text());b=json.loads(p.b.read_text())
    if len(a["case_rows"])!=96 or len(b["case_rows"])!=96:
        raise ValueError("V158_FROZEN_96_CASES_MISSING")
    if a["original_urdf_sha256"]!=b["original_urdf_sha256"]:
        raise ValueError("V158_SOURCE_SHA_CHANGED")
    if a["numerical_outcomes"]!=b["numerical_outcomes"]:
        raise ValueError("V158_SOLVER_STATUS_DRIFT")
    diff=numeric_max_diff(a,b)
    row_delta=[]
    for x,y in zip(a["case_rows"],b["case_rows"]):
        for key in ("phase","mode","seed","effort_condition","status","source_urdf_sha256"):
            if x[key]!=y[key]:raise ValueError("V158_CASE_ID_DRIFT_"+key)
        row_delta.append(numeric_max_diff(x,y))
    out=dict(
        source_urdf_sha256=a["original_urdf_sha256"],
        first_sha256=hashlib.sha256(p.a.read_bytes()).hexdigest(),
        second_sha256=hashlib.sha256(p.b.read_bytes()).hexdigest(),
        both_original_source_96_cases=True,
        byte_identical=p.a.read_bytes()==p.b.read_bytes(),
        numerical_outcomes=a["numerical_outcomes"],
        max_case_numeric_difference=diff,
        max_per_case_delta=max(row_delta),
        cases_with_nonzero_numeric_differences=sum(z>0 for z in row_delta),
        numerical_difference_is_not_new_human_or_PhysX_trial=True,
        no_recomputed_threshold_or_tuning=True,
        real_physics_success_count=None,
    )
    payload=(json.dumps(out,indent=2,ensure_ascii=False)+"\n").encode()
    p.output.parent.mkdir(parents=True,exist_ok=True)
    with p.output.open("xb") as f:f.write(payload)
    print(json.dumps(out,indent=2))
    print("INDEPENDENT_REPRO_SHA256",hashlib.sha256(payload).hexdigest())

if __name__=="__main__":main()
