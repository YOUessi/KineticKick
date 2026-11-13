#!/usr/bin/env python3
"""Independently validate 45-cell v141 provenance, classification and witnesses."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path

GRID={(float(m),float(t),float(w)) for m in [0,1,2,4,-1] for t in [0,1,3] for w in [.4,.8,1.2]}
SOURCE="CPU_ONLY_SYNTHETIC_ORIGINAL29_POLYNOMIAL_SCREEN"
STATES={"PARENT_QP_REJECTED","PASS","FAIL","UNRESOLVED"}
KINDS={"POSITION","ORIGINAL29_QDOT","RESEARCH_ACCEL"}
def audit(data:dict):
    if (data.get("version")!="v0.1.141" or data.get("provenance")!=SOURCE
        or data.get("protocol_sha")!="d0df1a047b058fc20d1c2696300f11ab6c370c5c"
        or data.get("source_parent_sha")!="df46cf40857f4cde177aa14cae6e93092dc3adff"
        or data.get("new_physx_rollouts")!=0
        or data.get("physical_success_rate") is not None
        or data.get("native_physx_contact_pair_certified") is not False
        or data.get("professional_human_instep_motion_learned") is not False
        or data.get("motor_execution_authorized") is not False):
        raise ValueError("V141_FORGED_PHYSICAL_PROVENANCE")
    rows=data.get("rows")
    if not isinstance(rows,list) or len(rows)!=45:
        raise ValueError("V141_INCOMPLETE_FROZEN_GRID")
    counts=Counter();seen=set()
    for row in rows:
        key=(row.get("multiplier"),row.get("target_mps"),row.get("halfwidth_s"))
        if key not in GRID or key in seen:
            raise ValueError("V141_DUPLICATE_OR_UNREGISTERED_CELL")
        seen.add(key)
        state=row.get("numeric_envelope_state")
        if (state not in STATES or type(row.get("parent_solved")) is not bool
            or type(row.get("v140_pass")) is not bool or
            type(row.get("v141_admitted")) is not bool or
            row.get("physical_success") is not False or
            row.get("native_physx_contact_pair_certified") is not False):
            raise ValueError("V141_INVALID_KINEMATIC_OR_PHYSICAL_CLASSIFICATION")
        if not isinstance(row.get("parent_qp_status"),str) or not isinstance(row.get("parent_reason"),str):
            raise ValueError("V141_MISSING_PARENT_STATUS")
        if row["v141_admitted"] != (row["parent_solved"] and row["v140_pass"] and state=="PASS"):
            raise ValueError("V141_FAKE_ENVELOPE_ADMISSION")
        if row["parent_solved"]:
            if row["parent_qp_status"]!="solved" or row["parent_reason"]!="PASS_SAMPLED_KINEMATICS_ONLY" or state=="PARENT_QP_REJECTED":
                raise ValueError("V141_PARENT_STATUS_DRIFT")
            if not isinstance(row.get("v140_veto_reasons"),list):
                raise ValueError("V141_MISSING_DENSE_VETO_REASONS")
            if row["v140_pass"]!= (len(row["v140_veto_reasons"])==0):
                raise ValueError("V141_FAKE_V140_ADMISSION")
            for name in ("polynomial_bounds_checked","bernstein_nodes_visited","max_bisection_depth",
                         "violations_count","unresolved_count"):
                x=row.get(name)
                if type(x) is not int or x<0:
                    raise ValueError("V141_BAD_ENVELOPE_COUNT")
            if row["polynomial_bounds_checked"]<29*3*7 or row["bernstein_nodes_visited"]<row["polynomial_bounds_checked"]:
                raise ValueError("V141_INCOMPLETE_INTERVAL_SCREEN")
            er=row.get("local_independent_crosscheck_max_abs_error")
            if not isinstance(er,(float,int)) or not math.isfinite(er) or not 0<=er<=2e-7:
                raise ValueError("V141_REFERENCE_CROSSCHECK_MISSING")
            if row.get("floating_point_conservative_not_formally_verified") is not True:
                raise ValueError("V141_FAKE_FORMAL_CERTIFICATE")
            if state=="FAIL" and not row["violations_count"] or state=="UNRESOLVED" and not row["unresolved_count"]:
                raise ValueError("V141_MISSING_NEGATIVE_WITNESS")
            if state=="PASS" and (row["violations_count"] or row["unresolved_count"]):
                raise ValueError("V141_FALSE_PASS")
            for plural,count in (("violation_witnesses",row["violations_count"]),("unresolved_witnesses",row["unresolved_count"])):
                if not isinstance(row.get(plural),list) or len(row[plural])!=count:
                    raise ValueError("V141_NEGATIVE_WITNESS_LIST_NOT_COMPLETE")
            for tag,req in (("first_violation",row["violations_count"]>0),
                            ("first_unresolved",row["unresolved_count"]>0)):
                w=row.get(tag)
                if bool(w)!=req:raise ValueError("V141_BAD_WITNESS_PRESENCE")
                if req and row["violation_witnesses" if tag=="first_violation" else "unresolved_witnesses"][0]!=w:
                    raise ValueError("V141_NEGATIVE_WITNESS_ORDER_DRIFT")
                if not req:continue
                if w.get("kind") not in KINDS or not isinstance(w.get("joint"),str):
                    raise ValueError("V141_BAD_WITNESS_SEMANTICS")
                seg=w.get("segment")
                if not (isinstance(seg,list) and len(seg)==2 and 0<=seg[0]<seg[1]<=14):
                    raise ValueError("V141_BAD_WITNESS_WINDOW")
                if w.get("state")!=("FAIL" if tag=="first_violation" else "UNRESOLVED"):
                    raise ValueError("V141_WRONG_WITNESS_CLASS")
                if w["state"]=="FAIL":
                    val=w.get("value");lower=w.get("lower");upper=w.get("upper");t=w.get("time_s")
                    if not all(isinstance(x,(float,int)) and math.isfinite(x) for x in (val,lower,upper,t)):
                        raise ValueError("V141_NONFINITE_NEGATIVE_WITNESS")
                    if not (seg[0]-1e-10<=t<=seg[1]+1e-10 and (val<lower-1e-8 or val>upper+1e-8)):
                        raise ValueError("V141_FAKE_NEGATIVE_WITNESS")
            counts["v140_pass"]+=int(row["v140_pass"])
            counts["numeric_envelope_pass"]+=int(state=="PASS")
            counts["numeric_envelope_fail"]+=int(state=="FAIL")
            counts["numeric_envelope_unresolved"]+=int(state=="UNRESOLVED")
            counts["newly_vetoed_from_v140"]+=int(row["v140_pass"] and not row["v141_admitted"])
        else:
            if row["v140_pass"] or row["v141_admitted"] or state!="PARENT_QP_REJECTED" or (row["parent_qp_status"]=="solved" and row["parent_reason"]!="POSTSOLVE_HARD_CONSTRAINT_VIOLATION") or (row["parent_qp_status"]!="solved" and not row["parent_reason"].startswith("QP_")):
                raise ValueError("V141_UPGRADED_REJECTED_PARENT")
            counts["parent_rejected_or_unresolved"]+=1
        counts["v141_pass"]+=int(row["v141_admitted"])
        counts["total"]+=1
    if seen!=GRID:raise ValueError("V141_MISSING_CELL")
    if dict(counts)!=data.get("summary"):
        raise ValueError("V141_SUMMARY_COUNT_DRIFT")
    return {"review":"INDEPENDENT_CLASSIFICATION_AND_NUMERIC_WITNESS_METADATA_NOT_NEW_PHYSICS",
            "total":45,"parent_rejected_or_unresolved":counts["parent_rejected_or_unresolved"],
            "dense_pass":counts["v140_pass"],"envelope_pass":counts["v141_pass"],
            "new_downgrades":counts["newly_vetoed_from_v140"],
            "numeric_fail":counts["numeric_envelope_fail"],
            "numeric_unresolved":counts["numeric_envelope_unresolved"],
            "by_target":[{"target_mps":t,
                "dense_pass":sum(r["v140_pass"] for r in rows if r["target_mps"]==t),
                "envelope_pass":sum(r["v141_admitted"] for r in rows if r["target_mps"]==t)}
                for t in (0.,1.,3.)],
            "new_real_physx_rollouts":0,"physical_success_rate":None}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("NO_AUDIT_OVERWRITE")
    result=audit(json.loads(a.input.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
