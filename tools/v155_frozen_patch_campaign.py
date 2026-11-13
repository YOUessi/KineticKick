#!/usr/bin/env python3
"""Frozen CPU campaign: 48 nominal, 96 negative, 8 v154 comparison, real history SHA audit."""
from __future__ import annotations
import argparse
import copy
import itertools
import json
from pathlib import Path

from v155_patch_wrench_qp import fixture,solve,frozen
from v155_independent_patch_audit import check
from v154_floating_base_qp import solve as point_solve
from v154_independent_wbc_audit import independent_check as point_check
from v154_cpu_feasibility_campaign import synthetic_problem,qualified_history

def corrupt(p,kind):
    x=copy.deepcopy(p)
    if kind=="missing_runtime_provenance":
        x["source_kind"]="runtime"
    elif kind=="unqualified_human_training":
        x["require_human_training"]=True
    elif kind=="missing_ground_identity":
        x["contacts"][0]["ground_only_signed_normal"]=False
    elif kind=="missing_angular_jacobian":
        x["contacts"][0]["J6"]=x["contacts"][0]["J6"][:3]
    elif kind=="bad_corner_geometry":
        x["contacts"][0]["corners_xy"][0][0]+=.02
    elif kind=="bad_normal":
        x["contacts"][0]["normal_axis"]="sole_up"
    elif kind=="nonfinite_dynamics":
        x["M"][0][0]=float("nan")
    elif kind=="wrong_original_joint_names":
        x["joint_names"][0],x["joint_names"][1]=x["joint_names"][1],x["joint_names"][0]
    elif kind=="invalid_collider_gap":
        x["guards"]=[]
    elif kind=="no_support":
        x["contacts"]=[]
    elif kind=="out_of_patch_roll_moment":
        x["h"][3]=2.
    elif kind=="bad_real_report":
        x["native_pair_claim"]=True
    else:
        raise ValueError("V155_UNDECLARED_NEGATIVE")
    return x

def run():
    cfg=frozen()
    nominal=[]
    for phase,stance,seed in itertools.product(cfg["phase_order"],cfg["supports"],cfg["seeds"]):
        p=fixture(phase,stance,seed)
        r=solve(p)
        v=check(p,r)
        nominal.append({"phase":phase,"stance":stance,"seed":seed,"solved":r["solved"],
                        "status":r["status"],"audit":v})
    negatives=[]
    for kind,stance,seed in itertools.product(cfg["negative_kinds"],cfg["supports"],cfg["seeds"]):
        p=corrupt(fixture("swing",stance,seed),kind)
        try:
            r=solve(p)
            audit=check(p,r)
            rejected=not audit["pass"]
            reason="SOLVER:"+r["status"]
        except (ValueError,KeyError) as exc:
            rejected=True
            reason=type(exc).__name__+":"+str(exc)
        negatives.append({"kind":kind,"stance":stance,"seed":seed,
                          "rejected":rejected,"reason":reason})
    baseline=[]
    for stance,seed in itertools.product(cfg["supports"],cfg["seeds"]):
        p0=synthetic_problem("swing",stance,seed)
        p0["h"][3]=2.
        raw=point_solve(p0)
        audit=point_check(p0,raw)
        p1=corrupt(fixture("swing",stance,seed),"out_of_patch_roll_moment")
        patch=solve(p1)
        patch_audit=check(p1,patch)
        baseline.append({"stance":stance,"seed":seed,
                         "point_solver_status":raw["status"],
                         "point_numeric_feasible":audit["pass"],
                         "point_base_roll_acceleration_rad_s2":raw.get("a35",[None]*35)[3],
                         "patch_solver_status":patch["status"],
                         "patch_numeric_feasible":patch_audit["pass"],
                         "physical_shot_claim":False})
    history=qualified_history()
    count=sum(x["true_historical_200hz_rows"] for x in history.values())
    if count!=6628:
        raise ValueError("V155_ORIGINAL_200HZ_PROVENANCE_DRIFT")
    out={
        "version":"v0.1.155",
        "frozen_protocol_sha":"6d9ff1530a77a7cd571060db85e415633f8c4dfb",
        "nominal_count":len(nominal),
        "nominal_passed":sum(v["audit"]["pass"] for v in nominal),
        "negative_count":len(negatives),
        "negative_rejected":sum(v["rejected"] for v in negatives),
        "baseline_count":len(baseline),
        "baseline_point_feasible":sum(v["point_numeric_feasible"] for v in baseline),
        "baseline_patch_rejected":sum(not v["patch_numeric_feasible"] for v in baseline),
        "verified_historical_200hz_rows":count,
        "verified_historical_sources":history,
        "new_physx_episodes":0,
        "native_wrist_hip_manifold":"UNKNOWN",
        "elite_human_training_complete":False,
        "actual_G1_dynamics_and_GRF_verified":False,
        "real_instep_legal_shot_success_rate":None,
        "strict_goal_ball_speed_qdot_torque_postrecover_nonfoot":"NOT_TESTED_NEW",
        "nominal":nominal,"negatives":negatives,"point_vs_patch_counterexample":baseline
    }
    if (len(nominal)!=cfg["positive_cases"] or len(negatives)!=
        len(cfg["negative_kinds"])*cfg["negative_seed_support_denominator"] or len(baseline)!=8):
        raise ValueError("V155_FROZEN_DENOMINATOR_MISMATCH")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    out=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({k:v for k,v in out.items() if k not in
                      ("nominal","negatives","verified_historical_sources",
                       "point_vs_patch_counterexample")},ensure_ascii=False))
if __name__=="__main__":
    main()
