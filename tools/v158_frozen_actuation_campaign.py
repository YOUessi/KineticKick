#!/usr/bin/env python3
"""v158 frozen 48 original-URDF source states × 2 torque-limit counterfactuals.

Write-once CPU JSON; keeps negative, infeasible and historical truth gaps.
Does not run or interact with PhysX/GPU, and cannot command the G1.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from v156_real_urdf_pin_model import OriginalUrdfModel, independent_invariants
from v156_independent_runtime_and_history_gate import independent_real_history_audit
from v158_source_actuation_wrench import (
    frozen,make_case,solve,validate_case,independently_audit_solution
)

ROOT = Path(__file__).resolve().parents[1]

def damage(p, kind):
    q = copy.deepcopy(p)
    if kind=="wrong_joint_names":
        q["original_joint_names"][0],q["original_joint_names"][1]=q["original_joint_names"][1],q["original_joint_names"][0]
    elif kind=="wrong_urdf_digest":
        q["source_urdf_sha256"]="0"*64
    elif kind=="invalid_root_quaternion":
        q["q36"][6]=2.0
    elif kind=="missing_support":
        q["contacts"]=[]
    elif kind=="fake_runtime_contact":
        q["source_kind"]="runtime"
        q["runtime_ground_force_certified"]=True
    elif kind=="fake_human_supervision":
        q["human_training_qualified"]=True
    elif kind=="wrong_actuator_limit":
        q["torque_caps"][0]*=3
    elif kind=="nonfinite_bias":
        q["h"][4]=float("nan")
    else: raise ValueError("V158_UNKNOWN_NEGATIVE")
    return q

def run():
    cfg=frozen()
    model=OriginalUrdfModel()
    data={"version":"v0.1.158",
          "frozen_commit":"aa9deaeba02024f175af1d7b9918d1764d5cd01a",
          "original_urdf_sha256":model.sha,
          "experimental_type":"offline original URDF 35D hypothetical stance, NOT PhysX",
          "new_physx_episodes":0,"new_human_motion_training_episodes":0,
          "case_rows":[],"negative_rejections":[],
          "independent_dynamics_results":[],
          "real_history_sha_audit":independent_real_history_audit(),
          "physx_contact_pair_manifold":"UNKNOWN",
          "physical_high_speed_legal_instep_success":None,
          "robot_control_permission":False}
    index=0
    for phase in cfg["phases"]:
        for mode in cfg["support_modes"]:
            for seed in cfg["seeds"]:
                for condition in cfg["effort_conditions"]:
                    p=make_case(model,phase,mode,seed,condition)
                    if index//2 < cfg["independent_dynamics_checks"] and condition==cfg["effort_conditions"][0]:
                        qq=np.asarray(p["q36"]);vv=np.asarray(p["v35"])
                        inv=independent_invariants(model,qq,vv,model.evaluate(qq,vv))
                        data["independent_dynamics_results"].append({
                            "phase":phase,"mode":mode,"seed":seed,**inv})
                    r=solve(p,model)
                    r["independent_audit"]=independently_audit_solution(p,r,model)
                    data["case_rows"].append(r)
                    index+=1
    for seed in (0,1):
        base=make_case(model,"approach","left",seed,"urdf_effort")
        for k in cfg["negative_kinds"]:
            try:
                validate_case(damage(base,k),model)
            except ValueError as ex:
                data["negative_rejections"].append({"seed":seed,"kind":k,"rejected":True,"reason":str(ex)})
            else:
                raise AssertionError("V158_NEGATIVE_NOT_REJECTED:"+k)
    if (len(data["case_rows"]),len(data["negative_rejections"]),
            len(data["independent_dynamics_results"])) != (cfg["expected_qp_solutions"],
                cfg["expected_negative_rejections"],cfg["independent_dynamics_checks"]):
        raise AssertionError("V158_FROZEN_MATRIX_COUNT_CHANGED")
    data["numerical_outcomes"]=dict(Counter(r["status"] for r in data["case_rows"]))
    paired=[]
    for x in range(0,len(data["case_rows"]),2):
        a,b=data["case_rows"][x:x+2]
        if (a["phase"],a["mode"],a["seed"])!=(b["phase"],b["mode"],b["seed"]):
            raise AssertionError("V158_EFFORT_ABLATION_PAIR_MISMATCH")
        paired.append({
            "phase":a["phase"],"mode":a["mode"],"seed":a["seed"],
            "nominal_solved":a["solved_numerically"],"relaxed_solved":b["solved_numerically"],
            "tracking_rmse_nominal":a.get("tracking_rmse_rad_s2"),
            "tracking_rmse_relaxed":b.get("tracking_rmse_rad_s2"),
            "delta_relaxed_minus_nominal_rmse":(
                b["tracking_rmse_rad_s2"]-a["tracking_rmse_rad_s2"]
                if a["solved_numerically"] and b["solved_numerically"] else None)})
    data["ablation_pairs"]=paired
    delta=[p["delta_relaxed_minus_nominal_rmse"] for p in paired if p["delta_relaxed_minus_nominal_rmse"] is not None]
    data["ablation_metrics"]={
        "fully_numeric_pairs":len(delta),
        "max_absolute_tracking_rmse_change":max((abs(z) for z in delta),default=None),
        "mean_tracking_rmse_delta_relaxed_minus_nominal":float(np.mean(delta)) if delta else None,
        "nominal_numerical_solved":sum(p["nominal_solved"] for p in paired),
        "relaxed_numerical_solved":sum(p["relaxed_solved"] for p in paired),
        "warning_numerical_optimization_is_NOT_physical_strike_success":True}
    return data

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    results=run()
    path=args.output
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as f:f.write(raw)
    print(json.dumps({"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),
                      "outcomes":results["numerical_outcomes"],
                      "ablation":results["ablation_metrics"],
                      "negative_controls":len(results["negative_rejections"]),
                      "original_history_states":results["real_history_sha_audit"]["actual_original_200hz_states"],
                      "new_physx_episodes":0},indent=2))

if __name__=="__main__": main()
