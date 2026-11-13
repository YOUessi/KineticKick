#!/usr/bin/env python3
"""Independent structural/falsification audit of v139's complete frozen grid.

Only reads written JSON. Never executes motors or assigns physical success.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

GRID_MULT=(0.,1.,2.,4.,-1.)
GRID_V=(0.,1.,3.)
GRID_W=(.4,.8,1.2)
EXPECTED={(a,b,w) for a in GRID_MULT for b in GRID_V for w in GRID_W}


def validate(a:dict)->dict:
    if (a.get("version")!="v0.1.139" or
        a.get("origin")!="SYNTHETIC_CPU_QP_NOT_PHYSICS" or
        a.get("original29_dof")!=29 or
        a.get("new_physx_rollouts")!=0 or
        a.get("human_professional_instep_learned") is not False or
        a.get("genuine_physical_shooting_success_rate") is not None):
        raise ValueError("AUDIT_FALSE_PHYSICAL_PROVENANCE")
    rows=a.get("attempts")
    if not isinstance(rows,list) or len(rows)!=45:
        raise ValueError("AUDIT_MISSING_FROZEN_ATTEMPTS")
    seen=set()
    counts=Counter()
    breakdown=[]
    for row in rows:
        combo=(row["multiplier"],row["target_mps"],row["halfwidth_s"])
        if combo not in EXPECTED or combo in seen:
            raise ValueError("AUDIT_UNREGISTERED_OR_DUPLICATE_CELL")
        seen.add(combo)
        if row["solver_iterations"]<0 or not isinstance(row["solver_status"],str):
            raise ValueError("AUDIT_INVALID_SOLVER_STATUS")
        if row["solved"]:
            if row["reason"]!="PASS_SAMPLED_KINEMATICS_ONLY":
                raise ValueError("AUDIT_FALSE_QP_PASS")
            if (row.get("full14s_200hz_samples")!=2800 or
                row["center_task_residual_mps"]>.002 or
                row["position_violation_rad"]>.0003 or
                row["velocity_violation_rad_s"]>.0003 or
                row["research_acceleration_violation_rad_s2"]>.005 or
                row["center_com_cap_violation_mps"]>.002 or
                row["center_arm_cap_violation_rad_s"]>.002 or
                row.get("kinematic_window_samples")!=331 or
                row.get("motor_execution_authorized") is not False or
                row.get("native_physx_contact_certified") is not False or
                row.get("strict_physics_success") is not False):
                raise ValueError("AUDIT_FAKE_KINEMATIC_OR_PHYSICAL_PASS")
            counts["kinematic_pass"]+=1
        else:
            if (row["reason"] is None or
                row.get("strict_physics_success") is True or
                row.get("motor_execution_authorized") is True):
                raise ValueError("AUDIT_UNTRUSTWORTHY_REJECT")
            counts["not_admitted"]+=1
            counts[row["reason"]]+=1
    if seen!=EXPECTED:
        raise ValueError("AUDIT_INCOMPLETE_FROZEN_GRID")
    for speed in GRID_V:
        for width in GRID_W:
            batch=[x for x in rows if x["target_mps"]==speed and x["halfwidth_s"]==width]
            breakdown.append({"target_ankle_origin_mps":speed,"halfwidth_s":width,
                              "attempts":len(batch),
                              "kinematic_admitted":sum(bool(x["solved"]) for x in batch),
                              "rejected":sum(not x["solved"] for x in batch),
                              "rejection_reasons":dict(Counter(x["reason"] for x in batch if not x["solved"]))})
    summary=a.get("summary",{})
    if (summary.get("attempts")!=45 or
        summary.get("qp_solved")!=counts["kinematic_pass"] or
        summary.get("qp_rejected")!=counts["not_admitted"] or
        counts["kinematic_pass"]+counts["not_admitted"]!=45):
        raise ValueError("AUDIT_SUMMARY_DENOMINATOR_MISMATCH")
    return {"origin":"INDEPENDENT_FROZEN_JSON_AUDIT_NOT_NEW_PHYSICS",
            "complete_cells":45,"sample_rate_hz":200,"original_dof":29,
            "kinematic_screen_admitted":counts["kinematic_pass"],
            "not_admitted":counts["not_admitted"],"reasons":dict(counts),
            "per_target_width":breakdown,"new_physx_rollouts":0,
            "strict_physics_success_rate":None,
            "native_pair_manifold_certified":False,
            "human_high_skill_instep_ground_truth_certified":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("REFUSE_EVIDENCE_OVERWRITE:"+str(args.output))
    result=validate(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"admitted":result["kinematic_screen_admitted"],
                      "rejected":result["not_admitted"],"attempts":45,"new_physx":0}))
if __name__=="__main__":
    main()
