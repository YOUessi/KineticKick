#!/usr/bin/env python3
"""Independently reject forged 45-cell synthetic dense-kinematics result summaries."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path

MULT=(0.,1.,2.,4.,-1.)
TARGET=(0.,1.,3.)
WIDTH=(.4,.8,1.2)
GRID={(a,b,w) for a in MULT for b in TARGET for w in WIDTH}
EPS=1e-12
VALID_VETO={"DENSE_POSITION","DENSE_ORIGINAL29_QDOT","DENSE_RESEARCH_ACCEL",
            "EXACT_POSE_ANKLE_VELOCITY","EXACT_POSE_VIRTUAL_COM"}
TOL={"dense_max_position_bound_violation_rad":.0003,
     "dense_max_original29_velocity_violation_rad_s":.0003,
     "dense_max_research_accel_violation_rad_s2":.005,
     "exact_pose_center_task_residual_mps":.01}
def audit(source:dict):
    if any((
        source.get("version")!="v0.1.140",
        source.get("origin")!="CPU_ONLY_SYNTHETIC_29DOF_KINEMATICS_NOT_PHYSICS",
        source.get("new_physx_rollouts")!=0,
        source.get("strict_physics_success_rate") is not None,
        source.get("native_contact_pair_certified") is not False,
        source.get("human_elite_instep_kinetics_truth") is not False,
        source.get("motor_execution_authorized") is not False,
    )):raise ValueError("FORGED_PHYSICAL_PROVENANCE")
    rows=source.get("attempts")
    if not isinstance(rows,list) or len(rows)!=45:raise ValueError("MISSING_GRID_CELLS")
    seen=set();stats=Counter(); breakdown=[]
    for row in rows:
        key=(row["multiplier"],row["target_mps"],row["halfwidth_s"])
        if key not in GRID or key in seen:raise ValueError("DUPLICATE_OR_UNREGISTERED_GRID_CELL")
        seen.add(key)
        passed=row.get("new_kinematic_admitted")
        if type(passed) is not bool or type(row.get("parent_solved")) is not bool:
            raise ValueError("INVALID_KINEMATIC_CLASSIFICATION")
        if not isinstance(row.get("parent_qp_status"),str) or not isinstance(row.get("parent_reason"),str):
            raise ValueError("INVALID_PARENT_SOLVER_PROVENANCE")
        if passed and not row["parent_solved"]:raise ValueError("FALSE_UPGRADE_FROM_FAILED_QP")
        reasons=row.get("veto_reasons")
        if not isinstance(reasons,list) or len(reasons)!=len(set(reasons)):
            raise ValueError("MALFORMED_VETO_REASONS")
        if row["parent_solved"]:
            if row["parent_reason"]!="PASS_SAMPLED_KINEMATICS_ONLY":
                raise ValueError("FORGED_PARENT_QP_PASS")
            if row.get("dense_samples")!=14000 or abs(row.get("dense_dt_s",0)-.001)>EPS:
                raise ValueError("FAKE_DENSE_GRID")
            if row.get("native_physx_pair_certified") is not False or row.get("motor_execution_authorized") is not False or row.get("strict_physics_success") is not False:
                raise ValueError("FAKE_PHYSICS_FROM_KINEMATICS")
            for k in TOL:
                if not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) or row[k]<0:
                    raise ValueError("INVALID_DENSE_NUMERIC")
            speed=row.get("exact_pose_center_com_relative_speed_max_abs_mps")
            if not isinstance(speed,(int,float)) or not math.isfinite(speed) or speed<0:
                raise ValueError("INVALID_COM_SPEED")
            expected=[k for k,limit in TOL.items() if row[k]>limit]
            expected.extend(["EXACT_POSE_VIRTUAL_COM"] if speed>.102 else [])
            alias={"dense_max_position_bound_violation_rad":"DENSE_POSITION",
              "dense_max_original29_velocity_violation_rad_s":"DENSE_ORIGINAL29_QDOT",
              "dense_max_research_accel_violation_rad_s2":"DENSE_RESEARCH_ACCEL",
              "exact_pose_center_task_residual_mps":"EXACT_POSE_ANKLE_VELOCITY"}
            correct={alias.get(k,k) for k in expected}
            if set(reasons)!=correct or any(x not in VALID_VETO for x in reasons):
                raise ValueError("MISCLASSIFIED_FROZEN_PHYSICS_VETO")
            if passed!= (not correct):raise ValueError("FALSE_DENSE_KINEMATIC_PASS")
            if abs(row.get("center_joint_pose_change_rad",-1))>1e-10:
                raise ValueError("CENTER_ODD_BASIS_Q_MUST_BE_UNCHANGED")
            stats["parent_admitted"]+=1
            stats["new_kinematic_admitted"]+=int(passed)
            for reason in correct:stats[reason]+=1
        else:
            if passed or reasons!=["PARENT_QP_NOT_ADMITTED"]:
                raise ValueError("UNJUSTIFIED_UPGRADE_OR_DROP")
            stats["parent_rejected"]+=1
    if seen!=GRID:raise ValueError("INCOMPLETE_GRID_DENOMINATOR")
    summary=source.get("summary")
    if not isinstance(summary,dict) or summary.get("total_attempts")!=45:
        raise ValueError("BAD_SUMMARY_DENOMINATOR")
    for k,v in stats.items():
        if summary.get(k)!=v:raise ValueError("BAD_SUMMARY_COUNT:"+k)
    if (summary.get("new_kinematic_rejected_or_unresolved")!=45-stats["new_kinematic_admitted"] or
        stats["parent_admitted"]+stats["parent_rejected"]!=45):
        raise ValueError("BAD_SUMMARY_ADMISSION_COUNT")
    for target in TARGET:
        breakdown.append(dict(target_mps=target,
            parent_pass=sum(x["parent_solved"] for x in rows if x["target_mps"]==target),
            nonlinear_dense_pass=sum(x["new_kinematic_admitted"] for x in rows if x["target_mps"]==target)))
    return dict(origin="INDEPENDENT_CPU_REVIEW_NO_NEW_PHYSICS",
        all_frozen_cells=45, parent_pass=stats["parent_admitted"],
        strict_dense_nonlinear_pass=stats["new_kinematic_admitted"],
        original_QP_rejected=stats["parent_rejected"],
        additional_veto=stats["parent_admitted"]-stats["new_kinematic_admitted"],
        veto_counts={key:stats[key] for key in sorted(VALID_VETO)},
        per_target=breakdown,new_real_physx_shots=0,
        strict_real_physics_success_rate=None,
        native_collider_pair_certified=False,
        high_skill_human_instep_learning_certified=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("EVIDENCE_FILE_EXISTS")
    result=audit(json.loads(a.input.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    main()
