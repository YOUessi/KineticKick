#!/usr/bin/env python3
"""Independent 1ms all-29DoF and true-pose forward-kinematics audit of v139 QPs.

All data remain synthetic CPU kinematics. Never touches Isaac, native PhysX or motors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np

import v135_chain29_offline as chain
import v136_g1_taskspace_mass_com as task
import v137_hard_support_relative_lp as lp
import v139_joint_window_qp as parent

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/v140_frozen_nonlinear_dense_audit.json"
FROZEN_SHA="d1334ec692a81ebb1d146e121940ff9c6a72542b"


def frozen()->dict:
    blob=subprocess.check_output(["git","-C",str(ROOT),"show",FROZEN_SHA+":configs/v140_frozen_nonlinear_dense_audit.json"])
    if CONFIG.resolve()!=(ROOT/"configs/v140_frozen_nonlinear_dense_audit.json").resolve():
        raise ValueError("V140_CONFIG_PATH_DRIFT")
    if blob!=CONFIG.read_bytes():
        raise ValueError("V140_CONFIG_BYTE_DRIFT")
    c=json.loads(blob)
    required={
        "version":"v0.1.140","parent_sha":"a9dac3c32532745f2782e4213a51a390017f8e78",
        "parent_protocol_sha":"eae275db0695ad6df6ca89a98192817a58e95bcd",
        "urdf_sha256":"34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761",
        "original_dof":29, "original_200hz_dt_s":.005,"independent_dense_dt_s":.001,
        "horizon_s":14.,"contact_proxy_time_s":2.95,
        "multipliers":[0,1,2,4,-1],"targets_mps":[0,1,3],
        "halfwidths_s":[.4,.8,1.2],"required_cells":45,
        "dense_position_violation_tolerance_rad":.0003,
        "dense_velocity_violation_tolerance_rad_s":.0003,
        "dense_research_acceleration_violation_tolerance_rad_s2":.005,
        "nonlinear_ankle_center_velocity_residual_tolerance_mps":.01,
        "nonlinear_com_center_velocity_cap_mps":.1,
        "nonlinear_com_cap_tolerance_mps":.002,
        "research_acceleration_cap_rad_s2":40.,
        "new_physx_rollouts":0,"strict_physics_success_rate":None,
    }
    if any(c.get(k)!=v for k,v in required.items()):
        raise ValueError("V140_FROZEN_PROTOCOL_DRIFT")
    if not all(c[k] for k in ("strict_original_qdot","strict_original_joint_positions",
                             "report_all_original_QP_failures","run_independent_provenance_validation",
                             "no_gpu_if_occupied")):
        raise ValueError("V140_UNSAFE_FALSE_CONSTRAINT")
    if any(c[k] is not False for k in ("native_physx_contact_pair_certified",
                                      "true_world_base_or_GRF_available",
                                      "physical_motor_execution_authorized",
                                      "elite_human_instep_kinetic_truth_available")):
        raise ValueError("V140_FABRICATED_PHYSICAL_WITNESS")
    if c["strict_physics_gates"]!={"instep_impulse_fraction_min":.95,
        "ball_speed_mps_min":3.5,"all_29_qdot_legal":True,
        "legal_goal":True,"no_nonfoot_collision":True,"full_14s_recovery":True}:
        raise ValueError("V140_PHYSICS_GATE_DRIFT")
    return c


def analytic_reference(t:np.ndarray,q0:np.ndarray,mult:float,base:dict):
    """Recreate *exact* seven-phase v135 synthetic trajectory at arbitrary timestamps."""
    t=np.asarray(t,dtype=np.float64)
    q0=np.asarray(q0,dtype=np.float64)
    if (t.ndim!=1 or len(t)==0 or q0.shape!=(29,) or
        not np.isfinite(t).all() or not np.isfinite(q0).all() or
        np.any(t<0) or np.any(t>=14) or
        isinstance(mult,bool) or mult not in base["multipliers"]):
        raise ValueError("INVALID_DENSE_ANALYTIC_REFERENCE")
    durations=np.asarray(base["phase_durations_s"],float)
    edges=np.r_[0.,np.cumsum(durations)]
    knots=np.asarray(chain.KNOTS,float)
    if knots.shape!=(8,4) or len(durations)!=7 or abs(edges[-1]-14)>1e-12:
        raise ValueError("INVALID_SEVEN_PHASE_BOUNDARIES")
    inds=np.clip(np.searchsorted(edges,t,side="right")-1,0,6)
    u=(t-edges[inds])/durations[inds]
    if np.min(u)<-1e-10 or np.max(u)>1+1e-10:
        raise ValueError("BAD_DENSE_PHASE_TIME")
    v,vd,vdd=chain.smooth(u)
    dz=knots[inds+1]-knots[inds]
    z=knots[inds]+v[:,None]*dz
    zd=vd[:,None]*dz/durations[inds,None]
    zdd=vdd[:,None]*dz/durations[inds,None]**2
    weights=chain.weight_matrix()
    amps=np.asarray(base["latent_amplitudes_rad"])*mult
    q=q0+(z*amps)@weights.T
    dq=(zd*amps)@weights.T
    ddq=(zdd*amps)@weights.T
    return q,dq,ddq


def recover_coefficients(reference, candidate:dict, center:float, width:float):
    """Independently identify four C2 coefficients from the *reported* 200Hz witness."""
    q,dq,ddq=(candidate[x] for x in ("_q","_dq","_ddq"))
    B,D,DD=parent.basis(reference.t,center,width)
    delta=q-reference.q
    if not all(np.isfinite(a).all() for a in (delta,dq,ddq)):
        raise ValueError("NONFINITE_PARENT_WITNESS")
    c,_,rank,_=np.linalg.lstsq(B,delta,rcond=None) # coefficient shape: 4 x 29
    if rank!=4 or c.shape!=(4,29):
        raise ValueError("V139_BASIS_RANK_FAILURE")
    error=(np.max(abs(B@c-delta)),
           np.max(abs(D@c-(dq-reference.dq))),
           np.max(abs(DD@c-(ddq-reference.ddq))))
    if error[0]>1e-7 or error[1]>1e-6 or error[2]>1e-5:
        raise ValueError("RECONSTRUCTION_CANNOT_ATTEST_ORIGINAL_SOLVER")
    return c,tuple(float(x) for x in error)


def exact_center_kinematics(tree,masses,q:np.ndarray,dq:np.ndarray,pc:dict):
    if q.shape!=(29,) or dq.shape!=(29,) or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError("BAD_NONLINEAR_CENTER_29DOF")
    left,Jl=task.point_jacobian(tree,q,pc["virtual_left_anchor_link"])
    right,Jr=task.point_jacobian(tree,q,pc["relative_swing_link"])
    com,Jc=lp.inertial_com_and_analytic_jacobian(tree,masses,q)
    return {
        "relative_right_left_origin_velocity_xyz_mps":((Jr-Jl)@dq).tolist(),
        "com_relative_virtual_left_origin_velocity_xyz_mps":((Jc-Jl)@dq).tolist(),
        "reference_root_fixed_left_ankle_origin_xyz_m":left.tolist(),
        "reference_root_fixed_right_ankle_origin_xyz_m":right.tolist(),
        "reference_root_fixed_com_xyz_m":com.tolist(),
    }


def dense_evaluate(p, witness:dict, lim:dict, tree,masses,pc:dict,c:dict,width:float,target:float):
    center=c["contact_proxy_time_s"]
    coeff,rec_error=recover_coefficients(p,witness,center,width)
    dt=c["independent_dense_dt_s"]
    t=np.arange(round(c["horizon_s"]/dt),dtype=np.float64)*dt
    q0=(lim["lo"]+lim["hi"])/2  # independently recover the unmodified synthetic anchor
    qb,db,ab=analytic_reference(t,q0,p.multiplier,chain.frozen())
    B,D,DD=parent.basis(t,center,width)
    q=qb+B@coeff
    dq=db+D@coeff
    ddq=ab+DD@coeff
    if q.shape!=(14000,29) or not all(np.isfinite(x).all() for x in (q,dq,ddq)):
        raise ValueError("DENSE_BAD_14S_ORIGINAL29")
    ci=round(center/dt)
    kin=exact_center_kinematics(tree,masses,q[ci],dq[ci],pc)
    expected=np.array([target,0.,0.])
    target_res=float(np.max(abs(np.asarray(kin["relative_right_left_origin_velocity_xyz_mps"])-expected)))
    com_speed=float(np.max(np.abs(np.asarray(kin["com_relative_virtual_left_origin_velocity_xyz_mps"]))))
    err_q=max(0.,float(np.max(lim["lo"]-q)),float(np.max(q-lim["hi"])))
    err_dq=max(0.,float(np.max(abs(dq)-lim["vmax"])))
    err_ddq=max(0.,float(np.max(abs(ddq)-c["research_acceleration_cap_rad_s2"])))
    issues=[]
    if err_q>c["dense_position_violation_tolerance_rad"]:issues.append("DENSE_POSITION")
    if err_dq>c["dense_velocity_violation_tolerance_rad_s"]:issues.append("DENSE_ORIGINAL29_QDOT")
    if err_ddq>c["dense_research_acceleration_violation_tolerance_rad_s2"]:issues.append("DENSE_RESEARCH_ACCEL")
    if target_res>c["nonlinear_ankle_center_velocity_residual_tolerance_mps"]:issues.append("EXACT_POSE_ANKLE_VELOCITY")
    if com_speed>c["nonlinear_com_center_velocity_cap_mps"]+c["nonlinear_com_cap_tolerance_mps"]:
        issues.append("EXACT_POSE_VIRTUAL_COM")
    # At exact center, every compact odd displacement basis is zero; the 'linearized'
    # center Jacobian from v139 is therefore mathematically evaluated at the TRUE pose.
    center_delta=float(np.max(abs(q[ci]-p.q[round(center/.005)])))
    return {
        "dense_samples":len(t),"dense_dt_s":dt,"center_joint_pose_change_rad":center_delta,
        "basis_reconstruction_error_position_velocity_acceleration":rec_error,
        "dense_max_position_bound_violation_rad":err_q,
        "dense_max_original29_velocity_violation_rad_s":err_dq,
        "dense_max_research_accel_violation_rad_s2":err_ddq,
        "dense_max_research_accel_rad_s2":float(np.max(abs(ddq))),
        "exact_pose_center_task_residual_mps":target_res,
        "exact_pose_center_com_relative_speed_max_abs_mps":com_speed,
        "exact_pose_center":kin,
        "new_kinematic_admitted":not issues,"veto_reasons":issues,
        "native_physx_pair_certified":False,"motor_execution_authorized":False,
        "strict_physics_success":False,
    }


def run():
    c=frozen()
    pc=lp.frozen()
    p139=parent.frozen()
    if c["parent_protocol_sha"]!=parent.FROZEN_SHA:
        raise ValueError("PARENT_PROTOCOL_DISAGREE")
    tree,lim=lp.original_kinematics(task.URDF,pc)
    masses=task.mass_inertials(task.URDF,tree)
    base=chain.frozen()
    q0=(lim["lo"]+lim["hi"])/2
    attempts=[]
    breakdown=Counter()
    for mult in c["multipliers"]:
        p=chain.generate(base,q0,mult)
        idx=round(c["contact_proxy_time_s"]/c["original_200hz_dt_s"])
        problem=lp.problem_at_pose(tree,masses,p.q[idx],pc)
        for target in c["targets_mps"]:
            for width in c["halfwidths_s"]:
                original=parent.solve(p,problem,lim,p139,target,width)
                row={"multiplier":float(mult),"target_mps":float(target),"halfwidth_s":float(width),
                     "parent_qp_status":original["solver_status"],"parent_solved":bool(original["solved"]),
                     "parent_reason":original["reason"],
                     "new_kinematic_admitted":False,"veto_reasons":[]}
                if original["solved"]:
                    row.update(dense_evaluate(p,original,lim,tree,masses,pc,c,width,target))
                    breakdown["parent_admitted"]+=1
                    breakdown["new_kinematic_admitted"]+=int(row["new_kinematic_admitted"])
                    for r in row["veto_reasons"]:breakdown[r]+=1
                else:
                    breakdown["parent_rejected"]+=1
                    row["veto_reasons"]=["PARENT_QP_NOT_ADMITTED"]
                attempts.append(row)
    if len(attempts)!=c["required_cells"] or len(set((r["multiplier"],r["target_mps"],r["halfwidth_s"]) for r in attempts))!=45:
        raise RuntimeError("INCOMPLETE_FROZEN_GRID")
    return {"version":c["version"],"protocol_sha":FROZEN_SHA,
        "parent_sha":c["parent_sha"],"urdf_sha256":c["urdf_sha256"],
        "origin":"CPU_ONLY_SYNTHETIC_29DOF_KINEMATICS_NOT_PHYSICS",
        "summary":{"total_attempts":len(attempts),**dict(breakdown),
            "new_kinematic_rejected_or_unresolved":45-breakdown["new_kinematic_admitted"]},
        "attempts":attempts,"new_physx_rollouts":0,"strict_physics_success_rate":None,
        "native_contact_pair_certified":False,"human_elite_instep_kinetics_truth":False,
        "motor_execution_authorized":False}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--output",type=Path,required=True)
    args=a.parse_args()
    if args.output.exists():raise FileExistsError("REFUSE_EVIDENCE_OVERWRITE:"+str(args.output))
    r=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(json.dumps(r["summary"],sort_keys=True))
    print("WROTE",args.output)


if __name__=="__main__":
    main()
