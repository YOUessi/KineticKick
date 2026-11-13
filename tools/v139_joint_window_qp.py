#!/usr/bin/env python3
"""v139: strictly preregistered CPU-only full-window 29DoF convex QP.

No PhysX, no learned human motion, no GRF / true support or motor commands.
The 40rad/s^2 bound is a research screen, NOT an actuator specification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sp

import v135_chain29_offline as chain
import v136_g1_taskspace_mass_com as task
import v137_hard_support_relative_lp as lp
import v138_temporal_full29 as prev

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v139_frozen_joint_window_qp.json"
FROZEN_SHA = "eae275db0695ad6df6ca89a98192817a58e95bcd"


def frozen() -> dict:
    raw = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                   FROZEN_SHA + ":configs/v139_frozen_joint_window_qp.json"])
    if CONFIG.resolve() != (ROOT / "configs/v139_frozen_joint_window_qp.json").resolve():
        raise ValueError("V139_CONFIG_PATH_DRIFT")
    if raw != CONFIG.read_bytes():
        raise ValueError("V139_CONFIG_BYTE_DRIFT")
    c = json.loads(raw)
    expected = {
        "version":"v0.1.139",
        "parent_sha":"a3a119932978be7e18b3483e06add87bb89dbd9c",
        "urdf_sha256":"34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761",
        "original_dof":29, "sample_period_s":.005, "horizon_s":14,
        "multipliers":[0,1,2,4,-1], "contact_proxy_time_s":2.95,
        "relative_right_minus_left_ankle_origin_x_targets_mps":[0,1,3],
        "relative_yz_targets_mps":[0,0],
        "compact_halfwidths_s":[.4,.8,1.2],
        "basis_odd_powers":[1,3,5,7],
        "research_only_qddot_screen_hard_rad_s2":40,
        "com_relative_to_virtual_left_ankle_center_speed_cap_mps":.1,
        "arm_joint_center_qdot_change_cap_rad_s":.25,
        "solver":"osqp", "solver_eps_abs":1e-5, "solver_eps_rel":1e-5,
        "solver_max_iter":20000, "solver_polishing":True,
        "required_grid_cells":45, "actual_new_physx_rollouts":0,
    }
    if any(c.get(k) != v for k,v in expected.items()):
        raise ValueError("V139_FROZEN_GRID_DRIFT")
    if (not c["original29_joint_position_hard_bounds"] or
        not c["original29_joint_velocity_hard_bounds"] or
        c["motor_execution_authorized"] or c["native_contact_pair_certified"] or
        c["real_grf_or_cop_available"] or c["human_professional_instep_force_truth_available"] or
        not c["report_all_solver_failures"] or
        c["strict_physics_gates"]["instep_impulse_fraction_min"] != .95):
        raise ValueError("V139_PHYSICAL_PROVENANCE_DRIFT")
    return c


def basis(t: np.ndarray, center: float, width: float, powers=(1,3,5,7)):
    """Compact C2 basis and *analytic* first and second time derivatives."""
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or not np.isfinite(t).all() or not np.isfinite(center) or width <= 0:
        raise ValueError("BAD_BASIS_TIME")
    s = (t-center)/width
    B=np.zeros((len(t),len(powers)))
    D=np.zeros_like(B)
    DD=np.zeros_like(B)
    # Snap floating-point roundoff at the mathematically compact endpoints.
    mask=np.abs(s)<1-1e-12
    x=s[mask]
    g=1-x*x
    for j,n in enumerate(powers):
        if n<1 or n%2!=1:
            raise ValueError("BAD_ODD_BASIS_POWER")
        B[mask,j] = width*x**n*g**4
        D[mask,j] = n*x**(n-1)*g**4 - 8*x**(n+1)*g**3
        DD[mask,j] = ((n*(n-1)*x**(n-2)*g**4 if n>1 else 0)
                      -8*(2*n+1)*x**n*g**3 +48*x**(n+2)*g**2)/width
    return B,D,DD


def constrained_problem(reference, problem: dict, lim: dict, c: dict,
                        target: float, width: float) -> dict:
    """One sparse QP for every joint coefficient across the *entire* window."""
    if isinstance(target,bool) or target not in c["relative_right_minus_left_ankle_origin_x_targets_mps"]:
        raise ValueError("UNFROZEN_V139_TARGET")
    if isinstance(width,bool) or width not in c["compact_halfwidths_s"]:
        raise ValueError("UNFROZEN_V139_WIDTH")
    if (reference.q.shape!=(2800,29) or reference.dq.shape!=(2800,29) or
        reference.ddq.shape!=(2800,29) or not np.isfinite(reference.q).all() or
        not np.isfinite(reference.dq).all() or not np.isfinite(reference.ddq).all()):
        raise ValueError("BAD_14S_FULL29_REFERENCE")
    tm=c["contact_proxy_time_s"]; dt=c["sample_period_s"]
    ci=round(tm/dt)
    if abs(reference.t[ci]-tm)>1e-12:
        raise ValueError("INVALID_SYNTHETIC_CLOCK")
    idx=np.where(np.abs(reference.t-tm)<=width+1e-12)[0]
    B,D,DD=basis(reference.t[idx],tm,width,tuple(c["basis_odd_powers"]))
    bc,dc,_=basis(np.array([tm]),tm,width,tuple(c["basis_odd_powers"]))
    k=B.shape[1]; n=29; m=len(idx)
    I=sp.eye(n,format="csc")
    Ab=sp.kron(I,sp.csc_matrix(B),format="csc")
    Ad=sp.kron(I,sp.csc_matrix(D),format="csc")
    Add=sp.kron(I,sp.csc_matrix(DD),format="csc")
    flatten=lambda z: np.asarray(z,dtype=float).T.ravel()
    qb=flatten(reference.q[idx]); db=flatten(reference.dq[idx]); ddb=flatten(reference.ddq[idx])
    lq=np.repeat(lim["lo"],m)-qb
    uq=np.repeat(lim["hi"],m)-qb
    lv=np.repeat(-lim["vmax"],m)-db
    uv=np.repeat(lim["vmax"],m)-db
    cap=float(c["research_only_qddot_screen_hard_rad_s2"])
    la=np.full(n*m,-cap)-ddb
    ua=np.full(n*m,cap)-ddb
    J=problem["Jrelative"]
    C=problem["Jcom_relative"]
    if J.shape!=(3,n) or C.shape!=(3,n):
        raise ValueError("MALFORMED_ORIGINAL29_TASK_JACOBIAN")
    task_rows=sp.kron(sp.csc_matrix(J),sp.csc_matrix(dc),format="csc")
    desired=np.array([target,0.,0.])-J@reference.dq[ci]
    com_rows=sp.kron(sp.csc_matrix(C),sp.csc_matrix(dc),format="csc")
    com_baseline=C@reference.dq[ci]
    com_cap=c["com_relative_to_virtual_left_ankle_center_speed_cap_mps"]
    arms=np.eye(n)[lp.ARM]
    arm_rows=sp.kron(sp.csc_matrix(arms),sp.csc_matrix(dc),format="csc")
    arm_cap=c["arm_joint_center_qdot_change_cap_rad_s"]
    A=sp.vstack([Ab,Ad,Add,task_rows,com_rows,arm_rows],format="csc")
    lo=np.r_[lq,lv,la,desired,-com_cap-com_baseline, np.full(len(lp.ARM),-arm_cap)]
    hi=np.r_[uq,uv,ua,desired, com_cap-com_baseline, np.full(len(lp.ARM),arm_cap)]
    obj=c["quadratic_objective"]
    Q=(obj["position_delta_weight"]*(B.T@B) +
       obj["velocity_delta_weight"]*(D.T@D) +
       obj["acceleration_delta_weight"]*(DD.T@DD))*dt
    Q+=obj["coefficient_ridge"]*np.eye(k)
    P=sp.kron(sp.diags(1/lim["vmax"]**2,format="csc"),sp.csc_matrix(Q),format="csc")
    return dict(A=A, lo=lo, hi=hi, P=P, n=n, k=k, ci=ci, idx=idx,
                B=B, D=D, DD=DD, task_rows=task_rows, com_rows=com_rows,
                J=J,C=C,desired=desired,com_cap=com_cap,arm_cap=arm_cap)


def solve(reference, problem: dict, lim: dict, c: dict,
          target: float, width: float) -> dict:
    qp=constrained_problem(reference,problem,lim,c,target,width)
    os=osqp.OSQP()
    os.setup(P=sp.triu(qp["P"]).tocsc(),q=np.zeros(29*qp["k"]),
             A=qp["A"],l=qp["lo"],u=qp["hi"],
             verbose=False,polish=c["solver_polishing"],
             eps_abs=c["solver_eps_abs"],eps_rel=c["solver_eps_rel"],
             max_iter=c["solver_max_iter"])
    result=os.solve()
    status=str(result.info.status)
    out=dict(solver_status=status,solver_iterations=int(result.info.iter),
             target_mps=float(target),halfwidth_s=float(width),
             solved=False,reason=None)
    if status!="solved" or result.x is None or not np.isfinite(result.x).all():
        out["reason"]="QP_"+status.upper().replace(" ","_")
        return out
    coeff=np.asarray(result.x,float).reshape(29,qp["k"]).T
    qb=qp["B"]@coeff
    vb=qp["D"]@coeff
    ab=qp["DD"]@coeff
    q=reference.q.copy(); dq=reference.dq.copy(); ddq=reference.ddq.copy()
    q[qp["idx"]]+=qb; dq[qp["idx"]]+=vb; ddq[qp["idx"]]+=ab
    qv=max(0.,float(np.max(lim["lo"]-q)),float(np.max(q-lim["hi"])))
    vv=max(0.,float(np.max(abs(dq)-lim["vmax"])))
    av=max(0.,float(np.max(abs(ddq)-c["research_only_qddot_screen_hard_rad_s2"])))
    task_res=float(np.max(abs(qp["J"]@dq[qp["ci"]]-np.array([target,0,0]))))
    com_v=max(0.,float(np.max(abs(qp["C"]@dq[qp["ci"]]))-qp["com_cap"]))
    arm_v=max(0.,float(np.max(abs(dq[qp["ci"],lp.ARM]-reference.dq[qp["ci"],lp.ARM]))-qp["arm_cap"]))
    checks=dict(position_violation_rad=qv,velocity_violation_rad_s=vv,
                research_acceleration_violation_rad_s2=av,
                center_task_residual_mps=task_res,center_com_cap_violation_mps=com_v,
                center_arm_cap_violation_rad_s=arm_v)
    out.update(checks)
    tol=("postsolve_max_absolute_position_violation_rad",
         "postsolve_max_absolute_velocity_violation_rad_s",
         "postsolve_max_absolute_acceleration_violation_rad_s2",
         "postsolve_max_center_task_velocity_residual_mps",
         "postsolve_max_com_center_velocity_violation_mps",
         "postsolve_max_arm_center_velocity_change_violation_rad_s")
    if any(v>c[t] for v,t in zip(checks.values(),tol)):
        out["reason"]="POSTSOLVE_HARD_CONSTRAINT_VIOLATION"
        return out
    out.update(solved=True,reason="PASS_SAMPLED_KINEMATICS_ONLY",
               max_analytic_qddot_rad_s2=float(np.max(abs(ddq))),
               max_qdot_limit_ratio=float(np.max(abs(dq)/lim["vmax"])),
               max_absolute_perturbation_rad=float(np.max(abs(q-reference.q))),
               center_velocity_xyz_mps=(qp["J"]@dq[qp["ci"]]).tolist(),
               center_com_relative_velocity_xyz_mps=(qp["C"]@dq[qp["ci"]]).tolist(),
               coefficients_nonzero=int(np.count_nonzero(abs(coeff)>1e-8)),
               full14s_200hz_samples=int(q.shape[0]),
               motor_execution_authorized=False,native_physx_contact_certified=False,
               real_foot_contact_or_grf_certified=False,
               genuine_instep_impulse_or_ball_speed_certified=False,
               strict_physics_success=False)
    out["_q"]=q;out["_dq"]=dq;out["_ddq"]=ddq
    return out


def run(urdf:Path=task.URDF)->dict:
    c=frozen()
    prev.frozen() # inherited parent protocol also byte locked
    pc=lp.frozen(); tree,lim=lp.original_kinematics(urdf,pc)
    masses=task.mass_inertials(urdf,tree)
    base=chain.frozen()
    q0=(lim["lo"]+lim["hi"])/2
    attempts=[]
    totals=dict(attempts=0,qp_solved=0,qp_rejected=0,qp_infeasible=0,
                postsolve_rejected=0,geometry_proximity_warnings=0)
    for mult in c["multipliers"]:
        p=chain.generate(base,q0,mult)
        ci=round(c["contact_proxy_time_s"]/c["sample_period_s"])
        prob=lp.problem_at_pose(tree,masses,p.q[ci],pc)
        for target in c["relative_right_minus_left_ankle_origin_x_targets_mps"]:
            for width in c["compact_halfwidths_s"]:
                row=solve(p,prob,lim,c,target,width)
                totals["attempts"]+=1
                totals["qp_solved"]+=int(row["solved"])
                totals["qp_rejected"]+=int(not row["solved"])
                totals["qp_infeasible"]+=int("infeasible" in row["solver_status"])
                totals["postsolve_rejected"]+=int(row["reason"]=="POSTSOLVE_HARD_CONSTRAINT_VIOLATION")
                if row["solved"]:
                    win=prev.kinematic_window(tree,masses,row.pop("_q"),prev.frozen(),
                                              base["geometry_pairs"])
                    row.pop("_dq");row.pop("_ddq")
                    row["kinematic_window_samples"]=int(win["samples"])
                    row["minimum_four_pair_urdf_capsule_margin_m"]=win["minimum_urdf_capsule_margin_m"]
                    row["geometry_warning_frames"]=win["count_urdf_proximity_warning_frames"]
                    totals["geometry_proximity_warnings"]+=int(row["geometry_warning_frames"]>0)
                attempts.append(dict(multiplier=float(mult),**row))
    if totals["attempts"]!=c["required_grid_cells"]:
        raise RuntimeError("FROZEN_GRID_INCOMPLETE")
    return dict(version=c["version"],original29_dof=29,origin="SYNTHETIC_CPU_QP_NOT_PHYSICS",
                protocol_sha=FROZEN_SHA,urdf_sha256=c["urdf_sha256"],
                baseline_parent_sha=c["parent_sha"],
                summary=totals,attempts=attempts, new_physx_rollouts=0,
                human_professional_instep_learned=False,
                genuine_physical_shooting_success_rate=None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():
        raise FileExistsError("REFUSE_OVERWRITE_RESEARCH_EVIDENCE:"+str(a.output))
    result=run()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result["summary"],sort_keys=True))
    print("WROTE",a.output)


if __name__=="__main__":
    main()
