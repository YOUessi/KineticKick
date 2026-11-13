#!/usr/bin/env python3
"""Independent, equation-level review of v154 QP outputs.

Does not import the QP constructor, solver, or fixture generator. This checks
numerical constraints but cannot certify collision geometry or physical model.
"""
from __future__ import annotations
import numpy as np

def _arr(value,shape,what):
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.isfinite(a).all():
        raise ValueError("V154_INDEPENDENT_BAD_"+what)
    return a

def independent_check(problem,result,*,dynamics_tol=2e-4,
                      stance_tol=2e-4,limit_tol=2e-4):
    if result.get("solved") is not True or result.get("status")!="solved":
        return {"pass":False,"reason":"NOT_SOLVED"}
    if result.get("real_physics_episode") is not False or \
       result.get("human_elite_technique_learned") is not False or \
       result.get("legal_high_speed_instep_success") is not None:
        raise ValueError("V154_NUMERICAL_SOLUTION_SPOOFED_AS_PHYSICS")
    a=_arr(result["a35"],(35,),"ACC")
    tau=_arr(result["tau29"],(29,),"TORQUE")
    M=_arr(problem["M"],(35,35),"MASS")
    h=_arr(problem["h"],(35,),"BIAS")
    f_list=result["stance_forces3"]
    if len(f_list)!=len(problem["contacts"]):
        raise ValueError("V154_INDEPENDENT_CONTACT_COUNT")
    forces=np.concatenate([_arr(f,(3,),"FORCE") for f in f_list])
    J=np.concatenate([_arr(c["J"],(3,35),"J") for c in problem["contacts"]],axis=0)
    Jdv=np.concatenate([_arr(c["Jdotv"],(3,),"JDV") for c in problem["contacts"]])
    S=np.zeros(35);S[6:]=tau
    # Rebuild, rather than trusting recorded solver residuals.
    dyn=M.dot(a)+h-S-J.T.dot(forces)
    stance=J.dot(a)+Jdv
    dt=.005
    q=_arr(problem["q"],(29,),"JOINT_Q")
    dq=_arr(problem["dq"],(29,),"JOINT_DQ")
    speed=dq+dt*a[6:]
    nextq=q+dt*dq+0.5*dt*dt*a[6:]
    vmax=_arr(problem["dq_max"],(29,),"VMAX")
    lower=_arr(problem["q_lo"],(29,),"QLOWER")
    upper=_arr(problem["q_hi"],(29,),"QUPPER")
    amax=_arr(problem["acc_max"],(29,),"AMAX")
    tmax=_arr(problem["tau_max"],(29,),"TAUMAX")
    bounds=np.concatenate([
        speed-vmax,-vmax-speed,nextq-upper,lower-nextq,
        np.abs(a[6:])-amax,np.abs(tau)-tmax])
    friction_bad=[]
    for c,force in zip(problem["contacts"],f_list):
        f=_arr(force,(3,),"FORCE")
        mu=float(c["mu"])
        friction_bad.extend([-f[2],abs(f[0])-mu*f[2],abs(f[1])-mu*f[2]])
    clearance_bad=[]
    for g in problem["guards"]:
        gradient=_arr(g["distance_gradient35"],(35,),"GRAD")
        projected=g["signed_distance_m"]+dt*g["signed_rate_mps"]+0.5*dt*dt*(
            np.dot(gradient,a)+g["acceleration_bias_mps2"])
        clearance_bad.append(g["min_distance_m"]-projected)
    max_d=float(np.max(np.abs(dyn)))
    max_j=float(np.max(np.abs(stance)))
    max_l=float(max(0.,*bounds,*friction_bad,*clearance_bad))
    return {
        "pass":bool(max_d<=dynamics_tol and max_j<=stance_tol and max_l<=limit_tol),
        "max_dynamics_residual":max_d,
        "max_stance_acceleration_residual":max_j,
        "max_joint_contact_clearance_violation":max_l,
        "minimum_stance_normal":float(min(f[2] for f in f_list)),
        "max_qdot_ratio_after_step":float(np.max(np.abs(speed)/vmax)),
        "max_tau_ratio":float(np.max(np.abs(tau)/tmax)),
        "per_group_joint_acceleration_abs_sum":{
            "legs":float(np.sum(np.abs(a[6:][[0,1,3,4,6,7,9,10,13,14,17,18]]))),
            "waist":float(np.sum(np.abs(a[6:][[2,5,8]]))),
            "arms":float(np.sum(np.abs(a[6:][[11,12,15,16,19,20,21,22,23,24,25,26,27,28]]))),
        },
        "source_kind":problem["source_kind"],
        "true_grf_and_cop_certified":False,
        "full_physical_shot_certified":False,
    }
