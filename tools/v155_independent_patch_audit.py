#!/usr/bin/env python3
"""Independent v155 4-corner support audit, intentionally without solver imports.

A numerical feasibility checker, NOT ground-truth PhysX, G1 motor, or elite learning
certification. Reconstructs 6D net wrench directly from the corner force samples.
"""
from __future__ import annotations
import numpy as np

def vec(x,shape,tag):
    z=np.asarray(x,dtype=float)
    if z.shape!=tuple(shape) or not np.isfinite(z).all():
        raise ValueError("V155_AUDIT_BAD_"+tag)
    return z

def check(p,r,tol=5e-4):
    if not r.get("solved") or r.get("status")!="solved":
        return {"pass":False,"reason":"NOT_SOLVED"}
    if r.get("real_physics_episode") is not False or r.get("legal_high_speed_instep_success") is not None or r.get("human_elite_technique_learned") is not False or r.get("native_collision_manifold_certified") is not False:
        raise ValueError("V155_AUDIT_SPOOFED_PHYSICS_OR_HUMAN_TRUTH")
    a=vec(r["a35"],(35,),"A")
    tau=vec(r["tau29"],(29,),"TAU")
    f=vec(r["corner_forces3"],(len(p["contacts"]),4,3),"CORNER_FORCES")
    claimed=vec(r["wrenches6"],(len(p["contacts"]),6),"WRENCH")
    M=vec(p["M"],(35,35),"M")
    h=vec(p["h"],(35,),"H")
    dyn=M@a+h
    dyn[6:]-=tau
    stance_errors=[]
    corner_violation=[]
    cop_violation=[]
    wrench_errors=[]
    out_cops=[]
    total_ground_wrench=[]
    for i,c in enumerate(p["contacts"]):
        xy=vec(c["corners_xy"],(4,2),"CORNERS")
        J=vec(c["J6"],(6,35),"J6")
        JD=vec(c["Jdotv6"],(6,),"JDOTV6")
        stance_errors.extend((J@a+JD).tolist())
        ff=f[i]
        total=ff.sum(axis=0)
        m=np.sum(np.cross(np.column_stack((xy,np.zeros(4))),ff),axis=0)
        wrench=np.concatenate([total,m])
        wrench_errors.extend((wrench-claimed[i]).tolist())
        total_ground_wrench.append(wrench.tolist())
        dyn-=J.T@wrench
        friction=float(c["mu"])
        for force in ff:
            corner_violation.extend([-force[2],abs(force[0])-friction*force[2],abs(force[1])-friction*force[2]])
        if total[2]>1e-10:
            # CoP equals the normal-load weighted position, = (-M_y/F_z, M_x/F_z).
            cx=-m[1]/total[2]
            cy=m[0]/total[2]
            out_cops.append([float(cx),float(cy)])
            cop_violation.extend([abs(cx)-float(np.max(abs(xy[:,0]))),
                                  abs(cy)-float(np.max(abs(xy[:,1])))])
        else:
            out_cops.append(None)
            corner_violation.append(tol+1)
    q=vec(p["q"],(29,),"Q")
    dq=vec(p["dq"],(29,),"DQ")
    qmax=vec(p["q_hi"],(29,),"QHI")
    qmin=vec(p["q_lo"],(29,),"QLO")
    vmax=vec(p["dq_max"],(29,),"VMAX")
    amax=vec(p["acc_max"],(29,),"AMAX")
    taumax=vec(p["tau_max"],(29,),"TAUMAX")
    qnew=q+.005*dq+.5*.005**2*a[6:]
    dnew=dq+.005*a[6:]
    limits=np.concatenate([qnew-qmax,qmin-qnew,abs(dnew)-vmax,
                           abs(a[6:])-amax,abs(tau)-taumax])
    gaps=[]
    for g in p["guards"]:
        gradient=vec(g["distance_gradient35"],(35,),"GAP")
        predicted=(g["signed_distance_m"]+.005*g["signed_rate_mps"]+
                  .5*.005**2*(gradient@a+g["acceleration_bias_mps2"]))
        gaps.append(g["min_distance_m"]-predicted)
    max_dyn=float(max(abs(dyn)))
    max_stance=float(max(abs(np.asarray(stance_errors))))
    max_wrench=float(max(abs(np.asarray(wrench_errors))))
    max_ineq=float(max(0.,*limits,*corner_violation,*cop_violation,*gaps))
    return {"pass":bool(max(max_dyn,max_stance,max_wrench,max_ineq)<=tol),
            "max_dynamics_residual":max_dyn,
            "max_6d_stance_residual":max_stance,
            "max_wrench_reconstruction_residual":max_wrench,
            "max_contact_joint_gap_violation":max_ineq,
            "max_all29_qdot_ratio":float(max(abs(dnew)/vmax)),
            "max_all29_tau_ratio":float(max(abs(tau)/taumax)),
            "min_corner_ground_normal_N":float(f[:,:,2].min()),
            "cop_xy_m":out_cops,
            "ground_wrenches6_fixture_only":total_ground_wrench,
            "actual_ground_grf_cop_verified":False,
            "native_contact_pair_known":False,
            "actual_physics_validated":False}
