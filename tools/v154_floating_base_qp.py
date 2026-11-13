#!/usr/bin/env python3
"""A 35D (6 floating+29 actuated) full-body acceleration QP feasibility gate.

CPU numerics ONLY. Inputs for real control MUST come from a calibrated same-tick
floating-base dynamics engine and independent ground-contact/collider witnesses.
Fixture matrices are not G1 mass, inertia, GRF, human motion or PhysX evidence.
This module never steps physics, changes controllers, or commands a robot.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import scipy.sparse as sp
import osqp

from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"configs/v154_frozen_wholebody_actuation_qp.json"
FROZEN_SHA="21f42e974196621137a209aa7c979956ac3068a8"
N=35
NA=29
DT=0.005

def frozen():
    raw=subprocess.check_output(["git","-C",str(ROOT),"show",FROZEN_SHA+
                                 ":configs/v154_frozen_wholebody_actuation_qp.json"])
    if raw!=PROTOCOL.read_bytes():
        raise ValueError("V154_FROZEN_PROTOCOL_CHANGED")
    cfg=json.loads(raw)
    if cfg["parent_sha"]!="c4777d50265f90bd052b449f5fe6497c8b0f35c5" or \
       cfg["physical_q_dof"]!=29 or cfg["generalized_velocity_dim"]!=35 or \
       cfg["control_dt_s"]!=DT or cfg["total_positive_fixture_cases"]!=48:
        raise ValueError("V154_PROTOCOL_ASSUMPTION_DRIFT")
    return cfg

def array(value,shape,tag):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=tuple(shape) or not np.isfinite(a).all():
        raise ValueError("V154_BAD_"+tag)
    return a

def _positive(value,tag):
    if isinstance(value,bool) or not isinstance(value,(float,int,np.integer,np.floating)) or \
       not math.isfinite(float(value)) or float(value)<=0:
        raise ValueError("V154_NOT_POSITIVE_"+tag)
    return float(value)

def validate_problem(p):
    """Refuse real telemetry lacking *ground-only*, signed and same-tick contacts."""
    frozen()
    if p.get("joint_names")!=list(G1_JOINT_ORDER):
        raise ValueError("V154_ORIGINAL_29_ORDER_UNKNOWN")
    if p.get("source_kind") not in ("synthetic","runtime"):
        raise ValueError("V154_SOURCE_KIND_UNKNOWN")
    if p["source_kind"]=="runtime":
        required=("measured_signed_ground_normal","calibrated_mass_and_bias",
                  "calibrated_stance_jacobians","real_collider_gap_gradients",
                  "exact_native_step_id","human_training_qualified")
        # Human label is needed to call this a learned/qualified football controller;
        # numerical dry-run with real physics matrices can leave it false.
        for flag in required[:-1]:
            if p.get("provenance",{}).get(flag) is not True:
                raise ValueError("V154_MISSING_RUNTIME_PROVENANCE_"+flag)
        if not p.get("provenance",{}).get("source_hashes"):
            raise ValueError("V154_MISSING_RUNTIME_SOURCE_HASHES")
    if p.get("require_human_training") and p.get("provenance",{}).get("human_training_qualified") is not True:
        raise ValueError("V154_NO_QUALIFIED_HUMAN_INSTEP_TRAINING")
    M=array(p["M"],(N,N),"M35")
    h=array(p["h"],(N,),"H35")
    if np.max(np.abs(M-M.T))>1e-9 or float(np.linalg.eigvalsh(M).min())<=1e-9:
        raise ValueError("V154_NONSPD_FLOATING_BASE_MASS")
    for field in ("q","dq","q_lo","q_hi","dq_max","tau_max","acc_max","a_ref_joint"):
        array(p[field],(NA,),field.upper())
    q,lo,hi=array(p["q"],(NA,),"q"),array(p["q_lo"],(NA,),"lo"),array(p["q_hi"],(NA,),"hi")
    if np.any(lo>=hi) or np.any(q<lo) or np.any(q>hi):
        raise ValueError("V154_INVALID_JOINT_LIMITS")
    for key in ("dq_max","tau_max","acc_max"):
        if np.min(array(p[key],(NA,),key))<=0:
            raise ValueError("V154_NONPOSITIVE_"+key)
    dq=array(p["dq"],(NA,),"dq")
    if np.any(np.abs(dq)>array(p["dq_max"],(NA,),"dq_max")+1e-9):
        raise ValueError("V154_ALREADY_QDOT_UNSAFE")
    contacts=p.get("contacts")
    if not isinstance(contacts,list) or not 1<=len(contacts)<=2:
        raise ValueError("V154_GROUND_CONTACT_MISSING")
    seen=set()
    for c in contacts:
        if c.get("body") not in ("left","right") or c["body"] in seen:
            raise ValueError("V154_INVALID_STANCE_BODY")
        seen.add(c["body"])
        array(c.get("J"),(3,N),"STANCE_J")
        array(c.get("Jdotv"),(3,),"STANCE_JDOTV")
        _positive(c.get("mu"),"MU")
        if c.get("ground_only_signed_normal") is not True:
            raise ValueError("V154_FOOT_NET_CONTACT_NOT_GROUND_GRF")
        if c.get("normal_axis")!="world_z":
            raise ValueError("V154_UNKNOWN_CONTACT_FRAME")
    guards=p.get("guards")
    if not isinstance(guards,list) or not guards:
        raise ValueError("V154_UNCALIBRATED_NONFOOT_COLLISION_GEOMETRY")
    for g in guards:
        if g.get("pair") not in ("left_wrist_to_left_hip","right_wrist_to_right_hip",
                                "body_to_environment"):
            raise ValueError("V154_INVALID_COLLIDER_PAIR")
        array(g.get("distance_gradient35"),(N,),"GAP_JACOBIAN")
        for k in ("signed_distance_m","min_distance_m","signed_rate_mps",
                  "acceleration_bias_mps2"):
            v=g.get(k)
            if type(v) not in (float,int) or not math.isfinite(v):
                raise ValueError("V154_UNCALIBRATED_COLLISION_"+k)
        if g["min_distance_m"]<0:
            raise ValueError("V154_INVALID_DISTANCE_MARGIN")
    return M,h

def _matrow(length,sparse_values):
    out=np.zeros(length,dtype=np.float64)
    for index,value in sparse_values:
        out[index]=value
    return out

def solve(p):
    """Solve one deterministic convex QP. Audit required before acceptance."""
    M,h=validate_problem(p)
    cfg=frozen()
    contacts=p["contacts"]
    nc=len(contacts)
    dim=N+NA+3*nc
    A=[];lower=[];upper=[]
    def add(row,l,u):
        A.append(row)
        lower.append(l)
        upper.append(u)
    J=np.vstack([array(c["J"],(3,N),"J") for c in contacts])
    Jdotv=np.concatenate([array(c["Jdotv"],(3,),"Jdotv") for c in contacts])
    # M a + h = S^T tau + J^T f
    for i in range(N):
        row=np.zeros(dim)
        row[:N]=M[i]
        if i>=6:row[N+i-6]=-1
        row[N+NA:]=-J[:,i]
        add(row,-h[i],-h[i])
    # Closed, non-sliding point foot contacts, no invented foot moments or CoP.
    for i in range(3*nc):
        row=np.zeros(dim);row[:N]=J[i]
        add(row,-Jdotv[i],-Jdotv[i])
    dt=DT
    q=array(p["q"],(NA,),"q")
    dq=array(p["dq"],(NA,),"dq")
    qlo=array(p["q_lo"],(NA,),"qlo");qhi=array(p["q_hi"],(NA,),"qhi")
    vmax=array(p["dq_max"],(NA,),"vmax")
    amax=array(p["acc_max"],(NA,),"amax")
    tau_max=array(p["tau_max"],(NA,),"tau")
    for j in range(NA):
        acc_low=max(-amax[j],(-vmax[j]-dq[j])/dt,
                    2*(qlo[j]-q[j]-dt*dq[j])/(dt*dt))
        acc_high=min(amax[j],(vmax[j]-dq[j])/dt,
                     2*(qhi[j]-q[j]-dt*dq[j])/(dt*dt))
        add(_matrow(dim,[(6+j,1.0)]),acc_low,acc_high)
        add(_matrow(dim,[(N+j,1.0)]),-tau_max[j],tau_max[j])
    for k,c in enumerate(contacts):
        mu=float(c["mu"])
        base=N+NA+3*k
        add(_matrow(dim,[(base+2,1)]),0,np.inf)
        for axis in (0,1):
            for sign in (-1,1):
                add(_matrow(dim,[(base+axis,sign),(base+2,-mu)]),-np.inf,0)
    for g in p["guards"]:
        grad=array(g["distance_gradient35"],(N,),"grad")
        lower_acc=2*(g["min_distance_m"]-g["signed_distance_m"]
                     -dt*g["signed_rate_mps"])/(dt*dt)-g["acceleration_bias_mps2"]
        row=np.zeros(dim);row[:N]=grad
        add(row,lower_acc,np.inf)
    # Meaningful full-body (29 joints) desired acceleration, NOT 1mrad perturbation.
    weights=cfg["objective_weights"]
    diag=np.concatenate([
        np.full(6,weights["base_acceleration"]),
        np.full(NA,weights["joint_acceleration_tracking"]),
        np.full(NA,weights["torque_regularizer"]),
        np.full(3*nc,weights["foot_force_regularizer"])])
    desired=np.zeros(dim)
    desired[6:N]=array(p["a_ref_joint"],(NA,),"ref")
    P=sp.diags(2*diag,format="csc")
    qvec=-2*diag*desired
    constraint=sp.csc_matrix(np.stack(A))
    opt=cfg["solver_controls"]
    solver=osqp.OSQP()
    solver.setup(P=P,q=qvec,A=constraint,l=np.asarray(lower),u=np.asarray(upper),
                 verbose=False,polish=True,eps_abs=opt["eps_abs"],eps_rel=opt["eps_rel"],
                 max_iter=opt["max_iter"],adaptive_rho=False)
    r=solver.solve()
    ok=r.info.status=="solved" and r.x is not None and np.isfinite(r.x).all()
    out={
        "source_kind":p["source_kind"],
        "solver":cfg["solver"],"status":r.info.status,
        "solved":bool(ok),"phase":p["phase"],"seed":p["seed"],
        "support":[x["body"] for x in contacts],
        "g1_original_joint_names":list(G1_JOINT_ORDER),
        "frozen_sha":FROZEN_SHA,
        "input_declares_runtime_source":p["source_kind"]=="runtime",
        "physical_model_independently_certified":False,
        "real_physics_episode":False,
        "legal_high_speed_instep_success":None,
        "human_elite_technique_learned":False,
    }
    if ok:
        x=np.asarray(r.x)
        out["a35"]=x[:N].tolist()
        out["tau29"]=x[N:N+NA].tolist()
        out["stance_forces3"]=[x[N+NA+3*k:N+NA+3*k+3].tolist() for k in range(nc)]
        out["objective"]=float(r.info.obj_val)
    return out
