#!/usr/bin/env python3
"""v129: read-only, SHA-qualified exact finite URDF-cylinder separation certificates.

Finite cylinders are *contained* in v127 bounding capsules. A near-zero
capsule lower bound does not imply the two finite cylinders touch.

For each unit direction n, the directional support gap
    min_(b in B) n.b - max_(a in A) n.a
is a *provable lower bound* on the Euclidean distance of two disjoint convex
cylinders. A positive gap certifies disjointness of the original URDF cylinders
at the observed posture, regardless of optimization convergence. The maximal
directional gap equals their true separation for disjoint compact convex sets.

Negative gaps, even from a solver, are NEVER evidence of direct PhysX contact.
Primal feasible point pairs give independent *upper bounds* at selected steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import v127_collider_proximity as geometry
import v128_v54_event_geometry as historical

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v129_frozen_exact_cylinder_separation.json"
FROZEN_SHA = "d614cb0f938a1968b461fb5f7683589509d5623b"


@dataclass(frozen=True)
class Cylinder:
    center: np.ndarray
    axis: np.ndarray
    radius: float
    half_length: float

    def __post_init__(self):
        c=np.asarray(self.center, dtype=float)
        a=np.asarray(self.axis, dtype=float)
        if (c.shape != (3,) or a.shape != (3,) or
            not np.isfinite(c).all() or not np.isfinite(a).all() or
            abs(np.linalg.norm(a)-1)>1e-8 or
            not (np.isfinite(self.radius) and self.radius>0) or
            not (np.isfinite(self.half_length) and self.half_length>0)):
            raise ValueError("INVALID_FINITE_CYLINDER")


def frozen():
    b=subprocess.check_output(
        ["git","-C",str(ROOT),"show",FROZEN_SHA+":configs/v129_frozen_exact_cylinder_separation.json"])
    if CONFIG.read_bytes()!=b:
        raise ValueError("FROZEN_PROTOCOL_DRIFT")
    cfg=json.loads(b)
    if (cfg["version"]!="v0.1.129" or cfg["native_physx_pair_certified"] or
        not cfg["no_gpu_or_actuation_during_gpu_mutex"] or
        not cfg["no_extrapolation_to_unsampled_sensor_event"] or
        not cfg["no_existing_evidence_overwrite"]):
        raise ValueError("UNSAFE_V129_PROTOCOL")
    hash128=hashlib.sha256((ROOT/"configs/v128_frozen_v54_collision_event.json").read_bytes()).hexdigest()
    if hash128!=cfg["frozen_v128_config_sha256"]:
        raise ValueError("PARENT_FROZEN_INPUT_DRIFT")
    return cfg


def normalized(v):
    v=np.asarray(v,dtype=float)
    d=float(np.linalg.norm(v))
    return v/d if d>1e-12 else None


def support_offset(c: Cylinder, direction):
    """Cylinder support minus center dot direction; valid for any n in R^3."""
    n=np.asarray(direction,dtype=float)
    ax=float(np.dot(n,c.axis))
    perp=n-ax*c.axis
    return c.half_length*abs(ax)+c.radius*float(np.linalg.norm(perp))


def projection_gap(a: Cylinder,b: Cylinder,direction):
    """Rigorously guaranteed directional *lower bound* for all solid-cylinder points."""
    n=np.asarray(direction,dtype=float)
    if n.shape!=(3,) or not np.isfinite(n).all():
        raise ValueError("INVALID_SEPARATING_DIRECTION")
    norm=float(np.linalg.norm(n))
    if norm<=1e-12:
        return 0.0
    n=n/norm
    return (float(np.dot(n,b.center-a.center))
            -support_offset(a,n)-support_offset(b,n))


def candidates(a,b):
    d=b.center-a.center
    out=[d, -d, *np.eye(3), *(-np.eye(3)), a.axis, -a.axis, b.axis, -b.axis,
         np.cross(a.axis,b.axis), -np.cross(a.axis,b.axis)]
    for axis in (a.axis,b.axis):
        projected=d-np.dot(d,axis)*axis
        out.extend((projected,-projected))
    # Closest segment-centerline witness can provide a useful narrow-gap direction.
    # Deterministic finite grid is not a contact detector or tuning parameter.
    for ta in (-a.half_length,0,a.half_length):
        for tb in (-b.half_length,0,b.half_length):
            w=b.center+tb*b.axis-a.center-ta*a.axis
            out.append(w)
    result=[]
    for v in out:
        n=normalized(v)
        if n is not None and not any(np.linalg.norm(n-prev)<1e-8 for prev in result):
            result.append(n)
    return result


def separating_certificate(a,b, *, optimize=True, extra_seeds=2):
    dirs=candidates(a,b)
    scored=[(projection_gap(a,b,n),n) for n in dirs]
    best,maxdir=max(scored,key=lambda x:x[0])
    nfev=0
    # SLSQP maximizes a convex-body dual over the unit ball. The output is
    # ALWAYS re-evaluated as a valid raw support direction before certification.
    if optimize:
        starts=[maxdir]+[n for _,n in sorted(scored,key=lambda x:-x[0])[1:1+extra_seeds]]
        for init in starts:
            def fun(x):
                mag=np.linalg.norm(x)
                if mag<1e-12:
                    return 0.0
                return -projection_gap(a,b,x)
            res=minimize(fun,init,method="SLSQP",
                         bounds=[(-1,1)]*3,
                         constraints=[{"type":"ineq","fun":lambda x: 1-np.dot(x,x)}],
                         options={"ftol":1e-12,"maxiter":85})
            nfev+=res.nfev
            for v in (res.x,):
                unit=normalized(v)
                if unit is not None:
                    g=projection_gap(a,b,unit)
                    if g>best:
                        best,maxdir=g,unit
    # A positive candidate guarantees *solid* cylinder separation even if
    # the optimizer failed. Zero or negative cannot prove a collision.
    return {"lower_bound_m":max(0.0,float(best)),
            "signed_max_candidate_m":float(best),
            "direction":[float(x) for x in maxdir],
            "strictly_separated":bool(best>1e-6),
            "native_physx_contact_pair":None,
            "optimizer_function_evals":int(nfev)}


def cylinder_frame(a_center,a_axis,r,h):
    a=np.asarray(a_axis,dtype=float)
    e=np.cross(a,[1.,0.,0.])
    if np.linalg.norm(e)<1e-8:
        e=np.cross(a,[0.,1.,0.])
    e=e/np.linalg.norm(e)
    f=np.cross(a,e)
    return np.asarray(a_center,dtype=float),np.column_stack((e,f,a))


def primal_upper_bound(a,b):
    """Feasible points on two finite solids give a numerical upper bound.

    Checks radial and axial feasibility, independently of the dual gap solver.
    Never used as evidence of direct contact ownership.
    """
    ca,Ua=cylinder_frame(a.center,a.axis,a.radius,a.half_length)
    cb,Ub=cylinder_frame(b.center,b.axis,b.radius,b.half_length)
    def xyz(x,ctr,U):
        return ctr+U@x
    def objective(v):
        da=xyz(v[:3],ca,Ua)-xyz(v[3:],cb,Ub)
        return float(np.dot(da,da))
    cons=[
        {"type":"ineq","fun":lambda v:a.radius*a.radius-v[0]*v[0]-v[1]*v[1]},
        {"type":"ineq","fun":lambda v:b.radius*b.radius-v[3]*v[3]-v[4]*v[4]},
    ]
    bounds=[(-a.radius,a.radius)]*2+[(-a.half_length,a.half_length)]+[
        (-b.radius,b.radius)]*2+[(-b.half_length,b.half_length)]
    best=float("inf")
    for seed in (np.zeros(6), np.array([0,0,a.half_length,0,0,-b.half_length],dtype=float)):
        opt=minimize(objective,seed,method="SLSQP",bounds=bounds,
                     constraints=cons,options={"ftol":1e-13,"maxiter":250})
        v=np.asarray(opt.x,dtype=float).copy()
        if not np.isfinite(v).all():
            continue
        # SLSQP can stop a few nm outside a circular constraint; such points
        # are NOT rigorous primal feasible witnesses. Project exactly back to
        # each disk and closed axial segment before computing the upper bound.
        for j,r in ((0,a.radius),(3,b.radius)):
            rad=np.linalg.norm(v[j:j+2])
            if rad>r:
                v[j:j+2]*=r/rad
        v[2]=np.clip(v[2],-a.half_length,a.half_length)
        v[5]=np.clip(v[5],-b.half_length,b.half_length)
        best=min(best,np.sqrt(objective(v)))
    if not np.isfinite(best):
        raise ValueError("NO_FEASIBLE_PRIMAL_WITNESS")
    return float(best)


def cylinders_from_q(tree, qs, first, second):
    poses=geometry.fk_targets(tree,qs,{first,second})
    out=[]
    for name in (first,second):
        p0,p1,r=geometry.cylinder_axis_endpoints(tree,poses,name)
        center=(p0+p1)*.5
        v=p1-p0
        length=np.linalg.norm(v,axis=1)
        out.append([Cylinder(center[i],v[i]/length[i],r,length[i]/2)
                    for i in range(len(qs))])
    return out


def evaluate(cfg):
    parent=historical.frozen_protocol()
    if parent["version"]!="v0.1.128":
        raise ValueError("BAD_PARENT")
    # This reexecutes strict v52/v54 instrument OFF/ON physical-field and
    # source SHA audits before any v129 geometric computation.
    parent_audit=historical.run(parent)
    if not parent_audit["original_read_only_instrumentation_all_common_physics_fields_identical"]:
        raise ValueError("READ_ONLY_PARITY_NOT_CERTIFIED")
    root=Path(parent["data_root"])
    tree=geometry.parse_tree(Path(parent["original_G1_urdf"]))
    first,second=cfg["primary_pair"]
    results={}
    report_around={}
    all_frames=0
    exact_checks={}
    for spec in parent["original_v54_cases"]:
        name=spec["case"]
        p=root/f"v054_{name}_contact_diagnostics.json"
        historical.require_sha(p,parent["input_sha256"][p.name])
        raw=json.loads(p.read_text())
        rows=historical.validate_rows(raw,spec["expected_trace_rows"],parent["dt_s"])
        t,q=rows["t"],rows["q"]
        A,B=cylinders_from_q(tree,q,first,second)
        capsule=geometry.clearance_for_episode(tree,q,[(first,second)])[first+"__"+second]
        if len(capsule)!=len(t):
            raise ValueError("CAPSULE_LENGTH_MISMATCH")
        certificate=[]
        for i,(a,b) in enumerate(zip(A,B)):
            # always run deterministic search, optimize every problematic capsule
            # frame and every frame in the final 100ms of the named v54 event.
            near=bool(capsule[i] <= .02 or
                 (cfg["failure_event_s"]-.1-1e-7<=t[i]<=cfg["failure_event_s"]))
            certificate.append(separating_certificate(a,b,optimize=near,extra_seeds=2))
        plane=np.array([x["lower_bound_m"] for x in certificate])
        signed=np.array([x["signed_max_candidate_m"] for x in certificate])
        # Candidate directions need not be optimal. For frames with a broad
        # geometric margin, a valid existing capsule *lower bound* can be
        # stronger than an unoptimized candidate direction. Combine two
        # mathematically valid bounds: maximum cannot overstate real distance.
        dual=np.maximum(plane,np.maximum(capsule,0.0))
        separated=dual>cfg["strict_positive_plane_certificate_epsilon_m"]
        pure_plane=plane>cfg["strict_positive_plane_certificate_epsilon_m"]
        times=(5.125,5.13,5.135,5.14)
        checks=[]
        for stamp in times:
            idx=np.flatnonzero(abs(t-stamp)<1e-4)
            if len(idx)!=1:
                raise ValueError("FROZEN_EVENT_TIME_MISSING")
            j=int(idx[0])
            ub=primal_upper_bound(A[j],B[j])
            lb=float(dual[j])
            if lb>ub+1e-9:
                raise ValueError("BROKEN_DUAL_PRIMAL_ORDER")
            checks.append({"s":float(t[j]),"capsule_lower_m":float(capsule[j]),
                           "finite_cylinder_certified_lower_m":lb,
                           "raw_support_plane_lower_m":float(plane[j]),
                           "finite_cylinder_feasible_upper_m":ub,
                           "optimization_duality_gap_m":max(0.,ub-lb),
                           "finite_URDF_cylinders_disjoint":bool(separated[j]),
                           "not_a_native_PhysX_contact":True})
        exact_checks[name]=checks
        all_frames+=len(t)
        results[name]={
            "total_real_physics_states":len(t),
            "source_data_sha256":parent["input_sha256"][p.name],
            "finite_cylinder_disjoint_certified_frames":int(np.sum(separated)),
            "support_plane_disjoint_certified_frames":int(np.sum(pure_plane)),
            "capsule_only_certified_frames":int(np.sum(separated & ~pure_plane)),
            "support_plane_stronger_than_capsule_frames":int(np.sum(plane > np.maximum(capsule,0)+1e-7)),
            "frames_without_disjoint_certificate":int(np.sum(~separated)),
            "minimum_certificate_lower_bound_m":float(dual.min()),
            "min_certificate_at_time_s":float(t[int(np.argmin(dual))]),
            "minimum_capsule_lower_bound_m":float(capsule.min()),
            "min_capsule_at_time_s":float(t[int(np.argmin(capsule))]),
            "greatest_certified_separation_gain_vs_capsule_m":float((dual-capsule).max()),
            "first_20mm_capsule_alert_s":float(t[np.flatnonzero(capsule<=.02)[0]]) if np.any(capsule<=.02) else None,
            "first_20mm_finite_lower_alert_s":float(t[np.flatnonzero(dual<=.02)[0]]) if np.any(dual<=.02) else None,
            "no_native_contact_pair_truth":True,
        }
        win=(t>=cfg["failure_event_s"]-.25-1e-7)&(t<cfg["failure_event_s"]-1e-4)
        report_around[name]={
            "time_s":t[win].tolist(),
            "capsule_lower_bound_m":capsule[win].tolist(),
            "exact_finite_cylinder_lower_bound_m":dual[win].tolist(),
            "raw_support_plane_lower_bound_m":plane[win].tolist(),
            "dual_signed_candidate_m":signed[win].tolist(),
            "nonfoot_force_N":rows["nonfoot"][win].tolist(),
        }
    if all_frames!=6628:
        raise ValueError("ORIGINAL_FRAME_COUNT_DRIFT")
    return {
        "version":cfg["version"],"frozen_protocol_commit":FROZEN_SHA,
        "physx_source_sha256_verified":True,"original_OFF_ON_shared_physical_fields_identical":True,
        "source_G1_dof":29,"source_physics_hz":200,"original_real_physics_states":all_frames,
        "new_physx_rollouts":0,"new_controller_actuations":0,
        "certificates_for_finite_URDF_cylinders_only":True,
        "native_physx_pair_certified":False,
        "no_sensor_event_q_extrapolation":True,
        "all_three_cases":results,
        "four_declared_near_event_primal_dual_crosschecks":exact_checks,
        "last_250ms_real_physics_curves":report_around,
        "prior_fail_contact_is_still_unknown":True,
        "professional_human_kinetics_learned":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    cfg=frozen()
    result=evaluate(cfg)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({"file":str(args.output),"steps":result["original_real_physics_states"],
       "cases":result["all_three_cases"],
       "checks":result["four_declared_near_event_primal_dual_crosschecks"]},indent=2))


if __name__=="__main__":
    main()
