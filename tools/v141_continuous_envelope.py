#!/usr/bin/env python3
"""v141: all-time *numerical* Bernstein envelope for original29 CPU QP motions.

No simulation, contact, learned human motion, force, or motor permission.  A
Bernstein convex hull is conservative for the polynomial it represents; here
its construction is floating-point, NOT formally interval-verified arithmetic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from numpy.polynomial import Polynomial as Poly

import v135_chain29_offline as chain
import v136_g1_taskspace_mass_com as task
import v137_hard_support_relative_lp as lp
import v139_joint_window_qp as qp
import v140_nonlinear_dense_reaudit as dense

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/v141_frozen_continuous_polynomial_envelope.json"
FROZEN_SHA="d0df1a047b058fc20d1c2696300f11ab6c370c5c"
GRID={(float(m),float(t),float(w)) for m in [0,1,2,4,-1] for t in [0,1,3] for w in [.4,.8,1.2]}

def frozen():
    if CONFIG.resolve()!=(ROOT/"configs/v141_frozen_continuous_polynomial_envelope.json").resolve():
        raise ValueError("V141_CONFIG_PATH_DRIFT")
    raw=subprocess.check_output(["git","-C",str(ROOT),"show",FROZEN_SHA+":configs/v141_frozen_continuous_polynomial_envelope.json"])
    if CONFIG.read_bytes()!=raw:
        raise ValueError("V141_CONFIG_BYTE_DRIFT")
    c=json.loads(raw)
    required={
        "version":"v0.1.141",
        "parent_sha":"df46cf40857f4cde177aa14cae6e93092dc3adff",
        "parent_protocol_sha":"d1334ec692a81ebb1d146e121940ff9c6a72542b",
        "original_g1_urdf_sha256":"34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761",
        "original_dof":29, "horizon_s":14.0, "frozen_grid_multipliers":[0,1,2,4,-1],
        "frozen_grid_target_mps":[0,1,3], "frozen_grid_halfwidth_s":[.4,.8,1.2],
        "frozen_grid_cells":45,"baseline_phase_durations_s":[1.,.8,.6,.5,.1,1.,10.],
        "correction_powers":[1,3,5,7],"bernstein_max_depth":20,
        "bernstein_max_nodes_per_polynomial":100000, "numerical_guard":1e-8,
        "sample_crosscheck_absolute_error":2e-7,
        "position_bound_tolerance_rad":.0003,"velocity_bound_tolerance_rad_s":.0003,
        "research_acceleration_bound_rad_s2":40.,
        "research_acceleration_bound_tolerance_rad_s2":.005,
        "new_physx_rollouts_when_gpu_owned":0,"physical_success_rate":None,
        "disk_floor_gib":20,
    }
    if any(c.get(k)!=v for k,v in required.items()):
        raise ValueError("V141_FROZEN_VALUE_DRIFT")
    if c["envelope"]!="FLOAT64_LOCAL_POWER_TO_BERNSTEIN_ADAPTIVE_CONSERVATIVE_NUMERIC_NOT_FORMAL_PROOF":
        raise ValueError("V141_FALSE_FORMAL_CERTIFICATE")
    if any(c[k] is not True for k in ("cannot_upgrade_parent_rejection",
            "cannot_upgrade_v140_dense_rejection","must_report_unresolved_without_ascribing_infeasibility",
            "must_compare_local_polynomial_against_independent_v140_analytic_samples")):
        raise ValueError("V141_MISSING_NEGATIVE_CONTROL")
    if any(c[k] is not False for k in ("motor_execution_authorized",
            "native_physx_contact_pair_certified","high_skill_human_instep_force_truth")):
        raise ValueError("V141_FALSE_PHYSICAL_CLAIM")
    if c["strict_physical_gates"]!={"instep_impulse_fraction_min":.95,"legal_goal":True,
          "minimum_ball_speed_mps":3.5,"all_original_29_qdot_legal":True,
          "no_nonfoot_collision":True,"full_14s_post_kick_recovery":True}:
        raise ValueError("V141_WEAK_PHYSICS_GATE")
    if subprocess.check_output(["git","-C",str(ROOT),"rev-parse",FROZEN_SHA+"^"],text=True).strip()!=c["parent_sha"]:
        raise ValueError("V141_NOT_PRECOMMITTED_FROM_PARENT")
    return c


def segments(width:float, center=2.95):
    """Complete disjoint piecewise polynomial partition including t=14 endpoint."""
    if width not in (.4,.8,1.2):
        raise ValueError("V141_UNREGISTERED_HALF_WIDTH")
    phase=np.r_[0.,np.cumsum(chain.frozen()["phase_durations_s"])]
    cuts=sorted({round(float(x),12) for x in [*phase,center-width,center+width]})
    if cuts[0]!=0 or cuts[-1]!=14:
        raise ValueError("V141_BAD_POLYNOMIAL_PARTITION")
    return [(cuts[i],cuts[i+1]) for i in range(len(cuts)-1)]


def polynomial_piece(a:float,b:float,mult:float,width:float,coeff:np.ndarray,
                     q0:np.ndarray,base:dict):
    """q(z), dq/dt(z), d2q/dt2(z), z=(t-a)/(b-a) on one phase+support piece."""
    coeff=np.asarray(coeff,float)
    q0=np.asarray(q0,float)
    if (coeff.shape!=(4,29) or q0.shape!=(29,) or
        not np.isfinite(coeff).all() or not np.isfinite(q0).all() or
        not np.isfinite([a,b,mult,width]).all() or a<0 or b>14 or b<=a):
        raise ValueError("V141_BAD_PIECE_INPUT")
    if mult not in [0,1,2,4,-1] or width not in [.4,.8,1.2]:
        raise ValueError("V141_UNREGISTERED_PIECE")
    durations=np.asarray(base["phase_durations_s"],float)
    edges=np.r_[0.,np.cumsum(durations)]
    mid=(a+b)/2
    k=int(np.searchsorted(edges,mid,side="right")-1)
    if k<0 or k>=7 or a<edges[k]-1e-10 or b>edges[k+1]+1e-10:
        raise ValueError("V141_PIECE_STRADDLES_PHASE")
    w=chain.weight_matrix()
    amps=np.asarray(base["latent_amplitudes_rad"],float)*mult
    initial=q0+(chain.KNOTS[k]*amps)@w.T
    delta=((chain.KNOTS[k+1]-chain.KNOTS[k])*amps)@w.T
    z=Poly([(a-edges[k])/durations[k],(b-a)/durations[k]])
    smooth=6*z**5-15*z**4+10*z**3
    active=(2.95-width<=mid<=2.95+width)
    corr=[]
    if active:
        s=Poly([(a-2.95)/width,(b-a)/width])
        g=(1-s*s)**4
        corr=[width*s**n*g for n in [1,3,5,7]]
    out=[]
    scale=1/(b-a)
    for j in range(29):
        p=Poly([float(initial[j])])+float(delta[j])*smooth
        for z,c in zip(corr,coeff[:,j]):
            if c!=0:p=p+float(c)*z
        out.append((p,p.deriv()*scale,p.deriv(2)*scale*scale))
    return out


def power_to_bernstein(p:Poly)->np.ndarray:
    if not isinstance(p,Poly) or not np.isfinite(p.coef).all() or len(p.coef)>16:
        raise ValueError("V141_INVALID_DEGREE_OR_NONFINITE_POLYNOMIAL")
    c=p.coef.astype(float)
    n=len(c)-1
    b=np.array([sum(c[k]*(math.comb(i,k)/math.comb(n,k))
                  for k in range(i+1)) for i in range(n+1)],float)
    if not np.isfinite(b).all():
        raise ValueError("V141_NONFINITE_BERNSTEIN")
    return b


def bisect_bernstein(b:np.ndarray):
    """Exact de Casteljau subdivision in real arithmetic; float64 here."""
    row=np.asarray(b,float)
    left=[row[0]];right=[row[-1]]
    for _ in range(len(row)-1):
        row=(row[:-1]+row[1:])*0.5
        left.append(row[0]);right.append(row[-1])
    return np.asarray(left),np.asarray(right[::-1])


def bound_one(p:Poly, lower:float, upper:float, a:float,b:float,cfg:dict):
    """Sufficient hull test for PASS, constructive endpoint sample for FAIL.

    UNRESOLVED is not an optimization infeasibility verdict.
    """
    if not np.isfinite([lower,upper,a,b]).all() or lower>=upper or a>=b:
        raise ValueError("V141_INVALID_BOUNDS")
    guard=cfg["numerical_guard"]
    stack=[(power_to_bernstein(p),a,b,0)]
    visited=0
    worst_depth=0
    while stack:
        v,l,r,depth=stack.pop()
        visited+=1
        worst_depth=max(worst_depth,depth)
        if visited>cfg["bernstein_max_nodes_per_polynomial"]:
            return {"state":"UNRESOLVED","nodes":visited,"depth":depth}
        if float(np.min(v))>=lower+guard and float(np.max(v))<=upper-guard:
            continue
        # At the subinterval endpoints the corresponding Bernstein values are
        # the *actual polynomial* values up to roundoff.
        for x,time in ((float(v[0]),l),(float(v[-1]),r)):
            if x<lower-guard or x>upper+guard:
                return {"state":"FAIL","nodes":visited,"depth":depth,
                    "time_s":float(time),"value":x,"lower":lower,"upper":upper}
        if depth>=cfg["bernstein_max_depth"]:
            return {"state":"UNRESOLVED","nodes":visited,"depth":depth}
        left,right=bisect_bernstein(v)
        m=(l+r)/2
        stack.append((right,m,r,depth+1))
        stack.append((left,l,m,depth+1))
    return {"state":"PASS","nodes":visited,"depth":worst_depth}


def verify_piece(pieces,a,b,baseline,coeff,mult,width,q0,cfg):
    ts=a+(b-a)*np.array([.173,.511,.823])
    q,dq,ddq=dense.analytic_reference(ts,q0,mult,baseline)
    B,D,DD=qp.basis(ts,2.95,width)
    truths=(q+B@coeff,dq+D@coeff,ddq+DD@coeff)
    z=(ts-a)/(b-a)
    diff=0.
    for derivative in range(3):
        computed=np.asarray([triple[derivative](z) for triple in pieces],float).T
        diff=max(diff,float(np.max(abs(computed-truths[derivative]))))
    if diff>cfg["sample_crosscheck_absolute_error"]:
        raise ValueError("V141_LOCAL_POLYNOMIAL_NOT_ORIGINAL_QP")
    return diff


def inspect(coeff:np.ndarray,limits:dict,mult:float,width:float,
            baseline:dict,q0:np.ndarray,cfg:dict):
    if coeff.shape!=(4,29):
        raise ValueError("V141_COEFFICIENT_SHAPE")
    count=0; nodes=0; depth=0; err=0.
    failures=[];unresolved=[]
    for a,b in segments(width):
        pieces=polynomial_piece(a,b,mult,width,coeff,q0,baseline)
        err=max(err,verify_piece(pieces,a,b,baseline,coeff,mult,width,q0,cfg))
        for j,triple in enumerate(pieces):
            bounds=[(limits["lo"][j]-cfg["position_bound_tolerance_rad"],
                     limits["hi"][j]+cfg["position_bound_tolerance_rad"],"POSITION"),
                    (-limits["vmax"][j]-cfg["velocity_bound_tolerance_rad_s"],
                     limits["vmax"][j]+cfg["velocity_bound_tolerance_rad_s"],"ORIGINAL29_QDOT"),
                    (-cfg["research_acceleration_bound_rad_s2"]-cfg["research_acceleration_bound_tolerance_rad_s2"],
                     cfg["research_acceleration_bound_rad_s2"]+cfg["research_acceleration_bound_tolerance_rad_s2"],"RESEARCH_ACCEL")]
            for p,(lo,hi,name) in zip(triple,bounds):
                row=bound_one(p,float(lo),float(hi),a,b,cfg)
                count+=1;nodes+=row["nodes"];depth=max(depth,row["depth"])
                if row["state"]!="PASS":
                    event={"joint":chain.G1_JOINT_ORDER[j],"kind":name,
                           "segment":[a,b],**row}
                    if row["state"]=="FAIL":failures.append(event)
                    else:unresolved.append(event)
    state="FAIL" if failures else ("UNRESOLVED" if unresolved else "PASS")
    return {"numeric_envelope_state":state, "polynomial_bounds_checked":count,
            "bernstein_nodes_visited":nodes, "max_bisection_depth":depth,
            "local_independent_crosscheck_max_abs_error":err,
            "violations_count":len(failures),"unresolved_count":len(unresolved),
            "violation_witnesses":failures,"unresolved_witnesses":unresolved,
            "first_violation":failures[0] if failures else None,
            "first_unresolved":unresolved[0] if unresolved else None,
            "floating_point_conservative_not_formally_verified":True}


def run():
    cfg=frozen()
    c140=dense.frozen(); parent=qp.frozen(); base=chain.frozen(); pc=lp.frozen()
    if dense.FROZEN_SHA!=cfg["parent_protocol_sha"]:
        raise ValueError("V141_WRONG_V140_PROTOCOL")
    tree,limits=lp.original_kinematics(task.URDF,pc)
    masses=task.mass_inertials(task.URDF,tree)
    q0=(limits["lo"]+limits["hi"])/2
    rows=[];counter=Counter()
    for mult in cfg["frozen_grid_multipliers"]:
        ref=chain.generate(base,q0,mult)
        problem=lp.problem_at_pose(tree,masses,ref.q[590],pc)
        for target in cfg["frozen_grid_target_mps"]:
            for width in cfg["frozen_grid_halfwidth_s"]:
                solution=qp.solve(ref,problem,limits,parent,target,width)
                row={"multiplier":float(mult),"target_mps":float(target),"halfwidth_s":float(width),
                     "parent_solved":bool(solution["solved"]),
                     "parent_qp_status":solution["solver_status"],
                     "parent_reason":solution["reason"],
                     "v140_pass":False,"numeric_envelope_state":"PARENT_QP_REJECTED",
                     "v141_admitted":False,"native_physx_contact_pair_certified":False,
                     "physical_success":False}
                if solution["solved"]:
                    dense_row=dense.dense_evaluate(ref,solution,limits,tree,masses,pc,c140,width,target)
                    row["v140_pass"]=dense_row["new_kinematic_admitted"]
                    row["v140_veto_reasons"]=dense_row["veto_reasons"]
                    coeff,recon=dense.recover_coefficients(ref,solution,2.95,width)
                    row.update(inspect(coeff,limits,mult,width,base,q0,cfg))
                    row["v141_admitted"]=bool(row["v140_pass"] and row["numeric_envelope_state"]=="PASS")
                    counter["v140_pass"]+=int(row["v140_pass"])
                    counter["numeric_envelope_pass"]+=int(row["numeric_envelope_state"]=="PASS")
                    counter["newly_vetoed_from_v140"]+=int(row["v140_pass"] and not row["v141_admitted"])
                    counter["numeric_envelope_fail"]+=int(row["numeric_envelope_state"]=="FAIL")
                    counter["numeric_envelope_unresolved"]+=int(row["numeric_envelope_state"]=="UNRESOLVED")
                else:
                    counter["parent_rejected_or_unresolved"]+=1
                counter["total"]+=1;counter["v141_pass"]+=int(row["v141_admitted"])
                rows.append(row)
    if len(rows)!=45 or {(x["multiplier"],x["target_mps"],x["halfwidth_s"]) for x in rows}!=GRID:
        raise ValueError("V141_GRID_UNREGISTERED")
    return {"version":cfg["version"],"protocol_sha":FROZEN_SHA,
            "source_parent_sha":cfg["parent_sha"],
            "provenance":"CPU_ONLY_SYNTHETIC_ORIGINAL29_POLYNOMIAL_SCREEN",
            "summary":dict(counter),"rows":rows,
            "new_physx_rollouts":0,"physical_success_rate":None,
            "native_physx_contact_pair_certified":False,
            "professional_human_instep_motion_learned":False,
            "motor_execution_authorized":False}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError("NO_EVIDENCE_OVERWRITE")
    x=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
    print("SUMMARY:",json.dumps(x["summary"],sort_keys=True))
    print("WRITTEN",args.output)

if __name__=="__main__":
    main()
