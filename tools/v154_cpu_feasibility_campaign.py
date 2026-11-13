#!/usr/bin/env python3
"""Deterministic v154 CPU numerical campaign and real history qualification.

The 48 nominal solves and adversarial controls are SYNTHETIC and NOT G1 PhysX.
All historical 200Hz files are SHA pinned; never reinterpret net foot sensor
magnitudes as measured signed ground-only GRFs or CoP.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from v154_floating_base_qp import frozen,solve,N,NA
from v154_independent_wbc_audit import independent_check
from wholebody_safety_interface_v055 import G1_JOINT_ORDER
from v146_real_200hz_coupled_audit import (
    verified_sources,split_real_5ms_rows,verify_g1_metadata)
from v142_native_pair_ledger import safe_write_once

PHASES=("approach","plant","coil","swing","followthrough","recover")
SUPPORT=("left","double")
SEEDS=(0,1,2,3)
NEGATIVES=("missing_runtime_mass","missing_ground_only_force",
           "missing_collision_jacobian","wrong_dimension","nonfinite",
           "no_ground","impossible_clearance","overdriven_base_force",
           "human_unqualified_for_training")

def synthetic_problem(phase,mode,seed):
    cfg=frozen()
    if phase not in PHASES or mode not in SUPPORT or seed not in SEEDS:
        raise ValueError("V154_UNDECLARED_FIXTURE")
    names=list(G1_JOINT_ORDER)
    left_hip=names.index("left_hip_pitch_joint")
    left_knee=names.index("left_knee_joint")
    right_hip=names.index("right_hip_pitch_joint")
    right_knee=names.index("right_knee_joint")
    # Unit-test mass model is explicitly NOT original G1 mass or gravity.
    M=np.eye(N)
    h=np.zeros(N);h[2]=9.81
    amplitude={"approach":.25,"plant":.35,"coil":.65,"swing":1.8,
               "followthrough":1.1,"recover":.35}[phase]
    rng=np.random.default_rng(seed+12)
    ref=np.asarray([amplitude*(.5+float(rng.uniform(0,.5)))*
                    (1 if (j+seed)%3==0 else -1) for j in range(NA)])
    contacts=[]
    for leg in (("left",left_hip,left_knee),("right",right_hip,right_knee)):
        body,hip,knee=leg
        if body=="right" and mode=="left":
            continue
        J=np.zeros((3,N))
        J[:,0:3]=np.eye(3)
        J[0,6+hip]=.08;J[1,6+hip]=-.04
        J[2,6+hip]=.12;J[2,6+knee]=.08
        contacts.append({"body":body,"J":J.tolist(),"Jdotv":[0.,0.,0.],
                         "mu":cfg["contact_friction_coefficient_fixture"],
                         "normal_axis":"world_z",
                         "ground_only_signed_normal":True})
    grad=np.zeros(N)
    grad[6+names.index("left_wrist_yaw_joint")]=1
    grad[6+names.index("left_hip_roll_joint")]=-1
    return {
        "source_kind":"synthetic","phase":phase,"seed":seed,
        "joint_names":names,"M":M.tolist(),"h":h.tolist(),
        "q":[0.]*NA,"dq":[0.]*NA,"q_lo":[-1.5]*NA,"q_hi":[1.5]*NA,
        "dq_max":[2.]*NA,"tau_max":[cfg["torque_limit_fixture"]]*NA,
        "acc_max":[cfg["joint_accel_limit_fixture_rad_s2"]]*NA,
        "a_ref_joint":ref.tolist(),"contacts":contacts,
        "guards":[{"pair":"left_wrist_to_left_hip",
                   "signed_distance_m":.16,"min_distance_m":.08,
                   "signed_rate_mps":-.02,"acceleration_bias_mps2":0.,
                   "distance_gradient35":grad.tolist()}],
        "provenance":{"human_training_qualified":False,
                      "fixture_only":True},
        "require_human_training":False,
    }

def mutate(p,label):
    q=copy.deepcopy(p)
    if label=="missing_runtime_mass":
        q["source_kind"]="runtime"
        q["provenance"]={}
    elif label=="missing_ground_only_force":
        q["contacts"][0]["ground_only_signed_normal"]=False
    elif label=="missing_collision_jacobian":
        q["guards"]=[]
    elif label=="wrong_dimension":
        q["M"]=q["M"][:-1]
    elif label=="nonfinite":
        q["M"][0][0]=float("nan")
    elif label=="no_ground":
        q["contacts"]=[]
    elif label=="impossible_clearance":
        q["guards"][0]["signed_distance_m"]=.04
        q["guards"][0]["min_distance_m"]=.12
        q["guards"][0]["distance_gradient35"]=[0.]*N
    elif label=="overdriven_base_force":
        q["h"][0]=80.
    elif label=="human_unqualified_for_training":
        q["require_human_training"]=True
    else:
        raise ValueError("V154_UNKNOWN_NEGATIVE")
    return q

def qualified_history():
    sources=verified_sources()
    cfg=frozen()
    if list(sources)!=list(cfg["historical_source_reaudit"]["v54_original_200hz_rows"]):
        raise ValueError("V154_HISTORY_CASE_DRIFT")
    out={}
    for name,path in sources.items():
        raw=path.read_bytes()
        data=json.loads(raw)
        limits,torque=verify_g1_metadata(data)
        rows,terminal=split_real_5ms_rows(data["traces"][0],
            cfg["historical_source_reaudit"]["v54_original_200hz_rows"][name])
        qdot=np.asarray([r["joint_vel"] for r in rows])
        support=np.asarray([[r["support_foot_force_N"][foot] for foot in ("left","right")]
                             for r in rows])
        nonfoot=np.asarray([r["nonfoot_contact_N"] for r in rows])
        out[name]={
            "sha256":hashlib.sha256(raw).hexdigest(),
            "true_historical_200hz_rows":len(rows),
            "extra_terminal_note_not_a_physics_step":len(terminal),
            "last_recorded_200hz_time_s":float(rows[-1]["time_s"]),
            "max_original29_qdot_ratio":float(np.max(np.abs(qdot)/np.asarray(limits))),
            "max_nonfoot_net_force_N":float(np.max(nonfoot)),
            "support_sensor_magnitude_not_signed_ground_grf":True,
            "floating_base_mass_and_bias_per_tick_recorded":False,
            "calibrated_stance_jacobians_per_tick_recorded":False,
            "ground_only_signed_force_or_cop_recorded":False,
            "native_wrist_to_hip_mesh_distance_and_gradient":False,
            "native_contact_manifold_truth":"UNKNOWN",
            "no_physics_after_terminal_annotation":True,
            "foot_net_magnitude_samples":int(support.size),
            "real_new_physx_episodes":0,
        }
    if sum(x["true_historical_200hz_rows"] for x in out.values())!=6628:
        raise ValueError("V154_HISTORY_TOTAL_DRIFT")
    return out

def run_campaign():
    cfg=frozen()
    if list(PHASES)!=cfg["phases"] or list(SUPPORT)!=cfg["fixture_support"] or \
       list(SEEDS)!=cfg["fixture_seeds"]:
        raise ValueError("V154_FROZEN_MATRIX_DRIFT")
    nominal=[]
    for phase,mode,seed in itertools.product(PHASES,SUPPORT,SEEDS):
        p=synthetic_problem(phase,mode,seed)
        result=solve(p)
        verdict=independent_check(p,result)
        nominal.append({"phase":phase,"support":mode,"seed":seed,
                        "solver_status":result["status"],
                        "independent":verdict,
                        "naive_desired_joint_acceleration_is_not_feasibility_certified":True,
                        "numerical_success_not_real_physics":True})
    negatives=[]
    for case,mode,seed in itertools.product(NEGATIVES,SUPPORT,SEEDS):
        p=mutate(synthetic_problem("swing",mode,seed),case)
        try:
            result=solve(p)
            checked=independent_check(p,result)
            rejected=not checked["pass"]
            reason="SOLVER_"+result["status"] if rejected else "UNSAFE_ACCEPTED"
        except (ValueError,KeyError) as exc:
            rejected=True
            reason=type(exc).__name__+":"+str(exc)[:120]
        negatives.append({"adversarial":case,"support":mode,"seed":seed,
                          "rejected":rejected,"reason":reason})
    history=qualified_history()
    return {
        "version":cfg["version"],"frozen_sha":cfg["parent_sha"],
        "actual_frozen_commit":"21f42e974196621137a209aa7c979956ac3068a8",
        "nominal_cpu_synthetic_count":len(nominal),
        "nominal_solved":sum(row["independent"]["pass"] for row in nominal),
        "negative_cpu_synthetic_count":len(negatives),
        "negative_rejected":sum(row["rejected"] for row in negatives),
        "all_real_historical_200hz_rows_verified":sum(x["true_historical_200hz_rows"]
                                                      for x in history.values()),
        "new_real_physx_runs":0,"new_real_instep_success_rate":None,
        "actual_native_collision_manifold":"UNKNOWN",
        "actually_learned_professional_human_kinetics":False,
        "hardware_actuation_certified":False,
        "true_foot_ground_GRFCOP_available":False,
        "baseline_has_no_off_on_physics_parity":True,
        "nominal":nominal,"negative":negatives,"history":history,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    result=run_campaign()
    safe_write_once(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in
                      ("nominal","negative","history")},ensure_ascii=False))

if __name__=="__main__":
    main()
