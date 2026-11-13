#!/usr/bin/env python3
"""Frozen v156 48 real-source URDF (CPU) experiments + strict truth provenance.

Only URDF coefficients are physically sourced. The 48 poses/velocities are
deterministic *artificial samples*. This is not PhysX, training, or shooting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from v156_real_urdf_pin_model import (
    frozen, OriginalUrdfModel, independent_invariants, independent_time_derivative)
from v156_independent_runtime_and_history_gate import independent_real_history_audit

def run_frozen():
    cfg=frozen()
    model=OriginalUrdfModel()
    rows=[]
    index=0
    for phase in cfg["phases"]:
        for mode in cfg["modes"]:
            for seed in cfg["seeds"]:
                q,v=model.fixture(phase,mode,seed)
                result=model.evaluate(q,v)
                inv=independent_invariants(model,q,v,result)
                fd=None
                if index < cfg["independent_kinematic_checks"]:
                    fd=independent_time_derivative(model,q,v,result)
                rows.append({
                    "case":index,"phase":phase,"support_hypothesis_only":mode,"seed":seed,
                    "urdf_source_sha256":model.sha,
                    "M35_min_eigen":inv["min_M_eigenvalue"],
                    "M35_condition_number":float(np.linalg.cond(result["M"])),
                    "rnea_CRBA_max_error":inv["max_rnea_consistency_error"],
                    "aba_RNEA_max_error":inv["max_aba_error"],
                    "Jdotv_fd_max_error":max(fd.values()) if fd else None,
                    "center_of_mass_urdf_w":result["com_model_w"].tolist(),
                    "wrist_hip_link_origin_distance_m":inv["wrist_hip_link_origin_distance_m"],
                    "model_only_not_actual_g1_physics":True,
                    "urdf_foot_J6_is_not_contact_grf_or_cop":True,
                    "actor_collider_contact_pair":"UNKNOWN",
                    "real_physx_success":None,
                })
                index+=1
    if len(rows)!=cfg["model_checks"] or sum(x["Jdotv_fd_max_error"] is not None for x in rows)!=cfg["independent_kinematic_checks"]:
        raise ValueError("V156_PREDECLARED_MATRIX_DRIFT")
    hist=independent_real_history_audit()
    return {
        "version":"v0.1.156",
        "pre_frozen_protocol_sha":"29809c5a8e7156363880808fa73afdc452258cec",
        "original_urdf_sha256":model.sha,
        "original_urdf_total_mass_kg":model.total_mass_urdf_kg,
        "joint_count_isaaclab":29,"floating_base_velocity_dimension":6,
        "Pin_q_dimension":36,"Pin_v_dimension":35,
        "original_actuator_order_verified":True,
        "urdf_has_original_sensor_foot_frame_ids":True,
        "source_kind":"ORIGINAL_G1_URDF_CPU_OFFLINE_ONLY",
        "full_35d_model_checks":len(rows),
        "finite_difference_6d_kinematic_checks":sum(x["Jdotv_fd_max_error"] is not None for x in rows),
        "max_identity_error":max(max(x["rnea_CRBA_max_error"],x["aba_RNEA_max_error"]) for x in rows),
        "max_6d_Jdotv_fd_error":max(x["Jdotv_fd_max_error"] or 0 for x in rows),
        "min_M_eig_over_grid":min(x["M35_min_eigen"] for x in rows),
        "min_wrist_hip_link_origin_distance_m":min(x["wrist_hip_link_origin_distance_m"] for x in rows),
        "max_wrist_hip_link_origin_distance_m":max(x["wrist_hip_link_origin_distance_m"] for x in rows),
        "actual_original_physx_historical_audit":hist,
        "no_human_motion_training":True,
        "new_real_physx_episode_count":0,
        "native_left_wrist_hip_contact_manifold":"UNKNOWN",
        "legal_high_speed_instep_episodes":0,
        "real_instability_or_recovery_certified":False,
        "new_real_29dof_controller_not_yet_admitted":True,
        "cases":rows,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args()
    result=run_frozen()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({k:v for k,v in result.items()
          if k not in ("cases","actual_original_physx_historical_audit")},ensure_ascii=False))

if __name__=="__main__":
    main()
