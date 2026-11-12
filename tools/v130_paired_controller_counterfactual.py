#!/usr/bin/env python3
"""v130 paired original v54 G1 physical traces: strictly matched frames only.

Comparing old v32, CMU10_02, CMU10_05 from the identical frozen seed/ball
position exposes how late a 29DoF reference affects the *physical trajectory*.
No physics is launched, no human force truths or collider manifold invented.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import v128_v54_event_geometry as previous
import v130_real_com_support_work as primary


def measured(case: str, frozen: dict, src: dict, root: Path):
    p = root / ("v054_" + case + "_contact_diagnostics.json")
    previous.require_sha(p, src["input_sha256"][p.name])
    raw = json.loads(p.read_text())
    trace = previous.validate_rows(raw, primary.CASES[case], frozen["dt_s"])
    rr = raw["traces"][0][:len(trace["t"])]
    tau = primary._finite_matrix(rr,"joint_torque_estimate",29)
    comvel = primary._finite_matrix(rr,"whole_body_com_velocity",3)
    return {
        **trace,
        "tau": tau,
        "comvel":comvel,
        "raw_sha256":src["input_sha256"][p.name]
    }


def evaluate_pair(base: dict, treatment: dict, cfg: dict, label: str):
    """Two trajectories only compared on their truly shared sampled frames."""
    n=min(len(base["t"]),len(treatment["t"]))
    if n<565 or not np.allclose(base["t"][:n],treatment["t"][:n],
                               atol=2e-5,rtol=0):
        raise ValueError("NO_SYNCHRONIZED_PHYSICS_COUNTERFACTUAL")
    qdiff=np.max(np.abs(base["q"][:n]-treatment["q"][:n]),axis=1)
    dqd=np.max(np.abs(base["dq"][:n]-treatment["dq"][:n]),axis=1)
    comdiff=np.linalg.norm(base["com"][:n]-treatment["com"][:n],axis=1)
    load_diff=np.linalg.norm(base["support"][:n]-treatment["support"][:n],axis=1)
    work0=np.einsum("ij,ij->i",base["tau"][:n],base["dq"][:n])
    work1=np.einsum("ij,ij->i",treatment["tau"][:n],treatment["dq"][:n])
    t=base["t"][:n]
    # Thresholds are descriptive sensor-precision choices, not fitted to PASS.
    def first_delta(x,eps):
        v=np.flatnonzero(x>eps)
        return float(t[v[0]]) if len(v) else None
    masks=primary.windows(t,base["impact"],cfg)
    early=t<1.405 - 1e-6   # external v123 independently established diagnostic
    intervals={"pre_reference_observation":early, **masks}
    summaries={}
    for name,mask in intervals.items():
        if not mask.any():
            raise ValueError("MISSING_PHASE_FOR_PAIRED_COMPARISON")
        summaries[name]={
            "shared_physical_frames":int(mask.sum()),
            "max_abs_29q_difference_rad":float(qdiff[mask].max()),
            "max_abs_29qdot_difference_rads":float(dqd[mask].max()),
            "max_com_world_position_difference_m":float(comdiff[mask].max()),
            "max_euclidean_support_proxy_difference_N":float(load_diff[mask].max()),
            "paired_robot_motor_signed_work_delta_J":
                float(np.sum((work1-work0)[mask])*cfg["dt_s"]),
            "max_abs_per_frame_29dof_power_delta_W":
                float(np.max(np.abs((work1-work0)[mask]))),
        }
    return {
        "control_comparison":"frozen_v32_vs_"+label,
        "shared_original_physical_frames":int(n),
        "shared_last_time_s":float(t[-1]),
        "first_q_difference_over_1e_9_rad_s":first_delta(qdiff,1e-9),
        "first_qdot_difference_over_1e_9_rads_s":first_delta(dqd,1e-9),
        "first_COM_difference_over_1e_7_m_s":first_delta(comdiff,1e-7),
        "all_traces_same_seed_same_ball_geometry":True,
        "inference_type":"paired_descriptive_observational_simulator_not_human_learning",
        "native_contact_pair_witness":False,
        "physical_rollout_executed_in_v130":False,
        "intervals":summaries,
    }


def paired_audit():
    cfg=primary.frozen()
    src=previous.frozen_protocol()
    root=Path(src["data_root"])
    base=measured("frozen_v32",cfg,src,root)
    return {
        "version":"v0.1.130",
        "protocol_commit":primary.FROZEN_SHA,
        "first_difference_measurement_exploratory_not_prereg_threshold":True,
        "new_physx_episodes":0,
        "pairs":[evaluate_pair(base,measured(x,cfg,src,root),cfg,x)
                 for x in ("cmu_10_02","cmu_10_05")],
    }


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--output",type=Path,
                   default=primary.ROOT/"reports/v130_paired_original_controller.audit.json")
    args=a.parse_args()
    result=paired_audit()
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    for pair in result["pairs"]:
        print(pair["control_comparison"],
              "shared",pair["shared_original_physical_frames"],
              "first_q_delta",pair["first_q_difference_over_1e_9_rad_s"],
              "first_com_delta",pair["first_COM_difference_over_1e_7_m_s"],
              "pre_ref_qmax",pair["intervals"]["pre_reference_observation"]["max_abs_29q_difference_rad"],
              flush=True)

if __name__=="__main__":
    main()
