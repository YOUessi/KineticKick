#!/usr/bin/env python3
"""Read-only actual 29DoF physics-body coordination effect of v40 candidate.

No scoring as human naturalness. The 3DSP source's 'shot annotation' is NOT
the G1 force-triggered contact time. A 11D proxy deviation is not a biomech
or visual-aesthetics metric.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from g1_contact_coordination_projection_v040 import (
    TorchG1Relations,load_robot_source,interpolate_source,build_human_relative_target)
import torch

ROOT=Path(__file__).resolve().parents[1]

def digest(p):
 return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

def audit(case,protocol_path="configs/g1_coordination_physics_v040.json"):
    study=json.loads((ROOT/protocol_path).read_text())
    registered=next(x for x in study["cases_in_conditional_order"] if x["id"]==case)
    baseline=ROOT/registered["baseline"]
    candidate=ROOT/registered["candidate"]
    if not candidate.exists():raise FileNotFoundError(candidate)
    ref=json.loads(baseline.read_text())
    new=json.loads(candidate.read_text())
    if ref["joint_names"]!=new["joint_names"]:
        raise ValueError("Joint semantics changed")
    human=np.load(ROOT/"data/research_human_kicks/3dsp_cvpr2024/processed_v039/coordination_3dsp_v039.npz",allow_pickle=False)
    idx=np.flatnonzero(human["clip_id"]=="00001")
    if len(idx)!=1 or human["split"][idx[0]]!="train":
        raise ValueError("Cannot reinterpret independent test as training source")
    source=human["relation_cosines"][idx[0]].astype(float)
    robot0=load_robot_source(baseline)
    robot1=load_robot_source(candidate)
    if len(ref["kinematics"][0])!=700 or len(new["kinematics"][0])!=700:
        raise ValueError("Only complete 700-control-step PhysX data evaluated")
    original_contact=float(ref["records"][0]["contact_time_s"])
    actual_contact=float(new["records"][0]["contact_time_s"])
    t=original_contact-.5+.04*np.arange(20)
    fk=TorchG1Relations(robot0["joint_names"])
    R0,Q0=interpolate_source(robot0,t)
    R1,Q1=interpolate_source(robot1,t)
    with torch.no_grad():
      v0=fk(torch.tensor(R0,dtype=torch.float64),torch.tensor(Q0,dtype=torch.float64)).numpy()
      v1=fk(torch.tensor(R1,dtype=torch.float64),torch.tensor(Q1,dtype=torch.float64)).numpy()
    target,signal=build_human_relative_target(v0,source,
                    scale=json.loads((ROOT/"configs/g1_coordination_generation_v040.json").read_text())["source_relative_scale"])
    # Evaluation is deliberately unchanged in world-time sampling, not
    # candidate-event-warped into favorable time registration.
    weights=(t<original_contact-.04).astype(float)
    mse=lambda v:float(np.sum((v-target)**2*weights[:,None])/
                             (weights.sum()*11))
    m0=mse(v0)
    m1=mse(v1)
    absq=abs(Q1-Q0)
    joints=robot0["joint_names"]
    groups={
      "stance_left_leg":[i for i,n in enumerate(joints) if n.startswith("left_") and any(z in n for z in ("hip","knee","ankle"))],
      "swing_right_leg":[i for i,n in enumerate(joints) if n.startswith("right_") and any(z in n for z in ("hip","knee","ankle"))],
      "waist":[i for i,n in enumerate(joints) if n.startswith("waist_")],
      "upper_body":[i for i,n in enumerate(joints) if any(z in n for z in ("shoulder","elbow","wrist"))]
    }
    assert sum(len(x) for x in groups.values())==29
    group_quantities={g:dict(joints=len(indices),
         maximum_physical_joint_position_delta_rad=float(absq[:,indices].max()),
         mean_physical_joint_position_delta_rad=float(absq[:,indices].mean()))
       for g,indices in groups.items()}
    measured_qdot0=np.asarray([r["joint_vel"] for r in ref["traces"][0]])
    measured_qdot1=np.asarray([r["joint_vel"] for r in new["traces"][0]])
    total=dict(
      version="0.1.40",case=case,
      original_source_sha256=digest(registered["baseline"]),
      candidate_source_sha256=digest(registered["candidate"]),
      human_train_proxy_clip_id="00001",human_exact_impact_truth=False,
      source_PhysX_actual_foot_ball_contact_time_s=original_contact,
      candidate_PhysX_actual_foot_ball_contact_time_s=actual_contact,
      comparison_samples=20,
      prespecified_precontact_samples=int(weights.sum()),
      original_pose_to_HUMAN_PROXY_mse=m0,
      new_pose_to_HUMAN_PROXY_mse=m1,
      new_minus_original_proxy_mse=m1-m0,
      mean_robot_relation_change=float(np.mean(abs(v1-v0))),
      maximum_robot_relation_change=float(np.max(abs(v1-v0))),
      max_actual_qrad_delta_in_window=float(absq.max()),
      max_actual_qdot_rad_s_change_over_full_physics=float(abs(measured_qdot1-measured_qdot0).max()),
      groups=group_quantities,
      original_goal=bool(ref["records"][0]["goal"]),
      candidate_goal=bool(new["records"][0]["goal"]),
      original_ball_outgoing_s=float(ref["records"][0]["contact_time_s"]),
      candidate_ball_outgoing_s=float(new["records"][0]["contact_time_s"]),
      interpretation="Actual physical movement of a G1 after altering whole-body joint references. Source is unverified human 3DSP relative joint-direction cue, NOT human motion reproduction or naturalness."
    )
    return total,dict(time_s=t,baseline=v0,physical=v1,human_proxy_target=target)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--case",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    p=ROOT/a.output
    if p.exists():raise FileExistsError(p)
    result,arrays=audit(a.case)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("V040_PHYSICAL_COORDINATION_EFFECT",json.dumps({
       k:result[k] for k in ("case","prespecified_precontact_samples",
         "original_pose_to_HUMAN_PROXY_mse","new_pose_to_HUMAN_PROXY_mse",
         "new_minus_original_proxy_mse","max_actual_qrad_delta_in_window",
         "max_actual_qdot_rad_s_change_over_full_physics","original_goal","candidate_goal")},
       ensure_ascii=False),flush=True)

if __name__=="__main__":main()
