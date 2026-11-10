#!/usr/bin/env python3
"""v49 independent read-only real G1 kinetic chain comparison and figure.

This is a 5ms original PhysX *Jacobian foot velocity decomposition*, not
licensed high-speed human instep dynamics. Hip/knee/ankle momentary WORLD -Y
signed foot-point velocity shares are NOT percentages of muscular impulse,
power or energy. G1 joint qdot peak times are likewise NOT a valid human
anatomical proximal-to-distal transfer label.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/kinetic_chain_observability_v049.json"
INPUT="data/research_human_kicks/kinetic_chain_v049/kinetic_chain_audit_v049.json"
DATA="data/research_human_kicks/kinetic_chain_v049/kinetic_chain_physx_v049.npz"
OUTPUT="data/research_human_kicks/kinetic_chain_v049/paired_kinetic_chain_comparison.json"

def sha(path):
 h=hashlib.sha256()
 with (ROOT/path).open("rb") as f:
  for part in iter(lambda:f.read(1<<20),b""):h.update(part)
 return h.hexdigest()

def compare():
    cfg=json.loads((ROOT/STUDY).read_text())
    for name,digest in cfg["source_sha256"].items():
        if sha(name)!=digest:raise ValueError("Physics source SHA drift "+name)
    manifest=json.loads((ROOT/INPUT).read_text())
    if manifest["source_protocol_sha256"]!=sha(STUDY) or manifest["v49_physics_npz_sha256"]!=sha(DATA):
        raise ValueError("Feature tensor/source provenance mismatch")
    if len(manifest["episodes"])!=12:raise ValueError("Only original 12 physical G1 shots")
    prev=json.loads((ROOT/"reports/v048_preimpact_full_physics_audit.json").read_text())
    lines=[]
    for i in range(0,12,2):
        a,b=manifest["episodes"][i:i+2]
        if a["case"]!=b["case"] or a["controller"]!="original_v32" or b["controller"]!="learned_preimpact_v48":
            raise ValueError("Original paired ball and learned body sources misaligned")
        pa=next(row for row in prev["cases"] if row["id"]==a["case"])
        result=dict(
          case=a["case"],baseline_verified_right_instep=a["lawful_right_instep"],
          v48_verified_right_instep=b["lawful_right_instep"],
          original_true_G1_foot_contact_forward_mps=a["original_forward_foot_contact_speed_mps"],
          v48_true_G1_foot_contact_forward_mps=b["original_forward_foot_contact_speed_mps"],
          paired_foot_forward_speed_delta_mps=b["original_forward_foot_contact_speed_mps"]-a["original_forward_foot_contact_speed_mps"],
          original_measured_ball_speed_xy_mps=pa["original_v32"]["true_outgoing_ball_speed_xy_mps"],
          learned_measured_ball_speed_xy_mps=pa["learned_preimpact_v48"]["true_outgoing_ball_speed_xy_mps"],
          true_original_ball_translational_energy_change_25ms_J=a["actual_ball_translation_energy_delta_25ms_J"],
          true_v48_ball_translational_energy_change_25ms_J=b["actual_ball_translation_energy_delta_25ms_J"],
          real_0p43kg_ball_energy_change_delta_J=b["actual_ball_translation_energy_delta_25ms_J"]-a["actual_ball_translation_energy_delta_25ms_J"],
          original_hip_forward_foot_velocity_contribution_mps=a["original_forward_velocity_from_parts_mps"]["hip"],
          v48_hip_forward_foot_velocity_contribution_mps=b["original_forward_velocity_from_parts_mps"]["hip"],
          original_knee_forward_foot_velocity_contribution_mps=a["original_forward_velocity_from_parts_mps"]["knee"],
          v48_knee_forward_foot_velocity_contribution_mps=b["original_forward_velocity_from_parts_mps"]["knee"],
          original_ankle_forward_foot_velocity_contribution_mps=a["original_forward_velocity_from_parts_mps"]["ankle"],
          v48_ankle_forward_foot_velocity_contribution_mps=b["original_forward_velocity_from_parts_mps"]["ankle"],
          original_root_forward_foot_velocity_contribution_mps=a["original_forward_velocity_from_parts_mps"]["base"],
          v48_root_forward_foot_velocity_contribution_mps=b["original_forward_velocity_from_parts_mps"]["base"],
          original_right_hip_knee_ankle_peak_qdot_phase_is_non_decreasing=a["original_hip_knee_ankle_speed_peak_order_example"],
          v48_right_hip_knee_ankle_peak_qdot_phase_is_non_decreasing=b["original_hip_knee_ankle_speed_peak_order_example"],
          actual_original_G1_physx_foot_decomposition_max_error_mps=a["foot_velocity_world_decomposition_max_numerical_error_mps"],
          actual_v48_G1_physx_foot_decomposition_max_error_mps=b["foot_velocity_world_decomposition_max_numerical_error_mps"])
        if not np.isfinite(result["paired_foot_forward_speed_delta_mps"]):
            raise ValueError("G1 original physical contact signal invalid")
        lines.append(result)
    cue=[e for e in manifest["episodes"] if e["controller"]=="learned_preimpact_v48"]
    old=[e for e in manifest["episodes"] if e["controller"]=="original_v32"]
    result=dict(
        version="0.1.49",
        exact_frozen_physics_source_SHA=sha(STUDY),
        extracted_G1_physx_feature_data_SHA=sha(DATA),
        original_source_ball_cases=6,
        original_fullbody_physical_episodes=12,
        checked_true_first_right_foot_ball_contact_as_event_anchor=True,
        original_foot_velocity_components_sum_to_real_PhysX_foot_point_velocity=True,
        original_hip_world_goal_direction_positive_count=sum(x["original_forward_velocity_from_parts_mps"]["hip"]>0 for x in old),
        original_knee_world_goal_direction_negative_count=sum(x["original_forward_velocity_from_parts_mps"]["knee"]<0 for x in old),
        original_ankle_world_goal_direction_negative_count=sum(x["original_forward_velocity_from_parts_mps"]["ankle"]<0 for x in old),
        learned_hip_world_goal_direction_positive_count=sum(x["original_forward_velocity_from_parts_mps"]["hip"]>0 for x in cue),
        learned_knee_world_goal_direction_negative_count=sum(x["original_forward_velocity_from_parts_mps"]["knee"]<0 for x in cue),
        learned_ankle_world_goal_direction_negative_count=sum(x["original_forward_velocity_from_parts_mps"]["ankle"]<0 for x in cue),
        original_hip_knee_ankle_joint_qdot_peak_nondecreasing_examples=sum(x["original_hip_knee_ankle_speed_peak_order_example"] for x in old),
        learned_hip_knee_ankle_joint_qdot_peak_nondecreasing_examples=sum(x["original_hip_knee_ankle_speed_peak_order_example"] for x in cue),
        mean_abs_paired_foot_forward_speed_change_mps=float(np.mean([abs(v["paired_foot_forward_speed_delta_mps"]) for v in lines])),
        maximum_abs_paired_foot_forward_speed_change_mps=float(np.max([abs(v["paired_foot_forward_speed_delta_mps"]) for v in lines])),
        average_original_foot_forward_speed_mps=float(np.mean([v["original_forward_foot_contact_speed_mps"] for v in old])),
        average_v48_foot_forward_speed_mps=float(np.mean([v["original_forward_foot_contact_speed_mps"] for v in cue])),
        training_or_new_PhysX_done_in_v49=False,
        source_is_valid_human_proximal_to_distal_biomechanics=False,
        statistical_population_naturalness_inference_supported=False,
        innovation_claim_allowed=False,
        human_professional_instestep_physics_verified=False,
        limits=[
          "Signed Jacobian velocity components are not percentages of energy/torque, not a muscle activation sequence, not a measure of professional human shooting form",
          "All G1 cases used in prior v48 test; no new held-out data or independent physical seed",
          "One 3DSP 20-frame video-estimated human clip does not identify human impact force, contact normal or anatomical kinetic chain",
          "Even if qdot peak ordering is reversed, angular velocities in different G1 joints and their coordinate axes are not directly equivalent to human segment angular speed ordering",
          "Actual ball energy uses 0.43kg PhysX mass and only change in translational energy from -25ms to +25ms, excludes ball spin and external rolling energy",
          "No new actor, physics rollout, reward tuning or default policy promotion"
        ],paired_cases=lines)
    dest=ROOT/OUTPUT
    if dest.exists():raise FileExistsError("Keep immutable original physics research analysis")
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    # Plot world-signed contribution for original and new at impact; not
    # fractions, and zero positive/negative sign has physical meaning.
    xx=np.arange(6)
    values=["base","hip","knee","ankle"]
    oldw=[[old[i]["original_forward_velocity_from_parts_mps"][name] for i in range(6)] for name in values]
    neww=[[cue[i]["original_forward_velocity_from_parts_mps"][name] for i in range(6)] for name in values]
    fig,axes=plt.subplots(2,1,figsize=(12.6,8.6),sharex=True)
    for ax,collection,label in zip(axes,(oldw,neww),("Original frozen v32","New learned v48 29DoF PPO")):
        bar_width=.18
        for j,(name,entry) in enumerate(zip(values,collection)):
            ax.bar(xx+(j-1.5)*bar_width,entry,bar_width,label=name.capitalize())
        ax.axhline(0,color="black",linewidth=.8)
        ax.set_ylabel("Signed foot-point velocity contribution toward goal (-world Y), m/s")
        ax.set_title(label+" — at first measured foot/ball contact")
        ax.grid(axis="y",alpha=.25)
        ax.legend(loc="upper right",fontsize=8,ncol=4)
    axes[-1].set_xticks(xx,[row["case"] for row in lines],rotation=13)
    fig.suptitle("KineticKick v0.1.49 | Measured original 5ms G1 whole-body instep velocity composition")
    fig.text(.015,.008,
       "Real 0.43kg G1 soccer ball / original PD and URDF. Signed Jacobian foot velocity decomposition is NOT an anatomical human kinetic chain or measured muscle power transfer. Six previously used v48 PhysX ball cases only.",
       fontsize=8)
    fig.tight_layout(rect=(0,.065,1,.965))
    path=ROOT/"media/v049_true_foot_velocity_body_decomposition.png"
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=155)
    plt.close(fig)
    print("V049_KINETIC_CHAIN_COMPARISON",json.dumps(dict(
      measured_original_shots=len(old),
      mean_foot_mps=result["average_original_foot_forward_speed_mps"],
      mean_new_foot_mps=result["average_v48_foot_forward_speed_mps"],
      mean_absolute_paired_velocity_delta_mps=result["mean_abs_paired_foot_forward_speed_change_mps"],
      original_hip_positive=result["original_hip_world_goal_direction_positive_count"],
      original_knee_negative=result["original_knee_world_goal_direction_negative_count"],
      original_ankle_negative=result["original_ankle_world_goal_direction_negative_count"],
      original_hip_knee_ankle_peak_order=result["original_hip_knee_ankle_joint_qdot_peak_nondecreasing_examples"],
      new_hip_knee_ankle_peak_order=result["learned_hip_knee_ankle_joint_qdot_peak_nondecreasing_examples"],
      chart=str(path),
      result_sha=sha(OUTPUT)),ensure_ascii=False),flush=True)
    return result

if __name__=="__main__":compare()
