#!/usr/bin/env python3
"""v51 compare REAL 120Hz human soccer pose and original G1 29DoF FK timeline.

CMU human 10_02 / 10_05 are the ONLY clips with 8+ contiguous ball-cluster
launch-motion frames; their first high-ball-speed event is NOT verified actual
foot-ball collision. Robot original v32 and v48 first actual physical contact
are observed by Isaac PhysX. Alignment is exploratory, NOT contact-ground-truth
supervision, human/robot anatomical equivalence, or professional kick mechanics.

Six semantically similar 3D segment-direction cosines share a written schema.
Do not compare original 8C human ankle-toe features to non-existent G1 toe
joints, nor claim this makes a 29DoF human-to-G1 motor retargeter.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from g1_contact_coordination_projection_v040 import TorchG1Relations,load_robot_source,interpolate_source
from g1_coordination_bridge_v039 import BODY

ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/cmu_g1_cross_morph_v051.json"
CMU="data/research_human_kicks/cmu_soccer_120hz_v050/processed_v050"
HUMAN_FEATURE_INDEX=(0,1,4,5,6,7)
SIX_FEATURE_NAMES=(
 "left_hip_knee_ankle_cos",
 "right_hip_knee_ankle_cos",
 "left_shoulder_elbow_wrist_cos",
 "right_shoulder_elbow_wrist_cos",
 "left_femur_to_trunk_cos",
 "right_femur_to_trunk_cos")
def sha(path):
 h=hashlib.sha256()
 with (ROOT/path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def vector_cos(a,b):
 den=torch.linalg.vector_norm(a,dim=-1)*torch.linalg.vector_norm(b,dim=-1)
 if not bool((den>1e-9).all()):
  raise ValueError("Robot original G1 link segment vector degenerate")
 return ((a*b).sum(dim=-1)/den).clamp(-1,1)

def g1_six_semantic_relations(fk,root,q):
 if root.ndim!=2 or q.ndim!=2 or len(root)!=len(q) or q.shape[1]!=29:
  raise ValueError("Original G1 29 joint FK expects batched Bx29 and root Bx7")
 with torch.no_grad():
  _,links=fk(root,q,with_links=True)
  p={name:links[link][:,:3,3] for name,link in BODY.items()}
  pelvis=.5*(p["left_hip"]+p["right_hip"])
  shoulder=.5*(p["left_shoulder"]+p["right_shoulder"])
  trunk=shoulder-pelvis
  out=torch.stack((
   vector_cos(p["left_hip"]-p["left_knee"],p["left_ankle"]-p["left_knee"]),
   vector_cos(p["right_hip"]-p["right_knee"],p["right_ankle"]-p["right_knee"]),
   vector_cos(p["left_shoulder"]-p["left_elbow"],p["left_wrist"]-p["left_elbow"]),
   vector_cos(p["right_shoulder"]-p["right_elbow"],p["right_wrist"]-p["right_elbow"]),
   vector_cos(trunk,p["left_knee"]-p["left_hip"]),
   vector_cos(trunk,p["right_knee"]-p["right_hip"])),dim=-1)
 if out.shape!=(len(q),6) or not bool(torch.isfinite(out).all()):
  raise ValueError("Wrong G1 6 semantic body relations")
 return out.numpy()

def human_six_timed(name,relative,summary):
 clip=next((z for z in summary["motions"] if z["id"]==name),None)
 if not clip:raise ValueError("Human source not registered")
 if clip["observed_ball_kinematics_quality"]!="continuous_fast_outgoing_ball_marker_motion_no_true_contact_truth":
  raise ValueError("Cannot time-align human launch without sufficiently observed ball marker travel")
 event=clip["inferred_event"]
 if event.get("verified_true_foot_ball_contact") is not False:
  raise ValueError("No CMU impact force ground truth exists")
 anchor=float(event["inferred_ball_launch_onset_s"])
 with np.load(ROOT/CMU/(name+".npz"),allow_pickle=False) as z:
  ts=z["time_s"].astype(float)
  channels=z["marker_derived_human_8_cosine_relations"].astype(float)
  valid=z["primary_marker_valid"].all(axis=1)
 if channels.ndim!=2 or channels.shape[1]!=8 or len(ts)!=len(channels) or not np.isfinite(channels[valid]).all():
  raise ValueError("Human measured pose 8D source missing")
 wanted=relative+anchor
 if wanted.min()<ts.min() or wanted.max()>ts.max():
  raise ValueError("Human observed phase not available")
 if not np.all(np.interp(wanted,ts,valid.astype(float))>.999):
  raise ValueError("Do not interpolate through invalid human C3D pose frames")
 x=np.stack([np.interp(wanted,ts,channels[:,i]) for i in HUMAN_FEATURE_INDEX],axis=-1)
 if x.shape!=(len(relative),6) or not np.isfinite(x).all():raise ValueError("Human timeline corrupted")
 return x,anchor

def robot_six_timed(source,relative):
 robot=load_robot_source(ROOT/source)
 anchor=float(robot["contact_time"])
 r,q=interpolate_source(robot,relative+anchor)
 fk=TorchG1Relations(robot["joint_names"])
 signal=g1_six_semantic_relations(fk,torch.tensor(r,dtype=torch.float64),torch.tensor(q,dtype=torch.float64))
 return signal,anchor

def audit():
 study=json.loads((ROOT/STUDY).read_text())
 if study["version"]!="0.1.51" or study["true_human_contact_time_qualified"] or study["training_G1_controller_in_this_trial"]:
  raise ValueError("No true human contact annotation or trained G1 retargeter permitted")
 for p,h in study["source_sha256"].items():
  if sha(p)!=h:raise ValueError("Frozen CMU/G1 source data changed "+p)
 human=json.loads((ROOT/CMU/"quality_audit.json").read_text())
 relative=np.arange(-.40,0.,.02).round(8)
 assert relative.shape==(20,) and relative[-1]==-.02
 humans={}
 robots={}
 for name in study["qualified_human_ball_marker_launch_clips"]:
  humans[name]=human_six_timed(name,relative,human)
 for name,source in study["original_robot_PhysX_runs"].items():
  robots[name]=robot_six_timed(source,relative)
 details=[]
 arrays={}
 for human_name,(h,h_anchor) in humans.items():
  for robot_name,(b,b_anchor) in robots.items():
   # A true human and robot do not share matching body skeleton offsets.
   # Report both ABS geometry mismatch and RELATIVE motion trend mismatch.
   h_rel=h-h[:1]
   b_rel=b-b[:1]
   mismatch_abs=float(np.mean((h-b)**2))
   trend=float(np.mean((h_rel-b_rel)**2))
   h_change=float(np.sqrt(np.mean(h_rel**2)))
   b_change=float(np.sqrt(np.mean(b_rel**2)))
   details.append(dict(
      human_source=human_name,robot_source=robot_name,
      human_ball_marker_motion_onset_s=h_anchor,
      robot_true_PhysX_impact_time_s=b_anchor,
      events_are_NOT_groundtruth_synchronized=True,
      compare_6_3D_segment_cosine_features_not_joint_angle=True,
      absolute_cross_morphology_cosine_MSE=mismatch_abs,
      within_own_body_first_frame_relative_motion_MSE=trend,
      human_relative_motion_rms=h_change,
      G1_relative_motion_rms=b_change,
      relative_motion_amplitude_ratio_G1_to_human=None if h_change<1e-9 else b_change/h_change,
      number_of_comparable_body_channels=6,
      expected_true_human_instestep_available=False,
      no_physical_G1_rollout_or_controller_improvement_claim=True))
   arrays[f"{human_name}_cos"]=h.astype(np.float32)
   arrays[f"{robot_name}_cos"]=b.astype(np.float32)
 if len(details)!=4:raise ValueError("Two original human pilots vs two frozen G1 controllers expected")
 out=ROOT/study["output_dir"]
 if out.exists():raise FileExistsError("Do not overwrite original exploratory cross-morph data")
 out.mkdir(parents=True)
 artifact=out/"cmu_g1_6d_exploratory_timeline.npz"
 np.savez_compressed(artifact,relative_event_time_s=relative.astype(np.float32),
     semantic_6_channels=np.asarray(SIX_FEATURE_NAMES),
     human_launch_time_is_verified_contact=np.asarray(False),
     G1_robot_real_PhysX_impact_verified=np.asarray(True),
     human_vs_G1_anatomical_joint_equivalence=np.asarray(False),
     original_120hz_CMU_and_50hz_G1_features_interpolated_without_source_gap=np.asarray(True),
     **arrays)
 z=dict(version="0.1.51",study_sha256=sha(STUDY),
    six_semantic_channels=list(SIX_FEATURE_NAMES),
    actual_human_ball_motion_observed_clips=list(humans),
    actual_G1_PhySX_controllers=list(robots),
    new_trained_PPO_policy=False,
    new_simulator_rollouts_count=0,
    contact_conditioned_true_human_kinetics_qualified=False,
    independent_heldout_generalization=False,
    number_of_exploratory_human_robot_pairs=len(details),
    original_120hz_human_features_and_50hz_G1_PhysX_URDF_compared=True,
    feature_npz_sha256=sha(str(artifact.relative_to(ROOT))),
    limitations=[
      "Human t=0 is the first sufficiently observed WEP ball-marker-triad speed burst, NOT force-measured instep impact",
      "G1 t=0 is original IsaacLab force-based contact, so event timing is a research convenience and not genuinely synchronized human-robot impact",
      "Only six named segment-angle cosines have comparable schemas; robot link origins do not reproduce human measured anatomical joint centers",
      "No ankle-toe channel is transferred because G1 does not have a verified human-analog toe segment",
      "Two human source clips and two retrospectively known robot controllers; no statistical generalization, no new robot action execution, no certified human motion naturalness or elite kicking force",
      "Absolute cross-body cosines and first-frame relative changes are descriptive evidence, not a learned human control policy or verified human kinetic-chain transfer"],
    comparisons=details)
 m=out/"cross_morphology_audit.json"
 m.write_text(json.dumps(z,ensure_ascii=False,indent=2)+"\n")
 print("V051_REAL_HUMAN_G1_CROSSMORPH_TIMELINE",json.dumps(dict(
    n_pairs=len(details),channels=6,figure_data_sha=z["feature_npz_sha256"],
    scores=[dict(h=x["human_source"],g=x["robot_source"],
      cosine_MSE=round(x["absolute_cross_morphology_cosine_MSE"],6),
      relative_MSE=round(x["within_own_body_first_frame_relative_motion_MSE"],6),
      relative_amplitude_G1_to_human=round(x["relative_motion_amplitude_ratio_G1_to_human"],4)) for x in details])),flush=True)

if __name__=="__main__":audit()
