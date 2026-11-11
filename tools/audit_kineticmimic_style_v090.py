#!/usr/bin/env python3
"""v90 optical-human vs real-G1 event-aligned 6D body geometry audit.

Two CMU 120Hz optical marker launches of the SAME subject (10_02/10_05);
not exact impact contact, synchronized GRF or professional instep power truth.
Three v85 ORIGINAL 29DoF G1 PhysX center scenarios with VERIFIED true robot
foot-ball first contact timestamps (200Hz) and original G1 URDF kinematics.

This is a stress test for KineticMimic *geometry/style loss*, not training an
actor or demonstrating human biomechanics, and no adjustments to old physics.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from kineticmimic_physics_v090 import sha,load_original_physics,strict_physics
from kineticmimic_loss_v090 import semantic_phase_cost,phase_chain_hypothesis_loss,TruthScope,validate_scopes
from cmu_g1_semantic_bridge_v051 import g1_six_semantic_relations
from g1_contact_coordination_projection_v040 import TorchG1Relations
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

HUMAN_INDICES=(0,1,4,5,6,7)

def human_relative(source_root,meta,manifest):
 if sha(source_root/meta["file"])!=meta["sha256"]:
  raise ValueError("Original 120Hz CMU optical provenance drift")
 quality=json.loads((source_root/manifest["quality_path"]).read_text())
 motion=next((x for x in quality["motions"] if x["id"]==meta["clip"]),None)
 if (not motion or motion["inferred_event"]["verified_true_foot_ball_contact"] is not False or
      motion["observed_ball_kinematics_quality"]!="continuous_fast_outgoing_ball_marker_motion_no_true_contact_truth"):
  raise ValueError("Missing verified distinction: CMU ball motion onset is NOT true collision")
 anchor=float(motion["inferred_event"]["inferred_ball_launch_onset_s"])
 with np.load(source_root/meta["file"],allow_pickle=False) as z:
  if z["original_human_subject"].item()!="10" or z["true_impact_event_available"].item() or z["force_plate_data_available"].item():
   raise ValueError("Original human subject / contact truth misqualified")
  ts=np.asarray(z["time_s"],float)
  cos=np.asarray(z["marker_derived_human_8_cosine_relations"],float)
  valid=np.asarray(z["primary_marker_valid"],bool).all(-1)
 wanted=anchor+np.arange(-.4,0,.02)
 if not (ts.min()<wanted.min() and wanted.max()<ts.max()):
  raise ValueError("Original human phase not present in real 120Hz clip")
 if np.any(np.interp(wanted,ts,valid.astype(float))<.99999):
  raise ValueError("Missing optical body observations in source human phase")
 out=np.stack([np.interp(wanted,ts,cos[:,i]) for i in HUMAN_INDICES],-1)
 if out.shape!=(20,6) or not np.isfinite(out).all():
  raise ValueError("Original human semantically normalized phase invalid")
 return out,anchor

def robot_relative(raw,first_impact_s):
 k=raw["kinematics"][0]
 t=np.asarray([z["time_s"] for z in k],float)
 original=np.asarray([z["root_pose"] for z in k],float)
 q=np.asarray([z["joint_pos"] for z in k],float)
 if (q.shape!=(len(t),29) or original.shape!=(len(t),7) or
     not np.isfinite(q).all() or not np.isfinite(original).all() or
     len(t)<100):
  raise ValueError("Insufficient true original G1 kinematic trajectory")
 wanted=first_impact_s+np.arange(-.4,0,.02)
 if wanted.min()<t.min() or wanted.max()>t.max():
  raise ValueError("Original real G1 joint state missing near actual ball contact")
 angles=np.stack([np.interp(wanted,t,q[:,j]) for j in range(29)],-1)
 xyz=np.stack([np.interp(wanted,t,original[:,j]) for j in range(3)],-1)
 q_xyzw=original[:,[4,5,6,3]]
 quat=Slerp(t,Rotation.from_quat(q_xyzw))(wanted).as_quat()
 root=np.concatenate([xyz,quat[:,[3,0,1,2]]],axis=1)
 fk=TorchG1Relations(raw["joint_names"])
 signal=g1_six_semantic_relations(fk,torch.tensor(root,dtype=torch.float64),
                                     torch.tensor(angles,dtype=torch.float64))
 if signal.shape!=(20,6) or not np.isfinite(signal).all():
  raise ValueError("Original measured G1 geometry missing")
 return signal

def report_case(source_root,manifest,key):
 item=manifest["robots"][key]
 path=Path(item["path"])
 if sha(path)!=item["sha256"]:raise ValueError("Original 200Hz free-root G1 source drift")
 raw=json.loads(path.read_text())
 phys=load_original_physics(path,item["sha256"],key)
 score=strict_physics(phys)
 if phys.first_impact_s is None:raise ValueError("No original measured foot-ball collision time")
 robo=robot_relative(raw,phys.first_impact_s)
 costs={}
 for hkey,hconf in manifest["humans"].items():
  human,anchor=human_relative(source_root,hconf,manifest)
  delta=(robo-robo[0])-(human-human[0])
  mse=float(np.mean(delta**2))
  cost=float(semantic_phase_cost(torch.tensor(robo[None],dtype=torch.float64),
                                 torch.tensor(human[None],dtype=torch.float64)))
  reversed_human=human[::-1].copy()
  reversal=float(semantic_phase_cost(torch.tensor(robo[None],dtype=torch.float64),
                                     torch.tensor(reversed_human[None],dtype=torch.float64)))
  costs[hkey]=dict(original_human_optical_launch_onset_s=anchor,
    geometric_relative_mse_not_naturalness=mse,
    monotone_softDTW_6D_not_human_impact_loss=cost,
    temporal_reversal_negative_6D_softDTW=reversal,
    human_true_impact_timing=False,human_GRF_or_muscle_power=False)
 use=(phys.times>=phys.first_impact_s-.4-1e-8)&(phys.times<phys.first_impact_s-.0001)
 rates=phys.dq[use]
 if not len(rates)>=60:raise ValueError("Original 200Hz preimpact G1 joint velocity coverage inadequate")
 chain,peaks=phase_chain_hypothesis_loss(torch.tensor(rates[None],dtype=torch.float64))
 return dict(case=key,source_original_200hz_SHA256=item["sha256"],
  original_foot_ball_measured_time_s=phys.first_impact_s,
  measured_5ms_physics_frames=phys.true_physics_rows,
  true_nonphysical_terminal_annotation=phys.terminal_annotation_count,
  original_joint_order_valid=raw["joint_names"]==list(G1_JOINT_ORDER),
  original_tracked_G1_29dof_ratio_max=float(phys.qdot_ratio.max()),
  original_strict_physics=score,source_abs_foot_ball_impulse_proxy_Ns=phys.estimated_foot_ball_scalar_impulse_Ns,
  source_instep_impulse_fraction=phys.measured_instep_impulse_fraction,
  source_contact_ball_speed_mps=phys.observed_ball_speed_5_to_105ms_mps,
  true_original_20step_G1_human_geom_relations=costs,
  normalized_200hz_G1_limb_group_soft_peak_phases=peaks,
  uncalibrated_experimental_phase_order_cost=float(chain),
  real_G1_wrist_hip_native_collider_manifold_certified=False,
  actual_human_professional_force_truth_available=False)

def run(source_root,protocol,output):
 if output.exists():raise FileExistsError("Cannot replace original geometric research result")
 cfg=json.loads(protocol.read_text())
 if (cfg["version"]!="v0.1.90-style" or
     set(cfg["robots"])!={"center_v32","center_cmu02","center_cmu05"} or
     set(cfg["humans"])!={"10_02","10_05"} or
     cfg["human_impact_truth_available"] or cfg["human_GRf_truth_available"]):
  raise ValueError("Wrong frozen style/control real G1 evidence")
 if sha(source_root/cfg["quality_path"])!=cfg["quality_sha256"]:
  raise ValueError("Original human optical metadata altered")
 validate_scopes(TruthScope())
 rows=[report_case(source_root,cfg,k) for k in cfg["robots"]]
 result=dict(version="v0.1.90",scope="existing 3 real v85 G1 physical policies, 2 same-person CMU optical-only subjects",
  protocol_sha256=sha(protocol),
  human_has_verified_true_instep_football_contact=False,
  person_level_holdout_independent_test=False,
  human_power_transfer_and_GRF_training_labels=False,
  new_physx_G1_experiments=0,
  evidence_kind="original 200Hz G1 measured PhysX vs 120Hz optical relative six cosines; no new trained actor",
  negative_temporal_reversal_test=True,canonical_physical_g1_runs=rows,
  do_not_promote_default_strategy=True)
 output.parent.mkdir(exist_ok=True,parents=True)
 output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
 print("KINETICMIMIC_V090_HUMAN_GEOMETRIC_REAL_G1_COUPLING",json.dumps(
   {r["case"]:dict(physics_pass=r["original_strict_physics"]["strict_pass"],
   contact_s=r["original_foot_ball_measured_time_s"],
   nonphysical_terminal=r["true_nonphysical_terminal_annotation"],
   relative_geometry_MSE={k:round(x["geometric_relative_mse_not_naturalness"],5)
                           for k,x in r["true_original_20step_G1_human_geom_relations"].items()})
    for r in rows}),flush=True)
 return result

if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--source-root",type=Path,required=True)
 p.add_argument("--protocol",type=Path,default=ROOT/"configs/kineticmimic_style_v090.json")
 p.add_argument("--output",type=Path,default=ROOT/"reports/v090_human_robot_relative_geometry.json")
 a=p.parse_args()
 torch.set_num_threads(4)
 if torch.cuda.is_available():raise RuntimeError("Original-only offline G1 and human geometry audit, no GPU")
 run(a.source_root,a.protocol,a.output)
