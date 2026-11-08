#!/usr/bin/env python3
"""v39: verify human/robot *semantic* coordination interface via G1 URDF FK.

The interface has 11 named skeleton direction-cosines. Robot sources are
link/joint-frame proxy coordinates (NOT calibrated anatomical landmarks).
Human sources are broadcast-estimated 3D pose joint approximations. Matching
11 names does not establish physical morphology equivariance, naturalness,
impact alignment, foot face or feasible action. NO motion retargeting here.
"""
from __future__ import annotations
import argparse,hashlib,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from build_professional_shot_coordination_v039 import PAIRS

ROOT=Path(__file__).resolve().parents[1]
URDF=ROOT/"configs/paid_original/paid_main_resolved.urdf"
CENTER=ROOT/"reports/v032_learned_center.json"
OLD_B=ROOT/"reports/v032_learned_heldout_v028_near_slow_17.json"
BODY={
 "left_hip":"left_hip_pitch_link","right_hip":"right_hip_pitch_link",
 "left_knee":"left_knee_link","right_knee":"right_knee_link",
 "left_ankle":"left_ankle_roll_link","right_ankle":"right_ankle_roll_link",
 "left_shoulder":"left_shoulder_pitch_link","right_shoulder":"right_shoulder_pitch_link",
 "left_elbow":"left_elbow_link","right_elbow":"right_elbow_link",
 "left_wrist":"left_wrist_roll_link","right_wrist":"right_wrist_roll_link",
}
# These are link-origin proxies, not genuine anatomical joint centers.
SEGMENT_ENDPOINTS={
 "trunk":("pelvis_center","shoulder_center"),
 "pelvis_line":("right_hip","left_hip"),
 "shoulder_line":("right_shoulder","left_shoulder"),
 "left_thigh":("left_hip","left_knee"),
 "right_thigh":("right_hip","right_knee"),
 "left_shank":("left_knee","left_ankle"),
 "right_shank":("right_knee","right_ankle"),
 "left_upper_arm":("left_shoulder","left_elbow"),
 "right_upper_arm":("right_shoulder","right_elbow"),
 "left_forearm":("left_elbow","left_wrist"),
 "right_forearm":("right_elbow","right_wrist"),
}
assert len(SEGMENT_ENDPOINTS)==11

def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

class G1URDFFK:
 def __init__(self,path=URDF):
  robot=ET.parse(path).getroot()
  names=[l.attrib["name"] for l in robot.findall("link")]
  self.links=set(names)
  self.root="pelvis"
  self.joints=[]
  for joint in robot.findall("joint"):
   typ=joint.attrib["type"]
   parent=joint.find("parent").attrib["link"]
   child=joint.find("child").attrib["link"]
   origin=joint.find("origin")
   origin_values=origin.attrib if origin is not None else {}
   pos=np.fromstring(origin_values.get("xyz","0 0 0"),sep=" ",dtype=float)
   rpy=np.fromstring(origin_values.get("rpy","0 0 0"),sep=" ",dtype=float)
   axis=np.fromstring(joint.find("axis").attrib.get("xyz","0 0 1"),sep=" ",dtype=float) if joint.find("axis") is not None else np.array([0.,0.,1.])
   if pos.shape!=(3,) or rpy.shape!=(3,) or axis.shape!=(3,) or typ not in ("fixed","revolute"):
    raise ValueError("Unsupported URDF joint schema")
   self.joints.append(dict(name=joint.attrib["name"],typ=typ,parent=parent,child=child,
                           pos=pos,rot=Rotation.from_euler("xyz",rpy).as_matrix(),axis=axis))
  self.active_names={j["name"] for j in self.joints if j["typ"]=="revolute"}
  if len(self.active_names)!=29 or len(self.links)<30:raise ValueError("Wrong G1 original 29DoF topology")
 def forward(self,root_pose,joint_names,joint_positions):
  root_pose=np.asarray(root_pose,float)
  pos=np.asarray(joint_positions,float)
  if root_pose.shape!=(7,) or pos.shape!=(29,) or len(joint_names)!=29 or set(joint_names)!=self.active_names:
   raise ValueError("Wrong original G1 root/joint arrays")
  if not np.isfinite(root_pose).all() or not np.isfinite(pos).all():
   raise ValueError("Nonfinite original G1 state")
  T=np.eye(4)
  T[:3,:3]=Rotation.from_quat([*root_pose[4:7],root_pose[3]]).as_matrix()
  T[:3,3]=root_pose[:3]
  found={self.root:T}
  q=dict(zip(joint_names,pos))
  pending=self.joints.copy()
  while pending:
   progress=False
   for joint in pending[:]:
    if joint["parent"] not in found:continue
    X=np.eye(4);X[:3,:3]=joint["rot"];X[:3,3]=joint["pos"]
    if joint["typ"]=="revolute":
     axis=joint["axis"]
     if np.linalg.norm(axis)<1e-8:raise ValueError("Bad original joint axis")
     Q=np.eye(4);Q[:3,:3]=Rotation.from_rotvec(axis/np.linalg.norm(axis)*q[joint["name"]]).as_matrix()
     X=X@Q
    found[joint["child"]]=found[joint["parent"]]@X
    pending.remove(joint)
    progress=True
   if not progress:raise ValueError("Cyclic or disconnected original URDF")
  if set(found)!=self.links:raise ValueError("Incomplete original link tree")
  return found

def robot_relations(link_poses):
  points={semantic:link_poses[link][:3,3] for semantic,link in BODY.items()}
  points["pelvis_center"]=(points["left_hip"]+points["right_hip"])/2
  points["shoulder_center"]=(points["left_shoulder"]+points["right_shoulder"])/2
  directions={}
  for k,(a,b) in SEGMENT_ENDPOINTS.items():
   diff=points[b]-points[a]
   length=np.linalg.norm(diff)
   if length<1e-5:raise ValueError("Robot link segment length vanished")
   directions[k]=diff/length
  values=np.array([np.clip(directions[a]@directions[b],-1.,1.) for a,b in PAIRS])
  if values.shape!=(11,) or not np.isfinite(values).all():
   raise ValueError("Invalid robot relation vector")
  return values

def audit_fk(run,fk):
    kin=run["kinematics"][0]
    names=run["joint_names"]
    errors=[]
    # Independent simulator-measured G1 foot BODY origin is the fixed
    # numerical ground truth; never calibrate FK by fitting a scale/offset.
    selected=[0,len(kin)//4,len(kin)//2,3*len(kin)//4,len(kin)-1]
    for i in selected:
      row=kin[i]
      model=fk.forward(row["root_pose"],names,row["joint_pos"])
      for side in ("left","right"):
       actual=np.asarray(row["feet"][side]["position"],float)
       estimated=model[side+"_ankle_roll_link"][:3,3]
       errors.append(dict(index=i,time_s=float(row["time_s"]),side=side,
          euclidean_error_m=float(np.linalg.norm(actual-estimated)),
          predicted_xyz_m=estimated.tolist(),actual_xyz_m=actual.tolist()))
    return errors

def main():
  p=argparse.ArgumentParser()
  p.add_argument("--output-dir",default="data/research_human_kicks/3dsp_cvpr2024/processed_v039/g1_bridge")
  args=p.parse_args()
  output=ROOT/args.output_dir
  if output.exists():raise FileExistsError("Refusing to overwrite actual bridge outputs")
  fk=G1URDFFK()
  clips=[("center",CENTER),("near_slow_17",OLD_B)]
  results=[]
  sequences=[]
  for name,source in clips:
   r=json.loads(source.read_text())
   assert r["root_fixed"] is False and r["post_initialization_state_writes"] is False
   original=r["records"][0]
   assert original["contact_time_s"] is not None
   checks=audit_fk(r,fk)
   maximum=max(x["euclidean_error_m"] for x in checks)
   if maximum>.004:
    raise ValueError(f"Original URDF link FK cannot reproduce actual PhysX body poses: {name} max_error={maximum}")
   frames=r["kinematics"][0]
   ts=np.asarray([q["time_s"] for q in frames])
   if not np.allclose(np.diff(ts),.02,atol=2e-6):
    raise ValueError("Source 50 Hz G1 control samples are incomplete")
   tc=float(original["contact_time_s"])
   target_t=tc-.5+.04*np.arange(20)
   if target_t[0]<ts[0] or target_t[-1]>ts[-1]:
    raise ValueError("No 20-frame valid near-impact robot trace")
   original_pose=np.asarray([x["root_pose"] for x in frames])
   original_joint=np.asarray([x["joint_pos"] for x in frames])
   out=[]
   residuals=[]
   for t in target_t:
    j=int(np.searchsorted(ts,t))
    j=min(max(j,1),len(ts)-1)
    i=j-1
    alpha=(t-ts[i])/(ts[j]-ts[i])
    xyz=(1-alpha)*original_pose[i,:3]+alpha*original_pose[j,:3]
    quats=original_pose[[i,j],3:][:,[1,2,3,0]]
    interp=Slerp([ts[i],ts[j]],Rotation.from_quat(quats))([t]).as_quat()[0]
    root=np.r_[xyz,interp[3],interp[:3]]
    angles=(1-alpha)*original_joint[i]+alpha*original_joint[j]
    pose=fk.forward(root,r["joint_names"],angles)
    out.append(robot_relations(pose))
    residuals.append(float(min(abs(t-ts[i]),abs(t-ts[j]))))
   arr=np.asarray(out,dtype=np.float32)
   if arr.shape!=(20,11) or not np.isfinite(arr).all() or np.max(abs(arr))>1+1e-6:
    raise ValueError("Invalid robot relation feature")
   sequences.append(arr)
   results.append(dict(case=name,original_report=str(source.relative_to(ROOT)),
        original_sha256=digest(source),contact_time_s=tc,
        right_instep=original["measured_surface"]=="instep",
        original_task_success=original["success"],
        fk_checked_link_positions=len(checks),
        maximum_fk_position_error_m=maximum,
        tested_samples=checks,
        target_event_window_s=list(target_t[[0,-1]]),
        interpolated_from_original_50hz=True,
        relation_samples_25hz=20,
        available_contact_truth_for_robot=True,
        available_contact_truth_for_human_3dsp=False))
  output.mkdir(parents=True)
  outfile=output/"g1_kinematic_relations_v039.npz"
  np.savez_compressed(outfile,relation_cosines=np.asarray(sequences),case=np.asarray([v["case"] for v in results]),
       is_robot_joint_fk_proxy=np.asarray(True),
       synchronized_robot_contact_truth=np.asarray(True),
       human_broadcast_impact_truth=np.asarray(False),
       relation_names=np.asarray([a+"__"+b for a,b in PAIRS]))
  manifest=dict(
     version="0.1.39",purpose="G1 link-origin FK feature-interface parity, NOT human biomechanical similarity",
     original_urdf_sha256=digest(URDF),original_g1_29dof=True,
     robot_reference_control="v32 accepted/failed original PhysX, untouched",
     source_root_joint_order_original=True,
     record_count=len(results),robot_frames=40,
     output_relative=str(outfile.relative_to(ROOT)),output_sha256=digest(outfile),
     common_11_feature_names=[a+"__"+b for a,b in PAIRS],
     robot_position_source="URDF link frames after FK from measured floating root+joints; validated against recorded PhysX left/right ankle body origins at 5 timestamps per clip",
     human_position_source="Video-estimated 3DSP keypoints with unverified metric scale/foot shape",
     semantic_matched_not_anatomically_equivalent=True,
     human_contact_time_unknown=True,
     human_to_robot_joint_angles_model_trained=False,
     human_to_robot_kinematic_retargeting_validated=False,
     actual_g1_action_policy_trained_from_3dsp=False,
     limits=["G1 link-frame origins are not calibrated human anatomical joint centers",
       "Human 3DSP 3D estimates lack foot heel/toe, exact impact, support force, skeletal bone lengths and external impulse labels",
       "G1 event anchors use measured contact; human 3DSP centers use coarse shot annotation, so time axes are NOT genuinely event-aligned",
       "Matching 11 feature names does NOT solve inverse kinematics, inverse dynamics, G1 joint feasibility or human naturalness",
       "No new G1 PhysX rollout was run; previously saved original physical trajectories were analyzed"],
     records=results)
  (output/"bridge_audit.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
  print("V039_G1_BRIDGE_READY",json.dumps({
      "num_clips":len(sequences),"sampled_frames":40,
      "maximum_fk_body_error_m":max(v["maximum_fk_position_error_m"] for v in results),
      "g1_output_sha":manifest["output_sha256"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
