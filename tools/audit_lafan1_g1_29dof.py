"""Validate retargeted LAFAN1 G1-29DoF motion and extract repeatable walking clips.

Kinematic references (30 FPS) ONLY, not physically executable G1 policies.
No motion files are distributed or uploaded to GitHub.
"""
from __future__ import annotations
import json,hashlib,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

BASE=Path("/home/you/projects/kinetickick")
DATA=BASE/"data/human_motion/lafan1_g1_29dof"
ASSET=BASE/"data/assets/g1_29dof_lafan1/robot_description/g1"
FPS=30
COLS=(
    "root_x","root_y","root_z","root_qx","root_qy","root_qz","root_qw",
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint",
    "left_shoulder_yaw_joint","left_elbow_joint",
    "left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint",
    "right_shoulder_yaw_joint","right_elbow_joint",
    "right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
)
assert len(COLS)==36


def audit_urdf():
    p=ASSET/"g1_29dof_rev_1_0.urdf"
    root=ET.parse(p).getroot()
    joints={j.attrib["name"]:j for j in root.findall("joint") if j.attrib["type"]!="fixed"}
    expected=set(COLS[7:])
    if set(joints)!=expected:
        raise ValueError(f"Mismatch in motion/URDF active joint names: {sorted(set(joints)^expected)}")
    mesh_paths=[m.attrib["filename"] for m in root.findall(".//mesh")]
    missing=[name for name in mesh_paths if not (ASSET/name).exists()]
    if missing:raise FileNotFoundError(f"Missing {len(missing)} URDF referenced mesh paths: {missing[:3]}")
    bounds={}
    for name,joint in joints.items():
        lim=joint.find("limit")
        bounds[name]=[float(lim.attrib["lower"]),float(lim.attrib["upper"])] if lim is not None else None
    return bounds,len(joints),len(mesh_paths)


def pick_clip(x,frames=240,target_speed_mps=.57,min_mean_speed_mps=.22):
    # Choose a smooth, forward-moving 8s window; no access to human annotation.
    if len(x)<frames+2:raise ValueError("motion is too short")
    speed=np.linalg.norm(np.diff(x[:,:2],axis=0),axis=1)*FPS
    dq=np.max(np.abs(np.diff(x[:,7:],axis=0)),axis=1)
    best=None
    for start in range(0,len(x)-frames-1,30):
        ss=speed[start:start+frames-1]
        dd=dq[start:start+frames-1]
        average=float(np.mean(ss))
        if average<min_mean_speed_mps:continue
        frac_static=float(np.mean(ss<.12))
        score=(abs(average-target_speed_mps)*1.5 + float(np.std(ss))*.5
               + frac_static*1.5 + float(np.mean(dd))*.18)
        if best is None or score<best[0]:
            best=(score,start)
    if best is None:
        raise RuntimeError("No adequate clip interval for requested speed")
    return best[1],best[0]


def main():
    bounds,nj,nmesh=audit_urdf()
    report={
       "source":"lvhaidong/LAFAN1_Retargeting_Dataset",
       "original_motion_license":"Upstream LAFAN1 terms must be checked separately before redistribution",
       "local_dataset_repo_license":"BSD 3-Clause file retained",
       "dataset_is_kinematic_retarget":True,
       "robot_29dof_active_joints":nj,
       "csv_cols":len(COLS),
       "fps":FPS,
       "urdf_foot_and_torso_mesh_reference_count":nmesh,
       "urdf_all_meshes_present":True,
       "motions":[],
    }
    for f in sorted(DATA.glob("*.csv")):
        x=np.loadtxt(f,delimiter=",",dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(COLS) or not np.all(np.isfinite(x)):
            raise ValueError(f"Invalid retargeted motion {f}")
        quat=np.linalg.norm(x[:,3:7],axis=1)
        if np.max(np.abs(quat-1.))>.03: raise ValueError(f"Invalid root quaternions in {f}")
        dp=np.linalg.norm(np.diff(x[:,:3],axis=0),axis=1)
        dq=np.abs(np.diff(x[:,7:],axis=0))
        speed=np.linalg.norm(np.diff(x[:,:2],axis=0),axis=1)*FPS
        violations={}
        for i,name in enumerate(COLS[7:],start=7):
            if bounds[name] is None:continue
            lo,hi=bounds[name]
            bad=(x[:,i]<lo-.04)|(x[:,i]>hi+.04)
            violations[name]=float(np.mean(bad))
        is_running=f.stem.lower().startswith("run")
        start,score=pick_clip(x,target_speed_mps=(1.55 if is_running else .57),
                              min_mean_speed_mps=(1.2 if is_running else .22))
        sub=x[start:start+240]
        clip_name=f.stem+"_selected8s.npz"
        save=DATA/clip_name
        np.savez_compressed(save,frames=sub.astype(np.float32),
                            fps=np.array(FPS,dtype=np.int32),
                            joint_names=np.array(COLS,dtype=str))
        summary={
           "filename":f.name,"frames":len(x),
           "duration_seconds":round(len(x)/FPS,3),
           "root_speed_p10_p50_p90_mps":[round(float(v),3) for v in np.percentile(speed,[10,50,90])],
           "root_height_p10_p50_p90_m":[round(float(v),3) for v in np.percentile(x[:,2],[10,50,90])],
           "max_frame_root_delta_m":round(float(dp.max()),4),
           "max_frame_joint_delta_rad":round(float(dq.max()),4),
           "root_quaternion_max_norm_error":round(float(np.max(np.abs(quat-1.))),6),
           "urdf_joint_bound_violation_frame_fraction":{
               k:round(v,5) for k,v in violations.items() if v>0},
           "selected_8s_start_frame":int(start),
           "selected_8s_score":round(float(score),4),
           "selected_8s_file":str(save),
           "source_csv_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),
        }
        report["motions"].append(summary)
        print("LAFAN1_G1_MOTION_VALIDATED",f.name,
              "frames",len(x),"duration_s",summary["duration_seconds"],
              "median_speed_mps",summary["root_speed_p10_p50_p90_mps"][1],
              "joint_limit_violation_items",len(summary["urdf_joint_bound_violation_frame_fraction"]),
              "selected_start_frame",start,flush=True)
    target=BASE/"reports/lafan1_g1_walk_motion_audit.json"
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("LAFAN1_G1_AUDIT_DONE",str(target),"num_motions",len(report["motions"]),flush=True)

if __name__=="__main__":main()
