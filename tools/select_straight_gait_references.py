"""Select actual STRAIGHT LAFAN1 retargeted G1 walking clips and align to a visible A -> B route.

Only one rigid XY/yaw world-coordinate transform is applied per source clip.
Every joint angle and the 30-FPS motion timeline is untouched: NOT synthetic
target-conditioned policy, NOT physics-driven walking, NOT a new AI gait model.
"""
from __future__ import annotations
from pathlib import Path
import json, math, hashlib
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path("/home/you/projects/kinetickick")
RAW=ROOT/"data/human_motion/lafan1_g1_29dof"
TARGET=ROOT/"data/human_motion/straight_reference29"
TARGET.mkdir(parents=True,exist_ok=True)
# Verified candidate windows from 30Hz original CSV, chosen before UI changes:
# Speed and trajectory straightness measured over the full ground-truth root XY.
OPTIONS={
 "WALK SLOW":("walk4_subject1.csv",1200,1380),
 "WALK NORMAL":("walk2_subject3.csv",555,735),
 "WALK BRISK":("walk1_subject1.csv",2415,2595),
 "RUN STRAIGHT":("run1_subject2.csv",3930,4020),
}
FPS=30

def build_clip(path,start,stop):
    raw=np.loadtxt(RAW/path,delimiter=",",dtype=np.float32)
    if raw.ndim!=2 or raw.shape[1]!=36:raise ValueError((path,raw.shape))
    if start<0 or stop>len(raw):raise ValueError("Out-of-range selection")
    data=raw[start:stop].copy()
    first=data[0,:2].astype(float)
    last=data[-1,:2].astype(float)
    displacement=last-first
    horizontal_dist=float(np.linalg.norm(displacement))
    if horizontal_dist<1.5:raise ValueError("Clip does not travel purposefully")
    direction=math.atan2(displacement[1],displacement[0])
    dxy=np.diff(data[:,:2],axis=0)
    path_len=float(np.linalg.norm(dxy,axis=1).sum())
    efficiency=horizontal_dist/path_len
    if efficiency<.985:raise ValueError((path,"non-straight motion",efficiency))
    d=np.max(np.abs(dxy),axis=1)
    if float(d.max())>.13:raise ValueError((path,"trajectory discontinuity",float(d.max())))
    yaw=Rotation.from_quat(data[:,3:7]).as_euler("xyz")[:,2]
    heading_vs_forward=np.arctan2(np.sin(yaw-direction),np.cos(yaw-direction))
    align=float(np.median(np.cos(heading_vs_forward)))
    if align<.95:raise ValueError((path,"not forward-facing walking",align))
    yaw_std=float(np.std(heading_vs_forward))
    if yaw_std>.18:raise ValueError((path,"body yaw fluctuates too much",yaw_std))
    # Rz(-direction) maps physical reference motion direction to world +X.
    transform=Rotation.from_euler("z",-direction)
    source_rot=Rotation.from_quat(data[:,3:7])
    aligned=data.copy()
    xyz=data[:,:3].astype(float)-np.array([first[0],first[1],0.])
    xyz=transform.apply(xyz)
    xyz[:,:2]+=np.array([0.,0.])
    aligned[:,:3]=xyz
    aligned[:,3:7]=(transform*source_rot).as_quat()
    lateral_max=float(np.max(np.abs(xyz[:,1])))
    measured_final=[float(v) for v in aligned[-1,:3]]
    assert abs(measured_final[0]-horizontal_dist)<1e-3
    assert abs(measured_final[1])<1e-3
    duration=(len(aligned)-1)/FPS
    return aligned,{
      "original_csv":str(RAW/path),
      "source_first_frame":start,"source_stop_frame_exclusive":stop,
      "source_sha256":hashlib.sha256((RAW/path).read_bytes()).hexdigest(),
      "frames":len(aligned),"source_fps":FPS,"play_duration_s":round(duration,3),
      "goal_world_xy":[round(horizontal_dist,4),0.],
      "actual_segment_path_length_m":round(path_len,4),
      "straightness_ratio":round(efficiency,5),
      "forward_facing_mean_cos":round(align,5),
      "relative_yaw_std_rad":round(yaw_std,5),
      "max_deviation_from_goal_line_m":round(lateral_max,4),
      "median_motion_speed_mps":round(float(np.median(np.linalg.norm(dxy,axis=1)*FPS)),4),
      "max_interframe_xy_step_m":round(float(d.max()),4),
      "world_rigid_rotation_rad":round(-direction,7),
      "source_joint_angles_unchanged":True,
      "source_kinematic_not_dynamics":True,
      "target_is_overlay_not_actual_goal_policy":True
    }


def main():
    manifest={}
    for label,(file,start,end) in OPTIONS.items():
        clip,meta=build_clip(file,start,end)
        target=TARGET/(label.lower().replace(" ","_")+".npz")
        np.savez_compressed(target,
            frames=clip.astype(np.float32),fps=np.int32(FPS),
            label=label,original_file=file)
        manifest[label]={"file":str(target),**meta}
        print("STRAIGHT_GAIT_SELECTED",label,"forward_m",meta["goal_world_xy"][0],
              "path_efficiency",meta["straightness_ratio"],
              "reference_speed_mps",meta["median_motion_speed_mps"],
              "n_frames",meta["frames"],flush=True)
    output=ROOT/"reports/straight_gait_reference_manifest.json"
    output.write_text(json.dumps({
        "scenario":"human-derived reference motion along explicit A->B line",
        "source_is_kinematic_not_learned_navigation":True,
        "cycle_behavior":"one-shot stop at B, manual replay at A",
        "clips":manifest
    },ensure_ascii=False,indent=2))
    print("STRAIGHT_GAIT_REFERENCES_READY",output,flush=True)
if __name__=="__main__":main()
