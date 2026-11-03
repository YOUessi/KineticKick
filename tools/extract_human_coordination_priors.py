"""Extract human-derived gait COORDINATION prior from 29DoF retargeted clips.

No dynamics, contact forces or assumption of mechanically realizable mocap.
Produces interpretable time phase statistics, not a trained G1 model.
"""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from g1_retarget_kinematics import (
    DATA, ROOT, JOINTS, load_robot_tree, fk_g1_pose
)
FPS=30

def corr(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.std()<1e-5 or b.std()<1e-5:return None
    return round(float(np.corrcoef(a,b)[0,1]),4)

def fundamental_period(x):
    x=np.asarray(x,dtype=float)
    x=x-np.mean(x)
    if x.std()<.01:return None
    amps=[]
    for lag in range(12,min(85,len(x)-2)):
        c=corr(x[lag:],x[:-lag])
        if c is not None:amps.append((c,lag))
    if not amps:return None
    # Bias to a reasonably stable human walking cycle; not a gait label.
    best=max(amps)
    if best[0]<.2:return None
    return round(best[1]/FPS,3),best[0]

def single_clip(name,root,urdf_joints):
    f=DATA/name
    motion=np.load(f)["frames"].astype(np.float64)
    assert motion.shape[1]==36 and len(motion)>=220
    qs={name:motion[:,7+i] for i,name in enumerate(JOINTS)}
    vel_xy=np.linalg.norm(np.diff(motion[:,:2],axis=0),axis=1)*FPS
    bodies=[fk_g1_pose(frame[:3],frame[3:7],frame[7:],root,urdf_joints)
            for frame in motion]
    feet={}
    for side in ("left","right"):
        path=side+"_ankle_roll_link"
        xyz=np.stack([s[path][:3,3] for s in bodies])
        vxy=np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1)*FPS
        # Relative height from each clip minimum as a kinematic stance proxy.
        minz=float(xyz[:,2].min())
        low=xyz[:-1,2]<minz+.038
        slow=vxy<.19
        inferred=np.logical_and(low,slow)
        feet[side]={
          "ankle_roll_link_min_z_m":round(minz,3),
          "ankle_roll_link_z_p50_m":round(float(np.median(xyz[:,2])),3),
          "foot_kinematic_horizontal_velocity_p50_mps":round(float(np.median(vxy)),3),
          "proxy_stance_frame_fraction":round(float(inferred.mean()),3),
          "proxy_foot_link_xy_speed_while_inferred_stance_p90":(
               round(float(np.percentile(vxy[inferred],90)),3) if inferred.any() else None),
          "stated_limitation":"Ground contacts are INFERRED from kinematics, NOT force sensor labels."
        }
    a=qs["left_hip_pitch_joint"]
    b=qs["right_hip_pitch_joint"]
    ls=qs["left_shoulder_pitch_joint"]
    rs=qs["right_shoulder_pitch_joint"]
    lk=qs["left_knee_joint"];rk=qs["right_knee_joint"]
    stride=fundamental_period(a)
    report={
       "motion_clip_file":name,
       "frames":len(motion),
       "fps":FPS,
       "reference_speed_xy_median_mps":round(float(np.median(vel_xy)),3),
       "reference_speed_xy_p90_mps":round(float(np.percentile(vel_xy,90)),3),
       "root_height_median_m":round(float(np.median(motion[:,2])),3),
       "left_right_hip_pitch_correlation":corr(a,b),
       "left_right_shoulder_pitch_correlation":corr(ls,rs),
       "left_hip_right_shoulder_pitch_correlation":corr(a,rs),
       "right_hip_left_shoulder_pitch_correlation":corr(b,ls),
       "left_knee_abs_median_rad":round(float(np.median(np.abs(lk))),3),
       "right_knee_abs_median_rad":round(float(np.median(np.abs(rk))),3),
       "left_right_knee_correlation":corr(lk,rk),
       "pelvis_waist_yaw_variation_std_rad":round(float(np.std(qs["waist_yaw_joint"])),3),
       "reference_hip_pitch_cycle_estimate_s":stride[0] if stride else None,
       "reference_hip_pitch_autocorr_strength":stride[1] if stride else None,
       "foot_link_reference_proxies":feet,
       "cannot_compare_angles_directly_to_37dof_controller":True,
       "not_a_physical_gait_or_contact_controller":True,
    }
    print("HUMAN_REFERENCE_PRIOR",name,"root_median_mps",report["reference_speed_xy_median_mps"],
          "cycle_s",report["reference_hip_pitch_cycle_estimate_s"],
          "hip_anti_phase",report["left_right_hip_pitch_correlation"],
          "arm_anti_phase",report["left_right_shoulder_pitch_correlation"],flush=True)
    return report

def main():
    root,links,joints=load_robot_tree()
    clips=sorted(x.name for x in DATA.glob("*_selected8s.npz"))
    data=[single_clip(name,root,joints) for name in clips]
    target=ROOT/"reports/lafan1_g1_human_coordination_priors.json"
    target.write_text(json.dumps({"source":"LAFAN1 retargeted G1 29DoF reference",
         "robot_model":"29DoF rev1_0, not the official 37DoF PPO model",
         "evaluation_only":True,
         "reference_kinematics_not_force_measurements":True,
         "file_reports":data},ensure_ascii=False,indent=2))
    print("HUMAN_REFERENCE_PRIORS_READY",str(target),flush=True)
if __name__=="__main__":main()
