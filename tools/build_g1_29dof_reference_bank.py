"""Pack LAFAN1 G1-29DoF retargeted mocap into a motion-tracking reference bank.

Training input ONLY. No physical validity, torque limits, contact force labels
or proof of real-robot execution. Do not transfer joint indexing to 37DoF G1.
"""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np
from g1_retarget_kinematics import DATA,ROOT,URDF,JOINTS,load_robot_tree,fk_g1_pose
FPS=30
FRAMES=240

def main():
    base_link,links,joints=load_robot_tree()
    clips=sorted(DATA.glob("*_selected8s.npz"))
    if len(clips)!=5:raise RuntimeError("Expected 5 audited G1 retargeted clips")
    streams=[]
    foot_names=["left_ankle_roll_link","right_ankle_roll_link"]
    for file in clips:
        x=np.load(file)["frames"].astype(np.float32)
        assert x.shape==(FRAMES,36)
        root_xyz=x[:,:3].copy()
        root_xyz[:,:2]-=root_xyz[0,:2]
        q=x[:,7:].copy()
        quat=x[:,3:7].copy()
        root_vel=np.gradient(root_xyz,1/FPS,axis=0).astype(np.float32)
        joint_vel=np.gradient(q,1/FPS,axis=0).astype(np.float32)
        footworld=[]
        for row in x:
            body=fk_g1_pose(row[:3],row[3:7],row[7:],base_link,joints)
            footworld.append([body[name][:3,3] for name in foot_names])
        footworld=np.asarray(footworld,dtype=np.float32)
        footworld[:,:,:2]-=x[0,:2]
        footvel=np.gradient(footworld,1/FPS,axis=0).astype(np.float32)
        mask=[]
        for side in range(2):
            z=footworld[:,side,2]
            v=np.linalg.norm(footvel[:,side,:2],axis=-1)
            # PSEUDO contact based on foot height/speed; no real load sensor.
            on=(z<(z.min()+.038))&(v<.19)
            mask.append(on)
        contact_proxy=np.stack(mask,axis=-1)
        streams.append({
          "label":file.stem,"q":q,"dq":joint_vel,
          "root_xyz":root_xyz,"root_xyzw":quat,
          "root_vel":root_vel,"feet_xyz":footworld,
          "feet_vel":footvel,"foot_stance_proxy":contact_proxy
        })
        print("REFERENCE_BANK_SOURCE",file.stem,"frames",FRAMES,
              "stance_proxy_frac",contact_proxy.mean(axis=0).round(3).tolist(),flush=True)
    out=DATA.parent/"g1_29dof_reference_bank_v0.npz"
    labels=np.asarray([s["label"] for s in streams])
    arrays={key:np.stack([s[key] for s in streams])
            for key in ("q","dq","root_xyz","root_xyzw","root_vel",
                        "feet_xyz","feet_vel","foot_stance_proxy")}
    np.savez_compressed(out,labels=labels,joint_names=np.asarray(JOINTS),
                        fps=np.int32(FPS),**arrays)
    summary={
       "path":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
       "source_lafan1_g1_29dof":True,
       "URDF_SHA256":hashlib.sha256(URDF.read_bytes()).hexdigest(),
       "motion_labels":labels.tolist(),
       "number_motions":len(streams),"frames_per_motion":FRAMES,
       "fps":FPS,
       "shapes":{key:list(val.shape) for key,val in arrays.items()},
       "no_force_contact_ground_truth":True,
       "stance_labels_are_kinematic_proxies_not_physics":True,
       "no_learned_policy":True,
       "for_29dof_morphology_only":True,
       "original_mocap_license_requires_independent_review":True
    }
    meta=ROOT/"reports/g1_29dof_reference_bank_manifest.json"
    meta.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print("REFERENCE_BANK_BUILT",str(out),str(meta),flush=True)
if __name__=="__main__":main()
