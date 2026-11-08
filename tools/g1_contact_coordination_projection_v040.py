"""v40 differentiable G1-URDF coordination projection (offline reference only).

This is a *research feasibility prototype*, not an established human-to-robot
retargeter or human explosive instep policy. The human source is game broadcast
video-lifted 3D with NO verified contact time, foot normal, GRF, or kick force.
We only transfer RELATIVE change of 11 normalized segment cosine relations.
Physical validity must be checked separately with the original G1/PhysX gate.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from g1_coordination_bridge_v039 import G1URDFFK,BODY,SEGMENT_ENDPOINTS
from build_professional_shot_coordination_v039 import PAIRS

ROOT=Path(__file__).resolve().parents[1]
URDF=ROOT/"configs/paid_original/paid_main_resolved.urdf"

class TorchG1Relations(torch.nn.Module):
    def __init__(self,joint_names,urdf=URDF):
        super().__init__()
        origin=G1URDFFK(urdf)
        if len(joint_names)!=29 or set(joint_names)!=origin.active_names:
            raise ValueError("G1 29 joint semantic order required")
        self.names=tuple(joint_names)
        self.link_order=("pelvis",)+tuple(j["child"] for j in origin.joints)
        self.edges=origin.joints
        self.joint_index={name:i for i,name in enumerate(self.names)}
        self.origin_matrices=[]
        self.axis_skews=[]
        I=np.eye(4)
        for joint in self.edges:
            T=I.copy()
            T[:3,:3]=joint["rot"]
            T[:3,3]=joint["pos"]
            self.origin_matrices.append(torch.tensor(T,dtype=torch.float64))
            a=joint["axis"].astype(float)
            a=a/np.linalg.norm(a)
            K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0.]])
            self.axis_skews.append(torch.tensor(K,dtype=torch.float64))
        self.BODY=BODY

    def forward(self,root_pose,angles,with_links=False):
        if root_pose.ndim!=2 or root_pose.shape[1]!=7 or angles.ndim!=2 or angles.shape[1]!=29 or len(root_pose)!=len(angles):
            raise ValueError("Root Bx7 and q Bx29 required")
        if not bool(torch.isfinite(angles).all()) or not bool(torch.isfinite(root_pose).all()):
            raise ValueError("Nonfinite FK input")
        B=len(angles)
        dtype=angles.dtype
        device=angles.device
        if root_pose.dtype!=dtype or root_pose.device!=device:
            raise ValueError("Floating root and q must share device/dtype")
        quat=root_pose[:,3:]
        quat=quat/quat.norm(dim=-1,keepdim=True)
        w,x,y,z=quat.unbind(dim=-1)
        R=torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
            2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
            2*(x*z-w*y), 2*(y*z+w*x),1-2*(x*x+y*y)),dim=-1).reshape(B,3,3)
        T=torch.eye(4,device=device,dtype=dtype).expand(B,4,4).clone()
        T=torch.cat((torch.cat((R,root_pose[:,:3,None]),dim=-1),
                    torch.tensor([0,0,0,1.],device=device,dtype=dtype).expand(B,1,4)),dim=-2)
        links={"pelvis":T}
        ident=torch.eye(3,device=device,dtype=dtype).expand(B,3,3)
        bottom=torch.tensor([0,0,0,1.],device=device,dtype=dtype).expand(B,1,4)
        zero=torch.zeros(B,3,1,device=device,dtype=dtype)
        for i,joint in enumerate(self.edges):
            parent=links[joint["parent"]]
            origin=self.origin_matrices[i].to(device=device,dtype=dtype)
            pose=parent@origin
            if joint["typ"]=="revolute":
                theta=angles[:,self.joint_index[joint["name"]]]
                K=self.axis_skews[i].to(device=device,dtype=dtype)
                s=torch.sin(theta)[:,None,None]
                c=(1-torch.cos(theta))[:,None,None]
                A=ident+s*K+c*(K@K)
                F=torch.cat((torch.cat((A,zero),dim=-1),bottom),dim=-2)
                pose=pose@F
            links[joint["child"]]=pose
        pts={k:links[v][:,:3,3] for k,v in BODY.items()}
        pts["pelvis_center"]=(pts["left_hip"]+pts["right_hip"])*.5
        pts["shoulder_center"]=(pts["left_shoulder"]+pts["right_shoulder"])*.5
        vec={}
        for name,(a,b) in SEGMENT_ENDPOINTS.items():
            diff=pts[b]-pts[a]
            vec[name]=diff/diff.norm(dim=-1,keepdim=True).clamp_min(1e-9)
        rel=torch.stack([(vec[a]*vec[b]).sum(dim=-1).clamp(-1,1) for a,b in PAIRS],dim=-1)
        if with_links:return rel,links
        return rel

def load_robot_source(report_path):
    import json
    from scipy.spatial.transform import Slerp
    report=json.loads(Path(report_path).read_text())
    if report["root_fixed"] or report["post_initialization_state_writes"]:
        raise ValueError("Robot source was fixed-root or state rewritten")
    contact=float(report["records"][0]["contact_time_s"])
    rows=report["kinematics"][0]
    ts=np.asarray([r["time_s"] for r in rows],float)
    if len(ts)!=700 or not np.allclose(np.diff(ts),.02,atol=1e-5):
        raise ValueError("Full 700-control-step robot source needed")
    q=np.asarray([r["joint_pos"] for r in rows],float)
    root=np.asarray([r["root_pose"] for r in rows],float)
    if q.shape!=(700,29) or root.shape!=(700,7):raise ValueError("Bad robot source")
    if not np.isfinite(q).all() or not np.isfinite(root).all():raise ValueError("Bad source")
    return dict(q=q,root=root,times=ts,contact_time=contact,
                joint_names=report["joint_names"],case=report["trials"][0]["name"],
                source_sha256=None)

def interpolate_source(robot,samples):
    from scipy.spatial.transform import Slerp
    t=np.asarray(robot["times"],float)
    q=np.stack([np.interp(samples,t,robot["q"][:,i]) for i in range(29)],-1)
    pos=np.stack([np.interp(samples,t,robot["root"][:,i]) for i in range(3)],-1)
    raw=robot["root"][:,3:]
    xyzw=raw[:,[1,2,3,0]]
    rotations=Slerp(t,Rotation.from_quat(xyzw))(samples).as_quat()
    root=np.concatenate([pos,rotations[:,[3,0,1,2]]],axis=1)
    return root,q

def build_human_relative_target(robot_cosines,human,scale=.14):
    if robot_cosines.shape!=human.shape or robot_cosines.shape!=(20,11):
        raise ValueError("Human and robot must match exactly 20x11 feature contract")
    if not np.isfinite(human).all() or not np.isfinite(robot_cosines).all():
        raise ValueError("Nonfinite human or robot relations")
    # Only relative changes around source's 8th-frame observation; not absolute
    # human skeleton coordinates. No force/impact meaning is inferred.
    signal=human-human[7:8]
    signal=np.clip(signal,-.30,.30)*scale
    raw=np.clip(robot_cosines+signal,-1,1)
    return raw,signal

def soft_gate(times,contact,start_delay=.50,precontact_gap=.04):
    t=np.asarray(times,float)
    begin=contact-start_delay
    stop=contact-precontact_gap
    ramp=np.clip((t-begin)/.12,0,1)
    ramp=ramp*ramp*(3-2*ramp)
    release=np.clip((stop-t)/.12,0,1)
    release=release*release*(3-2*release)
    return ramp*release

def project_trajectory(fk,root,qbaseline,target,source_times,weights,
                       iterations=120,max_rad=.012,lr=.1,
                       foot_plant_weight=10.,smooth_weight=12.,joint_weight=6.):
    if not (len(root)==len(qbaseline)==len(target)==len(source_times)==len(weights)==20):
        raise ValueError("Exactly 20 in-register near-strike source states required")
    dev=torch.device("cpu")
    X=torch.tensor(root,dtype=torch.float64,device=dev)
    Q=torch.tensor(qbaseline,dtype=torch.float64,device=dev)
    T=torch.tensor(target,dtype=torch.float64,device=dev)
    mask=torch.tensor(weights,dtype=torch.float64,device=dev)
    if not bool(torch.isfinite(T).all()):raise ValueError("Target nonfinite")
    with torch.no_grad():
        baseline,bodies=fk(X,Q,with_links=True)
        left=bodies["left_ankle_roll_link"][:,:3,3].detach()
        right=bodies["right_ankle_roll_link"][:,:3,3].detach()
    z=torch.nn.Parameter(torch.zeros_like(Q))
    optim=torch.optim.Adam([z],lr=lr)
    vals=[]
    for i in range(iterations):
        d=max_rad*torch.tanh(z)*mask[:,None]
        pred,bodies=fk(X,Q+d,with_links=True)
        r=(pred-T)
        # All 11 channels matter; reduced penalty after/away from intended
        # precontact event. Note human time alignment remains unverified.
        coord=((r.square().mean(dim=-1)*mask).sum()/mask.sum().clamp_min(1))
        foot=bodies["left_ankle_roll_link"][:,:3,3]
        rightfoot=bodies["right_ankle_roll_link"][:,:3,3]
        foot_loss=(foot-left).square().sum(dim=-1).mean()
        swing_loss=(rightfoot-right).square().sum(dim=-1).mean()
        joint_loss=d.square().mean()
        temporal=(d[1:]-d[:-1]).square().mean()
        acceler=(d[2:]-2*d[1:-1]+d[:-2]).square().mean()
        loss=coord+foot_plant_weight*foot_loss+foot_plant_weight*.5*swing_loss+joint_weight*joint_loss+smooth_weight*(temporal+acceler)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z],max_norm=2.0)
        optim.step()
        if i in (0,iterations-1):
            vals.append(dict(iteration=i+1,loss=float(loss.detach()),coord=float(coord.detach()),foot=float(foot_loss.detach())))
    with torch.no_grad():
        d=max_rad*torch.tanh(z)*mask[:,None]
        pred,links=fk(X,Q+d,with_links=True)
        err_before=float((((baseline-T).square().mean(dim=-1))*mask).sum()/mask.sum())
        err_after=float((((pred-T).square().mean(dim=-1))*mask).sum()/mask.sum())
        delta=d.detach().numpy()
        original_left=left.numpy()
        generated_left=links["left_ankle_roll_link"][:,:3,3].detach().numpy()
        original_right=right.numpy()
        generated_right=links["right_ankle_roll_link"][:,:3,3].detach().numpy()
        constraints=dict(
            coord_mse_baseline=err_before,
            coord_mse_generated=err_after,
            foot_support_max_displacement_m=float(np.linalg.norm(generated_left-original_left,axis=-1).max()),
            foot_swing_max_displacement_m=float(np.linalg.norm(generated_right-original_right,axis=-1).max()),
            joint_max_delta_rad=float(np.abs(delta).max()),
            discrete_25hz_max_delta_rad=float(np.abs(np.diff(delta,axis=0)).max()),
            mean_abs_delta_rad=float(np.mean(abs(delta))),
            learned_nonzero_joints=int((np.max(abs(delta),axis=0)>1e-6).sum()),
            relative_improvement=(err_before-err_after)/max(1e-9,err_before),
            optimizer_snapshots=vals)
    return delta,constraints
