"""Differentiate the 20 ms double-support dynamics, then fit a finite-horizon LQR."""
import argparse
import json
import numpy as np
import sapien
from pathlib import Path
from scipy.spatial.transform import Rotation
from v161_recovery_lqr import state
import v160_tracking_cpu as base

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path,required=True);cli.add_argument('--source',type=Path,required=True);cli.add_argument('--supported',action='store_true')
    a=cli.parse_args();cfg=json.loads(base.PROTOCOL.read_text());schema=json.loads((base.DEFAULT_ASSETS/base.SCHEMA).read_text())
    with base.cpu_policy(base.DEFAULT_ASSETS,cfg) as (meta,act):
        scene,system,robot,ball,drives,order,*_=base.make_plant(base.DEFAULT_ASSETS,cfg,meta,schema)
    a.output.mkdir(exist_ok=False,parents=True)
    with np.load(a.source/'rollout.npz') as tr:
        k=round(4.8*200)-1;xroot=tr['root_pose_wxyz'][k];rotation=Rotation.from_quat(xroot[[4,5,6,3]]).as_euler('xyz')
        ref=state(xroot[:3],xroot[3:],tr['root_velocity_w'][k],tr['joint_q_rad'][k],tr['joint_dq_rad_s'][k])
        u=tr['policy_commands_time_frame_action_target'][k//4,31:]
        commands=tr['policy_commands_time_frame_action_target'][:k//4+1,31:].copy()
    inv=np.argsort(order)
    def set_state(x):
        quat=Rotation.from_euler('xyz',x[3:6]).as_quat()[[3,0,1,2]]
        robot.set_root_pose(sapien.Pose(x[:3],quat));robot.set_root_linear_velocity(x[6:9]);robot.set_root_angular_velocity(x[9:12]);robot.set_qpos(x[12:41][inv]);robot.set_qvel(x[41:][inv])
    def step(x,u):
        set_state(x)
        for joint,val in zip(drives,u):joint.set_drive_target(float(val))
        for _ in range(4):scene.step()
        root=robot.root_pose
        return state(root.p,root.q,np.r_[robot.root_linear_velocity,robot.root_angular_velocity],robot.qpos[order],robot.qvel[order])
    idx=np.r_[np.arange(2,12),np.arange(12,70)]
    # Translational invariance in XY makes absolute horizontal positions neutral.
    N=len(idx);A=np.empty((N,N));B=np.empty((N,29));set_state(ref);snap=system.pack()
    def transition(x,v):
        system.unpack(snap)
        return step(x,v)[idx]
    for i,j in enumerate(idx):
        h=.0001 if j<6 else .001
        v=np.eye(70)[j]*h;A[:,i]=(transition(ref+v,u)-transition(ref-v,u))/(2*h)
    for i in range(29):
        du=.002*np.eye(29)[i];B[:,i]=(transition(ref,u+du)-transition(ref,u-du))/.004
    extra={}
    if a.supported:
        # Build the local state transition only in the contact-consistent perturbation
        # subspace obtained through REAL drive commands from the preceding trajectory.
        # The initial free root and q/dq are written once in this identification scene.
        with np.load(base.DEFAULT_ASSETS/cfg['clip']) as data:
            robot.set_root_pose(sapien.Pose(data['body_pos_w'][0,0],data['body_quat_w'][0,0]))
            robot.set_root_linear_velocity(data['body_lin_vel_w'][0,0]);robot.set_root_angular_velocity(data['body_ang_vel_w'][0,0])
            robot.set_qpos(data['joint_pos'][0,inv]);robot.set_qvel(data['joint_vel'][0,inv])
        def advance(target):
            for j,v in zip(drives,target):j.set_drive_target(float(v))
            for _ in range(4):scene.step()
            pose=robot.root_pose
            return state(pose.p,pose.q,robot.root_linear_velocity.tolist()+robot.root_angular_velocity.tolist(),robot.qpos[order],robot.qvel[order])[idx]
        for target in commands:advance(target)
        snapshot=system.pack();rng=np.random.default_rng(161801)
        X=[];Y=[];T=[]
        for _ in range(340):
            system.unpack(snapshot);old=advance(u)
            for j in range(9):
                delta=rng.normal(0,.009 if j<3 else .014,29)
                new=advance(u+delta)
                X.append(old);T.append(delta);Y.append(new);old=new
        X=np.array(X);T=np.array(T);Y=np.array(Y)
        Z=np.c_[X-X.mean(0),T-T.mean(0)]
        st=np.maximum(Z.std(0),1e-5)
        norm=Z/st
        linear=np.linalg.solve(norm.T@norm+.0001*len(Z)*np.eye(Z.shape[1]),norm.T@(Y-Y.mean(0))).T/st
        A=linear[:,:N];B=linear[:,N:]; extra={'response_fit_rms':np.sqrt(np.mean(((Y-Y.mean(0))-Z@linear.T)**2,axis=0))}
    Q=np.diag(np.r_[[5,45,45,.1,2,2,.3,.9,.9,.1],np.ones(29)*.6,np.ones(29)*.008]);R=np.eye(29)*.35;P=Q.copy()
    for _ in range(160):
        K=np.linalg.solve(R+B.T@P@B,B.T@P@A);M=A-B@K;P=Q+M.T@P@M+K.T@R@K
    np.savez(a.output/'lqr.npz',gain=K,indexes=idx,A=A,B=B,**extra)
    print('max feedback',abs(K).max(),'max closed eigenvalue',max(abs(np.linalg.eigvals(M))))
if __name__=='__main__':main()
