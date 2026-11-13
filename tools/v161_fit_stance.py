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
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path,required=True);cli.add_argument('--source',type=Path,required=True)
    a=cli.parse_args();cfg=json.loads(base.PROTOCOL.read_text());schema=json.loads((base.DEFAULT_ASSETS/base.SCHEMA).read_text())
    with base.cpu_policy(base.DEFAULT_ASSETS,cfg) as (meta,act):
        scene,system,robot,ball,drives,order,*_=base.make_plant(base.DEFAULT_ASSETS,cfg,meta,schema)
    a.output.mkdir(exist_ok=False,parents=True)
    with np.load(a.source/'rollout.npz') as tr:
        k=round(4.8*200)-1;xroot=tr['root_pose_wxyz'][k];rotation=Rotation.from_quat(xroot[[4,5,6,3]]).as_euler('xyz')
        ref=state(xroot[:3],xroot[3:],tr['root_velocity_w'][k],tr['joint_q_rad'][k],tr['joint_dq_rad_s'][k])
        u=tr['policy_commands_time_frame_action_target'][k//4,31:]
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
    Q=np.diag(np.r_[[5,45,45,.1,2,2,.3,.9,.9,.1],np.ones(29)*.6,np.ones(29)*.008]);R=np.eye(29)*.35;P=Q.copy()
    for _ in range(160):
        K=np.linalg.solve(R+B.T@P@B,B.T@P@A);M=A-B@K;P=Q+M.T@P@M+K.T@R@K
    np.savez(a.output/'lqr.npz',gain=K,indexes=idx,A=A,B=B)
    print('max feedback',abs(K).max(),'max closed eigenvalue',max(abs(np.linalg.eigvals(M))))
if __name__=='__main__':main()
