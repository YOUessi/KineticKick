"""One pre-registered whole-clip right-arm clearance correction on CMU train 10_02."""
import argparse
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import osqp
import pinocchio as pin
from scipy import sparse
from scipy.spatial.transform import Rotation

from v162_human_support import ROOT, difference, source


def capsule(urdf, name):
    shapes=urdf.findall(f"link[@name='{name}']/collision")
    if len(shapes)!=1 or shapes[0].find('geometry/cylinder') is None:
        raise ValueError(f'Expected a single original capsule on {name}')
    origin=shapes[0].find('origin')
    center=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
    rot=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
    primitive=shapes[0].find('geometry/cylinder')
    offset=.5*float(primitive.get('length'))*rot[:,2]
    return np.stack([center-offset, center+offset]),float(primitive.get('radius'))


def witnesses(first, second):
    p, end=first
    k, far=second
    u,v,w=end-p,far-k,p-k
    a,b,c,d,e=u@u,u@v,v@v,u@w,v@w
    det=a*c-b*b
    s=0. if abs(det)<1e-14 else np.clip((b*e-c*d)/det,0.,1.)
    t=(b*s+e)/c
    if t<0:
        t=0.
        s=np.clip(-d/a,0.,1.)
    elif t>1:
        t=1.
        s=np.clip((b-d)/a,0.,1.)
    return p+s*u, k+t*v


class ArmGap:
    def __init__(self, robot, urdf, cfg):
        self.robot=robot
        self.idx=np.array([robot.model.idx_qs[robot.model.getJointId(name)] for name in cfg['arm_joints']])
        self.link=[robot.index[name] for name in cfg['links']]
        xml=ET.parse(urdf)
        self.shapes=[capsule(xml,name) for name in cfg['links']]

    def at(self, q):
        robot=self.robot
        feat,fj=robot.features(q)
        frames=[robot.ids[i] for i in self.link]
        poses=[robot.data.oMf[f] for f in frames]
        seg=[pose.translation+end@pose.rotation.T for pose,(end,r) in zip(poses,self.shapes)]
        pw,ph=witnesses(*seg)
        points=[pw,ph]
        distance=np.linalg.norm(pw-ph)
        if distance<1e-8:
            raise ValueError('Coincident capsule centerlines')
        j=[]
        for f,c,point in zip(frames,poses,points):
            jac=pin.getFrameJacobian(robot.model,robot.data,f,pin.LOCAL_WORLD_ALIGNED)[:,self.idx]
            j.append(jac[:3]+np.cross(jac[3:].T,point-c.translation).T)
        norm=(pw-ph)/distance
        return distance-self.shapes[0][1]-self.shapes[1][1],norm@(j[0]-j[1]),feat,fj[:,:,self.idx]


def run(cfg, before, asset, out):
    if source.sha(before)!=cfg['input_sha256']:
        raise ValueError('Training clip changed')
    urdf=asset/cfg['urdf']
    schema=json.loads((asset/cfg['schema']).read_text())
    if source.sha(urdf)!=cfg['urdf_sha256']:
        raise ValueError('Original geometry changed')
    robot=source.Robot(urdf,schema)
    arm=ArmGap(robot,urdf,cfg)
    with np.load(before,allow_pickle=False) as f:
        arr={k:f[k].copy() for k in f.files}
    if arr['source_clip'].item()!='10_02' or arr['split'].item()!='train' or arr['joint_names'].tolist()!=schema['joint_names']:
        raise ValueError('Only specified training clip/order is eligible')
    q0=arr['joint_pos'][:,np.argsort(robot.runtime_indices)].astype(float)
    q=q0.copy()
    ids=arm.idx
    human=arr['human_segment_direction_root'].astype(float)
    human/=np.linalg.norm(human,axis=-1,keepdims=True)
    n,k=len(q),len(ids)
    eye=sparse.eye(n*k,format='csc')
    delta=sparse.kron(difference(n,1),sparse.eye(k),format='csc')
    d2=sparse.kron(difference(n,2),sparse.eye(k),format='csc')
    reg=cfg['diff1_weight']**2*(delta.T@delta)+cfg['diff2_weight']**2*(d2.T@d2)
    vmax=np.tile(cfg['velocity_fraction']*robot.vlim[ids]/50.,n-1)
    orig_value=np.array([arm.at(x)[0] for x in q])
    trace=[]
    weights=np.sqrt(np.asarray(cfg['direction_weights']))[:,None]
    for ii in range(cfg['iterations']):
        rows=[arm.at(x) for x in q]
        gaps,jac,features,fjac=[np.array(c) for c in zip(*rows)]
        matrices=(fjac*weights[None,:,:,None]).reshape(n,54,k)
        error=((human-features)*weights).reshape(n,54)
        h=matrices.transpose(0,2,1)@matrices
        h+=np.eye(k)[None]*(cfg['pose_weight']**2+cfg['damping'])
        dr=(q-q0)[:,ids].ravel()
        g=-np.einsum('nrk,nr->nk',matrices,error).ravel()+cfg['pose_weight']**2*dr+reg@dr
        p=sparse.block_diag([sparse.csc_matrix(x) for x in h],format='csc')+reg
        j=sparse.block_diag([sparse.csc_matrix(x[None]) for x in jac],format='csc')
        a=sparse.vstack([eye,j,delta],format='csc')
        vel=delta@q[:,ids].ravel()
        low=np.r_[np.maximum((robot.low[ids]-q[:,ids]).ravel(),-cfg['max_step_rad']),
                  cfg['clearance_m']-gaps,-vmax-vel]
        high=np.r_[np.minimum((robot.high[ids]-q[:,ids]).ravel(),cfg['max_step_rad']),
                   np.full(n,np.inf),vmax-vel]
        qp=osqp.OSQP()
        qp.setup(P=sparse.triu(p).tocsc(),q=g,A=a,l=low,u=high,verbose=False,eps_abs=1e-6,eps_rel=1e-6,
                 max_iter=12000,polish=True)
        solution=qp.solve()
        if solution.info.status not in ('solved','solved inaccurate') or solution.x is None:
            raise RuntimeError(f'SQP {ii}: {solution.info.status}')
        residual=max(0.,(low-a@solution.x).max(),(a@solution.x-high).max())
        if residual>2e-5:
            raise ValueError(f'Linear inequality residual {residual}')
        q[:,ids]+=solution.x.reshape(n,k)
        trace.append(dict(iterations=solution.info.iter, status=solution.info.status,linear_residual=float(residual)))
    post=[arm.at(x) for x in q]
    new_gap=np.array([r[0] for r in post])
    direction=np.array([r[2] for r in post])
    e=np.degrees(np.arccos(np.clip((direction*human).sum(-1),-1,1)))
    posrot=[robot.poses(x) for x in q]
    root=arr['root_pos_w']
    orient=Rotation.from_quat(arr['root_quat_wxyz'][:,[1,2,3,0]]).as_matrix()
    world=root[:,None]+np.einsum('tij,tkj->tki',orient,np.stack([y[0] for y in posrot]))
    rot=orient[:,None]@np.stack([y[1] for y in posrot])
    before_foot=arr['body_pos_w'][:,[robot.index[s+'_ankle_roll_link'] for s in ('left','right')]]
    after_foot=world[:,[robot.index[s+'_ankle_roll_link'] for s in ('left','right')]]
    foot_err=float(np.max(np.linalg.norm(before_foot-after_foot,axis=-1)))
    order=robot.runtime_indices
    kin_limit=max(np.maximum(robot.low-q,0.).max(),np.maximum(q-robot.high,0.).max())
    v_ratio=float((np.abs(np.diff(q,axis=0))*50./robot.vlim).max())
    if foot_err>1e-6 or kin_limit>1e-6 or v_ratio>1.+1e-6:
        raise ValueError('Unchanged foot geometry or original joint bounds violated')
    arr.update(joint_pos=q[:,order].astype('f4'),joint_vel=np.gradient(q[:,order],.02,axis=0,edge_order=2).astype('f4'),
        body_pos_w=world.astype('f4'),body_quat_w=source.angle_quaternions(rot).astype('f4'),
        body_lin_vel_w=np.gradient(world,.02,axis=0,edge_order=2).astype('f4'),
        body_ang_vel_w=source.world_angular_velocity(rot,.02).astype('f4'),ik_feature_error_deg=e.astype('f4'),
        wrist_ipsilateral_hip_capsule_clearance_m=new_gap.astype('f4'))
    safely=asset/cfg['safe_reference']
    if source.sha(safely)!=cfg['safe_reference_sha256']:
        raise ValueError('Safe comparison reference changed')
    with np.load(safely,allow_pickle=False) as f:
        safe_gap=np.array([arm.at(x)[0] for x in f['joint_pos'][:,np.argsort(order)]])
    summary={'target_train_source':str(before),'input_sha256':source.sha(before),'safe_original_PAID_reference':str(safely),
        'safe_original_PAID_gap_first_m':float(safe_gap[0]),'safe_original_PAID_gap_min_m':float(safe_gap.min()),
        'gap_first_before_after_m':[float(orig_value[0]),float(new_gap[0])],
        'gap_min_before_after_m':[float(orig_value.min()),float(new_gap.min())],
        'overlap_frame_count_before_after':[int((orig_value<0).sum()),int((new_gap<0).sum())],
        'min_clearance_enforced_all_frames_m':cfg['clearance_m'],'max_foot_position_delta_m':foot_err,
        'max_joint_delta_rad':float(np.abs(q-q0).max()),'mean_18_direction_error_after_deg':float(e.mean()),
        'reference_edge_velocity_ratio':v_ratio,'iterations':trace,
        'scope':'Single training clip, one identified wrist/hip pair, original 50 Hz reference poses. Dynamic tracking depends on released controller.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    with tempfile.TemporaryDirectory(prefix='arm-clip-',dir=out.parent) as tmp:
        tmp=Path(tmp);result=tmp/'10_02.npz';np.savez_compressed(result,**arr)
        summary['output_sha256']=source.sha(result);summary['code_sha256']=source.sha(Path(__file__))
        (tmp/'geometry.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        tmp.rename(out)
    return summary


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config',type=Path,default=ROOT/'configs/v162_human_armclear.json')
    cli.add_argument('--input',type=Path,default=ROOT/'experiments/v162_human/references/10_02.npz')
    cli.add_argument('--out',type=Path,default=ROOT/'experiments/v162_human/armclear')
    cli.add_argument('--asset-root',type=Path,default=Path('/home/you/projects/kinetickick'))
    args=cli.parse_args();print(json.dumps(run(json.loads(args.config.read_text()),args.input,args.asset_root,args.out),indent=2))
