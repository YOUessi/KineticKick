#!/usr/bin/env python3
"""Candidate standing start into a finite running reference; no dynamics claim."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation
from running_reference_geometry import ReferenceGeometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acceleration-seconds',type=float,default=1.2)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    if not .8<=a.acceleration_seconds<=2.:p.error('Acceleration is bounded to0.8..2seconds')
    root=Path(__file__).resolve().parents[1];geom=ReferenceGeometry(root);src=np.load(a.source)
    if np.linalg.norm(src['body_lin_vel_w'][-1,0])>.01:raise ValueError('Source must end in a stationary standing hold')
    dt=.02;hold=20;frames=round(a.acceleration_seconds/dt);T=frames*dt;t=np.arange(frames)*dt
    q0=src['joint_pos'][-1].astype(float);q1=src['joint_pos'][0].astype(float)
    p1=src['body_pos_w'][0,0].astype(float);v1=src['body_lin_vel_w'][0,0].astype(float)
    p0=p1-v1*T/2;p0[2]=src['body_pos_w'][-1,0,2]
    # Large incoming RUN velocities make a long unconstrained cubic overshoot
    # joint limits and fold the foot into the shin. Bound endpoint tangents
    # to the monotone Hermite range; disclose the remaining velocity seam.
    secant=(q1-q0)/T
    end_dq=np.sign(secant)*np.clip(src['joint_vel'][0]*np.sign(secant),0.,3*np.abs(secant))
    q=CubicHermiteSpline([0,T],np.stack([q0,q1]),np.stack([np.zeros(29),end_dq]))(t)
    pos=CubicHermiteSpline([0,T],np.stack([p0,p1]),np.stack([np.zeros(3),v1]))(t)
    r0=Rotation.from_quat(src['body_quat_w'][-1,0,[1,2,3,0]]);r1=Rotation.from_quat(src['body_quat_w'][0,0,[1,2,3,0]])
    u=t/T;rot=Rotation.from_rotvec((3*u*u-2*u*u*u)[:,None]*(r1*r0.inv()).as_rotvec())*r0
    q=np.concatenate([np.tile(q0,(hold,1)),q]);pos=np.concatenate([np.tile(p0,(hold,1)),pos])
    quat=np.concatenate([np.tile(r0.as_quat()[[3,0,1,2]],(hold,1)),rot.as_quat()[:,[3,0,1,2]]])
    audit,bottom=geom.check(q,pos,quat)
    if audit['nonadjacent_capsule_overlap_frames']:raise ValueError('Initial blend self-intersects: '+json.dumps(audit['nonadjacent_capsule_overlap_examples']))
    pos[:,2]+=np.maximum(0.,.001-bottom)
    bp=[];bq=[]
    for qi,pi,ri in zip(q,pos,quat):
        fk=geom.body_poses(qi,pi,ri);bp.append([fk[n][0] for n in geom.body_names]);bq.append([fk[n][1].as_quat()[[3,0,1,2]] for n in geom.body_names])
    bp=np.array(bp);bq=np.array(bq);av=np.zeros_like(bp)
    for j in range(len(geom.body_names)):
        rr=Rotation.from_quat(bq[:,j,[1,2,3,0]])
        av[1:-1,j]=(rr[2:]*rr[:-2].inv()).as_rotvec()/(2*dt);av[-1,j]=(rr[-1]*rr[-2].inv()).as_rotvec()/dt
    prefix=dict(joint_pos=q,joint_vel=np.gradient(q,dt,axis=0,edge_order=2),body_pos_w=bp,body_quat_w=bq,
        body_lin_vel_w=np.gradient(bp,dt,axis=0,edge_order=2),body_ang_vel_w=av)
    # Constant standing poses have exactly zero velocities; eliminate gradient roundoff.
    for key in ('joint_vel','body_lin_vel_w','body_ang_vel_w'):prefix[key][:hold-1]=0.
    arrays={k:np.concatenate([prefix[k].astype('f4'),src[k]]) if k in prefix else src[k] for k in src.files}
    assert np.max(abs(arrays['joint_vel'][0]))==0 and np.max(abs(arrays['body_lin_vel_w'][0]))==0 and np.max(abs(arrays['body_ang_vel_w'][0]))==0
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:np.savez_compressed(f,**arrays)
    meta=dict(status='kinematic_candidate_only',source=str(a.source),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),stationary_start=True,standing_hold_s=.4,acceleration_s=T,
        duration_s=(len(arrays['joint_pos'])-1)*dt,prefix_frames=len(q),source_preserved_after_prefix=True,
        max_joint_velocity_seam_rad_s=float(np.max(np.abs(end_dq-src['joint_vel'][0]))),
        min_nonadjacent_capsule_gap_m=audit['min_nonadjacent_capsule_gap_m'],warning='No standing-start dynamics acceptance; no shooting handoff')
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))


if __name__=='__main__':main()
