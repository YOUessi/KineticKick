"""Teacher-driven physical development search. Full closed-loop confirmation is mandatory."""
import argparse
import gzip
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import sapien
import v160_tracking_cpu as base
from v161_strike_search import LEG, residual_weights
from free_shooting_metrics import paid_contact_patch

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--initial',type=Path)
    cli.add_argument('--seed',type=int,default=161041)
    cli.add_argument('--generations',type=int,default=12)
    cli.add_argument('--population',type=int,default=48)
    a=cli.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(base.PROTOCOL.read_text());assets=base.DEFAULT_ASSETS
    schema=json.loads((assets/base.SCHEMA).read_text())
    with base.cpu_policy(assets,cfg) as (meta,act):
        scene,system,robot,ball,drives,order,limits,gains,links,ids=base.make_plant(assets,cfg,meta,schema)
    source=assets/cfg['clip']
    with np.load(source) as ref:
        robot.set_root_pose(sapien.Pose(ref['body_pos_w'][0,0],ref['body_quat_w'][0,0]))
        robot.set_root_linear_velocity(ref['body_lin_vel_w'][0,0])
        robot.set_root_angular_velocity(ref['body_ang_vel_w'][0,0])
        robot.set_qpos(ref['joint_pos'][0,np.argsort(order)])
        robot.set_qvel(ref['joint_vel'][0,np.argsort(order)])
    original=base.REPO/'experiments/v160_tracking/official_001_cpu/rollout.npz'
    with np.load(original) as tr:
        targets=tr['policy_commands_time_frame_action_target'][:,31:]
    onset=103;end=155
    for target in targets[:onset]:
        for joint,val in zip(drives,target): joint.set_drive_target(float(val))
        for _ in range(4):scene.step()
    snapshot=system.pack()
    ix=[schema['joint_names'].index('right_'+j+'_joint') for j in LEG]
    rs=links['right_ankle_roll_link']
    weights=np.stack([residual_weights(t*.02) for t in range(onset,end)])
    def trial(coeff):
        system.unpack(snapshot)
        j_all=j_good=0.; aligned=0.;gap=1.;nonfoot=0.;ct=None;foot0=None;advance=0.;reladvance=0.;follow_v=0.;prev_p=None;lowest=2.;limit_ex=0.
        targets_now=targets[onset:end].copy()
        targets_now[:,ix]+=weights@coeff.reshape(2,6)
        for k,target in enumerate(targets_now):
            for joint,val in zip(drives,target):joint.set_drive_target(float(val))
            for d in range(4):
                tick=(k+onset)*4+d+1
                fp=rs.entity.pose; fr=Rotation.from_quat(fp.q[[1,2,3,0]]);pre_ball=ball.entity.pose.p
                face=fp.p+fr.apply([.075,0,-.015]);axis=fr.apply([0,0,1])
                dist=pre_ball-face;norm=np.linalg.norm(dist)
                if ct is None and norm<gap:
                    gap=norm;near_alignment=float(np.dot(axis,dist/max(1e-6,norm)))
                scene.step()
                q=robot.qpos[order];limit_ex=max(limit_ex,np.max(limits[:,0]-q),np.max(q-limits[:,1]))
                lowest=min(lowest,robot.root_pose.p[2])
                for c in system.get_contacts():
                    names=[b.entity.name for b in c.bodies]
                    if set(names)=={'right_ankle_roll_link','SoccerBall'}:
                        s=1 if names[0]=='SoccerBall' else -1
                        for p in c.points:
                            J=float(np.linalg.norm(p.impulse))
                            if J<1e-7:continue
                            out=np.array(p.normal)*s; loc=fr.inv().apply(p.position-p.separation*out-fp.p); n=fr.inv().apply(out)
                            j_all+=J; aligned+=J*n[2]
                            if paid_contact_patch(loc,n)=='instep': j_good+=J
                            if ct is None:ct=tick;foot0=(face.copy(),robot.root_pose.p.copy())
                    elif any(n not in ('left_ankle_roll_link','right_ankle_roll_link','ground','SoccerBall') for n in names):
                        if c.points:
                            nonfoot=max(nonfoot,np.linalg.norm(sum(np.array(p.impulse) for p in c.points))/.005)
                if ct is not None and tick-ct<=50:
                    advance=max(advance,foot0[0][1]-face[1])
                    reladvance=max(reladvance,foot0[0][1]-face[1]+robot.root_pose.p[1]-foot0[1][1])
                    if tick-ct==24 and prev_p is not None:follow_v=(prev_p-face[1])/.005
                prev_p=face[1]
        v=ball.linear_velocity
        share=j_good/max(1e-9,j_all);alignment=aligned/max(1e-9,j_all)
        speed=float(-v[1]);accuracy=abs(v[0]/max(.3,speed))
        merit=2*share+2*alignment+min(1.,speed/3.8)+.4*min(1,advance/.22)+.4*min(1,reladvance/.15)+.3*min(1,max(0,follow_v)/.6)-min(1.5,accuracy*2)-min(1.5,nonfoot/40)-10*max(0,.65-lowest)-100*limit_ex
        if ct is None:merit=-3-4*gap+near_alignment-min(1.5,nonfoot/40)
        return dict(merit=float(merit),instep=share,alignment=alignment,out_forward=speed,lateral=float(v[0]),nonfoot=float(nonfoot),min_z=float(lowest),advance=float(advance),relative=float(reladvance),v120=float(follow_v),excess=float(limit_ex),contact=ct)
    rng=np.random.default_rng(a.seed)
    mu=np.array([-.15,-.04,.65,.53,.78,0.,-.55,0.,.35,-.22,.6,0.])
    sd=np.array([.42,.20,.60,.45,.50,.21,.40,.19,.40,.38,.45,.15])
    if a.initial:mu=np.array(json.loads(a.initial.read_text())['coefficients']).reshape(-1)
    best=-np.inf
    with (a.output/'rollouts.jsonl').open('x',buffering=1) as out:
        for g in range(a.generations):
            xs=rng.normal(mu,sd,size=(a.population,12)); xs[0]=mu
            if g==0:xs[1]=0
            group=[]
            for i,x in enumerate(xs):
                r=trial(x);row={'generation':g,'candidate':i,'coefficients':x.tolist(),**r}
                out.write(json.dumps(row)+'\n');group.append(row)
                if r['merit']>best:
                    best=r['merit'];(a.output/'best.json').write_text(json.dumps({'module':'v161_strike_search',**row},indent=2)+'\n')
            top=sorted(group,key=lambda z:z['merit'],reverse=True)[:max(4,len(group)//5)]
            elite=np.array([e['coefficients'] for e in top]);mu=.8*elite.mean(0)+.2*mu;sd=np.maximum(.055,.75*elite.std(0)+.25*sd)
            print(g,top[0]['merit'],top[0]['instep'],top[0]['out_forward'],flush=True)
if __name__=='__main__':main()
