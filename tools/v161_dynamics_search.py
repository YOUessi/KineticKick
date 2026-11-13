"""Teacher-driven physical development search. Full closed-loop confirmation is mandatory."""
import argparse
import gzip
import json
import os
import subprocess
import atexit
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
    cli.add_argument('--pitch-cap',type=float)
    cli.add_argument('--feasible',action='store_true',help='Select around the original joint, surface and shot thresholds')
    cli.add_argument('--spread',type=float,default=1.)
    cli.add_argument('--live',action='store_true',help='Recompute PAiD recurrent actions from each perturbed state')
    a=cli.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(base.PROTOCOL.read_text());assets=base.DEFAULT_ASSETS
    schema=json.loads((assets/base.SCHEMA).read_text())
    with base.cpu_policy(assets,cfg) as (meta,act):
        scene,system,robot,ball,drives,order,limits,gains,links,ids=base.make_plant(assets,cfg,meta,schema)
    live_actor=None
    if a.live:
        live_actor=subprocess.Popen([cfg['python_for_onnx'],'-u',str(Path(__file__).with_name('v161_policy_worker.py')),str(assets)],
            text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=1)
        atexit.register(live_actor.terminate)
        if json.loads(live_actor.stdout.readline())!=meta:
            raise ValueError('Mismatched checkpoint metadata')
        def request(msg):
            live_actor.stdin.write(json.dumps(msg)+'\n')
            live_actor.stdin.flush()
            return json.loads(live_actor.stdout.readline())
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
    if a.live:
        with np.load(source) as data: live_ref={k:data[k] for k in data}
        anchor=schema['body_names'].index(meta['anchor_body_name'])
        goal=np.array(cfg['goal_xyz_m'])
        with np.load(original) as tr:
            pre_actions=tr['policy_commands_time_frame_action_target'][:,2:31]
            pre_obs=tr['policy_observations']
    onset=103;end=180 if a.live else 155
    if a.live:
        for frame,obs in enumerate(pre_obs[:onset]):request([frame,obs.tolist()])
        request('save')
    for target in targets[:onset]:
        for joint,val in zip(drives,target): joint.set_drive_target(float(val))
        for _ in range(4):scene.step()
    snapshot=system.pack()
    ix=[schema['joint_names'].index('right_'+j+'_joint') for j in LEG]
    rs=links['right_ankle_roll_link']
    weights=np.stack([residual_weights(t*.02) for t in range(onset,end)])
    def trial(coeff):
        system.unpack(snapshot)
        if a.live: 
            request('restore')
            previous=pre_actions[onset-1]
        after_xy=None; began=False; down=0.; back=0.; was=False; late=False; pos_history=[]; last_foot=0.
        good_post=0.; outward_score=0.
        j_all=j_good=0.; aligned=0.;gap=1.;nonfoot=0.;ct=None;foot0=None;advance=0.;reladvance=0.;follow_v=0.;prev_p=None;lowest=2.;limit_ex=-np.inf
        targets_now=targets[onset:end].copy()
        targets_now[:,ix]+=weights@coeff.reshape(2,6)
        for k,target in enumerate(targets_now):
            if a.live:
                frame=onset+k
                obs=base.observation(robot.qpos[order],robot.qvel[order],robot.root_pose,robot.root_angular_velocity,ball.entity.pose.p,goal,previous,live_ref,frame,anchor,gains['default_joint_pos'])
                previous=np.clip(request([frame,obs.tolist()]),-100.,100.)
                target=gains['default_joint_pos']+gains['action_scale']*previous
                target[ix]+=weights[k]@coeff.reshape(2,6)
            if a.pitch_cap is not None and 2.12<=(onset+k)*.02<=2.30:
                target[ix[4]]=min(target[ix[4]],a.pitch_cap)
            for joint,val in zip(drives,target):joint.set_drive_target(float(val))
            for d in range(4):
                tick=(k+onset)*4+d+1
                fp=rs.entity.pose; fr=Rotation.from_quat(fp.q[[1,2,3,0]]);pre_ball=ball.entity.pose.p
                face=fp.p+fr.apply([.075,0,-.015]);axis=fr.apply([0,0,1])
                dist=pre_ball-face;norm=np.linalg.norm(dist)
                if ct is None and norm<gap:
                    gap=norm;near_alignment=float(np.dot(axis,dist/max(1e-6,norm)))
                loaded=False
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
                            fin=rs.entity.pose;postrot=Rotation.from_quat(fin.q[[1,2,3,0]])
                            if paid_contact_patch(postrot.inv().apply(p.position-p.separation*out-fin.p),postrot.inv().apply(out))=='instep':good_post+=J
                            if J/.005>2: loaded=True
                            if paid_contact_patch(loc,n)=='instep': j_good+=J
                            if ct is None:ct=tick;foot0=(face.copy(),robot.root_pose.p.copy())
                    elif any(n not in ('left_ankle_roll_link','right_ankle_roll_link','ground','SoccerBall') for n in names):
                        if c.points:
                            nonfoot=max(nonfoot,np.linalg.norm(sum(np.array(p.impulse) for p in c.points))/.005)
                if ct is not None and tick-ct<=50:
                    advance=max(advance,foot0[0][1]-face[1])
                    reladvance=max(reladvance,foot0[0][1]-face[1]+robot.root_pose.p[1]-foot0[1][1])
                    if tick-ct==24 and prev_p is not None:follow_v=(prev_p-face[1])/.005
                if ct is not None:
                    val=-face[1]
                    pos_history.append(val)
                    elapsed=(tick-ct)*.005
                    if .22<=elapsed<=.75 and len(pos_history)>4:back=max(back,(pos_history[-5]-val)/.02)
                    if loaded and elapsed>=.08:late=True
                if was and not loaded:after_xy=ball.linear_velocity[:2].copy()
                was=loaded
                prev_p=face[1]
        v=ball.linear_velocity
        share=j_good/max(1e-9,j_all);alignment=aligned/max(1e-9,j_all)
        speed=float(-v[1]);accuracy=abs(v[0]/max(.3,speed))
        merit=2*share+2*alignment+min(1.,speed/3.8)+.4*min(1,advance/.22)+.4*min(1,reladvance/.15)+.3*min(1,max(0,follow_v)/.6)-min(1.5,accuracy*2)-min(1.5,nonfoot/40)-10*max(0,.65-lowest)-100*max(0,limit_ex)
        if a.live and ct is not None:
            sx,sy=after_xy if after_xy is not None else v[:2]
            sh=min(share,good_post/max(1e-9,j_all))
            merit=2.5*sh+1.6*alignment+min(1.,-sy/3.75)+.5*min(1,advance/.235)+.5*min(1,reladvance/.16)+.45*min(1,max(0,follow_v)/.7)-min(2.,abs(sx/max(.25,-sy))*3)-min(2.,nonfoot/30)-max(0.,.64-lowest)*13-100*max(0,limit_ex)-.2*max(0,back-1.8)-.8*late
        if a.feasible and a.live and ct is not None:
            # Keep supplying a gradient below the direction-alignment classifier boundary.
            sh=min(share,good_post/max(1e-9,j_all))
            deficit=5*max(0,.975-sh)+1.3*max(0,3.8+sy)+4*max(0,abs(sx/max(.25,-sy))-.15)+.2*max(0,back-1.9)+.8*late
            deficit+=min(6.,nonfoot/14)+min(5.,500000*max(0.,limit_ex-5e-7))+0.4*max(0.,min(1.,(limit_ex+.015)/.015))+max(0,.68-lowest)*15
            deficit+=1.5*max(0,.23-advance)+2*max(0,.16-reladvance)+.4*max(0,.72-follow_v)
            merit=8-deficit+.25*alignment+.1*min(-sy,6)
        if ct is None:merit=-3-4*gap+near_alignment-min(1.5,nonfoot/40)
        return dict(backswing_pitch_target_cap=a.pitch_cap,merit=float(merit),instep=share,alignment=alignment,out_forward=speed,lateral=float(v[0]),nonfoot=float(nonfoot),min_z=float(lowest),advance=float(advance),relative=float(reladvance),v120=float(follow_v),excess=float(limit_ex),instep_post=good_post/max(1e-9,j_all),return_speed=float(back),late_contact=late,release_xy=None if after_xy is None else after_xy.tolist(),contact=ct)
    rng=np.random.default_rng(a.seed)
    mu=np.array([-.15,-.04,.65,.53,.78,0.,-.55,0.,.35,-.22,.6,0.])
    sd=np.array([.42,.20,.60,.45,.50,.21,.40,.19,.40,.38,.45,.15])
    if a.initial:mu=np.array(json.loads(a.initial.read_text())['coefficients']).reshape(-1)
    sd*=a.spread
    best=-np.inf
    if not a.live and a.initial: raise ValueError('Warm start reserved for live search')
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
