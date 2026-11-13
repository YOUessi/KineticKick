"""Closed-loop CPU physical development from a recurrent-and-plant checkpoint; only 14s raw runs qualify."""
import argparse
import atexit
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import sapien
from scipy.spatial.transform import Rotation

import v160_tracking_cpu as base
from free_shooting_metrics import paid_contact_patch
from v161_strike_search import LEG, residual_weights
from v162_vertical import correction, layout, extra_axes
from vertical_instep import FOOT_CORNERS

CORNERS=np.array(FOOT_CORNERS)
V161=base.REPO/'configs/v161_final.json'
DT=.005


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--initial',type=Path);p.add_argument('--generations',type=int,default=6)
    p.add_argument('--population',type=int,default=18);p.add_argument('--spread',type=float,default=1.)
    p.add_argument('--profile',choices=['feasible','pose','bridge','strict'],default='feasible')
    p.add_argument('--speed-wave',action='store_true')
    p.add_argument('--whole-spine',action='store_true')
    p.add_argument('--progress-deg',type=float,default=0.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(base.PROTOCOL.read_text());original=json.loads(V161.read_text())
    assets=base.DEFAULT_ASSETS;schema=json.loads((assets/base.SCHEMA).read_text());names=schema['joint_names']
    with np.load(assets/cfg['clip']) as d:ref={s:d[s] for s in d.files}
    with base.cpu_policy(assets,cfg) as (meta,_):
        scene,system,robot,ball,drives,order,limits,gains,links,ids=base.make_plant(assets,cfg,meta,schema)
    actor=subprocess.Popen([cfg['python_for_onnx'],'-u',str(Path(__file__).with_name('v161_policy_worker.py')),str(assets)],
                           text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=1,
                           env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    atexit.register(actor.terminate)
    if json.loads(actor.stdout.readline())!=meta:raise ValueError('Mismatched released policy')
    def request(msg):
        actor.stdin.write(json.dumps(msg)+'\n');actor.stdin.flush()
        return json.loads(actor.stdout.readline())
    order_j=np.argsort(order)
    robot.set_root_pose(sapien.Pose(ref['body_pos_w'][0,0],ref['body_quat_w'][0,0]))
    robot.set_root_linear_velocity(ref['body_lin_vel_w'][0,0]);robot.set_root_angular_velocity(ref['body_ang_vel_w'][0,0])
    robot.set_qpos(ref['joint_pos'][0,order_j]);robot.set_qvel(ref['joint_vel'][0,order_j])
    anchor=schema['body_names'].index(meta['anchor_body_name']);goal=np.array(cfg['goal_xyz_m'])
    direction=goal-np.asarray(cfg['ball_initial_xyz_m']);direction[2]=0.;direction/=np.linalg.norm(direction)
    start,end=90,168; prev=np.zeros(29); leg_idx=[names.index('right_'+n+'_joint') for n in LEG]
    idx=layout(names);foot=links['right_ankle_roll_link']; shoulder=names.index('left_shoulder_roll_joint')
    wrist=links['left_wrist_yaw_link'];hip=links['left_hip_roll_link']
    def act(tick,prev):
        obs=base.observation(robot.qpos[order],robot.qvel[order],robot.root_pose,robot.root_angular_velocity,
                             ball.entity.pose.p,goal,prev,ref,tick,anchor,gains['default_joint_pos'])
        action=np.clip(request([tick,obs.tolist()]),-100.,100.)
        return action,gains['default_joint_pos']+gains['action_scale']*action
    for k in range(start):
        prev,target=act(k,prev)
        for drive,val in zip(drives,target):drive.set_drive_target(float(val))
        for _ in range(4):scene.step()
    snapshot=system.pack();request('save');seed_prev=prev.copy()
    base_c=np.asarray(original['coefficients']).reshape(2,6)
    if a.whole_spine:
        if not a.speed_wave:raise ValueError('Full-body model includes the six speed coefficients')
        idx += extra_axes(names)
    def trial(params):
        system.unpack(snapshot);request('restore');prev=seed_prev.copy()
        impulse=good_pre=good_post=down_j=face_j=0.;loads=[];poses=[];foot_ranges=[];peak_other=0.;first=None
        first_down=first_fwd=None;exit_xy=None;last_loaded=False;q_ex=-np.inf;min_root=9.
        fast=0.;last_wrist=None;wrong=False
        for k in range(start,end):
            prev,target=act(k,prev);t=k*.02;target[leg_idx]+=residual_weights(t)@base_c
            if 2.12<=t<=2.30:target[leg_idx[4]]=min(target[leg_idx[4]],original['backswing_pitch_target_cap'])
            if first is not None:
                gap=np.linalg.norm(wrist.entity.pose.p-hip.entity.pose.p)
                rate=0. if last_wrist is None else (gap-last_wrist)/.02
                last_wrist=gap
                target[shoulder]+=original['wrist_clearance']*np.interp(t-first*DT,[.18,.34,1.25,1.6],[0,1,1,0],left=0.,right=0.)*np.clip((.22-gap-.08*min(0.,rate))/.07,0.,1.)
            target[idx]+=correction(t,params)
            for drive,val in zip(drives,target):drive.set_drive_target(float(val))
            for sub in range(4):
                tick=k*4+sub+1;before=foot.entity.pose;pre=Rotation.from_quat(before.q[[1,2,3,0]])
                scene.step();post=foot.entity.pose;r=Rotation.from_quat(post.q[[1,2,3,0]]);mat=r.as_matrix()
                clearance=float(post.p[2]+(CORNERS@mat[2]).min())
                poses.append([tick,float(post.p@direction),float(robot.root_pose.p@direction),clearance])
                foot_ranges.append(np.linalg.norm(post.p-ball.entity.pose.p))
                q=robot.qpos[order];q_ex=max(q_ex,float(np.max(limits[:,0]-q)),float(np.max(q-limits[:,1])))
                min_root=min(min_root,float(robot.root_pose.p[2]));fast=max(fast,float(np.max(np.abs(robot.qvel[order])/limits[:,2])))
                load=False
                for c in system.get_contacts():
                    bodies=[b.entity.name for b in c.bodies]
                    if not c.points:continue
                    net=np.linalg.norm(np.sum([np.array(p.impulse) for p in c.points],axis=0))/.005
                    if set(bodies)=={'SoccerBall','right_ankle_roll_link'}:
                        if net>2: load=True
                        sign=1 if bodies[0]=='SoccerBall' else -1
                        for p in c.points:
                            j=float(np.linalg.norm(p.impulse))
                            if j==0: continue
                            n=sign*np.array(p.normal);pt=p.position-p.separation*n
                            if paid_contact_patch(pre.inv().apply(pt-before.p),pre.inv().apply(n))=='instep':good_pre+=j
                            if paid_contact_patch(r.inv().apply(pt-post.p),r.inv().apply(n))=='instep':good_post+=j
                            impulse+=j;down_j+=j*(-mat[2,0]);face_j+=j*(mat[:,2]@direction)
                    elif any(name not in ('left_ankle_roll_link','right_ankle_roll_link','ground','SoccerBall') for name in bodies):
                        peak_other=max(peak_other,float(net))
                        if 'SoccerBall' in bodies and net>2:wrong=True
                loads.append(load)
                if load and first is None:first=tick;first_down=-mat[2,0];first_fwd=mat[:,2]@direction
                if last_loaded and not load:exit_xy=ball.linear_velocity[:2].copy()
                last_loaded=load
        if first is None or impulse==0 or exit_xy is None:
            return dict(merit=float(-25.-5.*min(foot_ranges)-min(5.,peak_other/40)),first=None,nonfoot_N=peak_other)
        ae=np.asarray(poses);offs=ae[:,0]-first;i=np.where(offs==0)[0][0];ds=ae[:,1]-ae[i,1]
        relative=ds-(ae[:,2]-ae[i,2]);window=(offs>=0)&(offs<=50);adv=max(ds[window]);rel=max(relative[window])
        v=(ae[4:,1]-ae[:-4,1])/.020
        v120=(ae[i+24,1]-ae[i+20,1])/.02
        back1=float(-v[(offs[4:]>=44)&(offs[4:]<=90)].min());back2=float(-v[(offs[4:]>=44)&(offs[4:]<=150)].min())
        late=bool(np.any(np.asarray(loads)&(offs>=16)&(offs<=150)))
        clear=ae[(offs>=-8)&(offs<=4),3].min();share=min(good_pre,good_post)/impulse
        weighted_down=down_j/impulse;weighted_fwd=face_j/impulse
        speed=float(np.linalg.norm(exit_xy));heading=abs(np.arctan2(exit_xy[0],-exit_xy[1])-np.arctan2(direction[0],-direction[1]))
        angle=np.rad2deg(np.arcsin(np.clip(min(first_down,weighted_down),-1,1)));fwd=min(first_fwd,weighted_fwd)
        # Deficits use native quantities, including all surrogate failures; 14s independent audit remains decisive.
        penalty=min(5.,4*max(0.,.97-share)) + .8*max(0.,3.65-speed) + 5*max(0.,heading-.17)
        penalty+=min(6.,peak_other/15.)+2.*wrong+min(5.,max(0.,q_ex)*5e5)
        penalty+=min(2.,200.*max(0.,.002+q_ex))+2.*max(0.,fast-1.)+8.*max(0.,.60-min_root)
        penalty+=.35*max(0.,back1-1.8)+.2*max(0.,back2-2.3)+1.3*late+2.*max(0,.22-adv)+2.*max(0,.15-rel)+.4*max(0,.6-v120)
        pose_merit=.08*min(77.,angle)+2.*min(.92,fwd)+25.*min(.014,clear)
        if a.profile=='strict':
            deficit = (max(0.,75.-angle)/8. + max(0.,.90-fwd)/.075 +
                       max(0.,3.57-speed)/.27 + max(0.,.97-share)/.10 +
                       max(0.,heading-.17)/.09 + max(0.,.011-clear)/.009)
            deficit += min(6.,peak_other/18.) + 2*wrong + min(6.,max(0.,q_ex-1e-6)*5e5)
            deficit += max(0.,back1-1.9)*.75 + max(0.,back2-2.38)*.5 + .8*late
            deficit += 3*max(0.,.22-adv)+4*max(0.,.15-rel)+max(0.,.60-v120)
            score=12.-deficit + .011*angle + .10*min(4.4,speed)
            if a.progress_deg:
                score-=a.progress_deg*max(0.,75.-angle)/8.
        elif a.profile=='pose': score=pose_merit-.35*penalty
        elif a.profile=='bridge':score=pose_merit-.70*penalty
        else:score=pose_merit-1.3*penalty
        return dict(merit=float(score),first=int(first),angle_deg=float(angle),first_deg=float(np.rad2deg(np.arcsin(np.clip(first_down,-1,1)))),
                    weighted_deg=float(np.rad2deg(np.arcsin(np.clip(weighted_down,-1,1)))),first_forward=float(first_fwd),weighted_forward=float(weighted_fwd),
                    clearance_m=float(clear),instep_pre=good_pre/impulse,instep_post=good_post/impulse,exit_speed=speed,exit_xy=exit_xy.tolist(),
                    nonfoot_N=peak_other,wrong_ball=bool(wrong),q_excess_rad=q_ex,qd_ratio=fast,advance_m=float(adv),rel_m=float(rel),
                    v120=float(v120),return_early_mps=back1,return_mps=back2,late_touch=late,minimum_root_z=min_root,heading_error_rad=float(heading))
    rng=np.random.default_rng(a.seed);mu=np.zeros(18)
    if a.initial:mu=np.array(json.loads(a.initial.read_text())['vertical_parameters'])
    if not a.initial and a.profile=='pose':mu[[0,3,4,6,9,10,12,13,14]]=[.06,.42,.17,.04,.30,.26,-.22,.45,-.19]
    sd=a.spread*np.array([.23,.12,.17,.30,.3,.10,.18,.12,.18,.26,.21,.10,.16,.22,.14,.09,.12,.12])
    if a.speed_wave:
        if mu.size==18:mu=np.r_[mu,np.zeros(6)]
        sd=np.r_[sd,a.spread*np.array([.55,.18,.22,.55,.40,.12])]
    if a.whole_spine:
        if mu.size==24:mu=np.r_[mu,np.zeros(6)]
        sd=np.r_[sd,a.spread*np.array([.22,.32,.38,.32,.38,.32])]
    def save(cfgpath,record):
        cfg={**original,'module':'v162_vertical','vertical_parameters':record['parameters'],'developed_on':'v162 fixed centre CPU only'}
        cfgpath.write_text(json.dumps(cfg,indent=2)+'\n')
    best=-np.inf
    with (a.output/'all.jsonl').open('x',buffering=1) as stream:
        for gen in range(a.generations):
            xs=mu+rng.normal(size=(a.population,len(mu)))*sd;xs[0]=mu
            if gen==0:xs[1]=np.zeros_like(mu)
            results=[]
            for index,x in enumerate(xs):
                row=dict(generation=gen,index=index,parameters=x.tolist(),**trial(x))
                stream.write(json.dumps(row,allow_nan=False)+'\n');results.append(row)
                if row['merit']>best:
                    best=row['merit'];save(a.output/'best.json',row)
            elite=sorted(results,key=lambda x:-x['merit'])[:max(3,a.population//5)]
            coeff=np.array([r['parameters'] for r in elite]);mu=.80*coeff.mean(0)+.20*mu;sd=np.maximum(.04,.75*coeff.std(0)+.25*sd)
            print(gen,round(best,3),{k:elite[0].get(k) for k in ('angle_deg','first_forward','exit_speed','nonfoot_N','instep_post')},flush=True)
if __name__=='__main__':main()
