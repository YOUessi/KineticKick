#!/usr/bin/env python3
"""Experimental ballistic intent -> measured-foot Cartesian shooting feedback.

No calibration claim: desired launch velocity is a planning input. Real ball
velocity and goal-plane height are measured after the trial. Ordinary joint
position/velocity targets only; the existing physical evaluator is unchanged.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[1]


def ballistic_intent(ball,target,horizontal_speed):
    import numpy as np
    b,t=np.array(ball,float),np.array(target,float)
    if b.shape!=(3,) or t.shape!=(3,) or not np.isfinite(np.r_[b,t,horizontal_speed]).all():raise ValueError('Finite3-D points and speed required')
    d=t[:2]-b[:2];distance=float(np.linalg.norm(d))
    if not 1.<=distance<=8. or not 2.<=horizontal_speed<=8. or not .15<=t[2]<=1.:raise ValueError('Intent outside bounded pilot range')
    flight=distance/horizontal_speed
    vz=(t[2]-b[2]+.5*9.81*flight**2)/flight
    return np.r_[d/distance*horizontal_speed,vz],flight


def prepared_foot_path(start_position,start_velocity,start_rotation,contact,impact_velocity,impact_rotation,elapsed,duration):
    """C1 preparation from measured state, then upward follow-through after impact."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    if not .04<=duration<=.8:raise ValueError('Preparation duration must be0.04..0.8s')
    p0,v0,p1,v1=map(lambda a:np.asarray(a,float),(start_position,start_velocity,contact,impact_velocity))
    r0,r1=np.asarray(start_rotation,float),np.asarray(impact_rotation,float)
    if elapsed<=duration:
        u=float(np.clip(elapsed/duration,0,1));u2=u*u;u3=u2*u
        position=(2*u3-3*u2+1)*p0+(u3-2*u2+u)*duration*v0+(-2*u3+3*u2)*p1+(u3-u2)*duration*v1
        velocity=((6*u2-6*u)*p0+(3*u2-4*u+1)*duration*v0+(-6*u2+6*u)*p1+(3*u2-2*u)*duration*v1)/duration
        delta=Rotation.from_matrix(r1@r0.T).as_rotvec();blend=3*u2-2*u3
        rotation=Rotation.from_rotvec(delta*blend).as_matrix()@r0
        angular=delta*(6*u-6*u2)/duration
    else:
        tau=elapsed-duration;lift_time=min(tau,.08)
        # Continue with the accumulated upward speed after the acceleration ramp.
        lift=25*lift_time**2+4*max(tau-.08,0.)
        position=p1+v1*tau+np.array([0,0,lift]);velocity=v1+np.array([0,0,50*lift_time])
        rotation=r1;angular=np.zeros(3)
    return position,velocity,rotation,angular


def solve_contact_normal(leg,point,normal,local_point,local_normal,root,seed):
    """Constrain contact position and surface normal, leaving rotation about normal free."""
    import numpy as np
    q=np.clip(seed,leg.limits[:,0],leg.limits[:,1]).copy()
    normal=np.asarray(normal,float);normal/=np.linalg.norm(normal)
    for _ in range(48):
        pose,J=leg.forward(q,root);offset=pose[:3,:3]@local_point
        actual_point=pose[:3,3]+offset;actual_normal=pose[:3,:3]@local_normal
        dp=point-actual_point;dn=normal-actual_normal
        if np.linalg.norm(dp)<.0005 and np.linalg.norm(dn)<.005:break
        Jpoint=J[:3]+np.cross(J[3:].T,offset).T
        Jnormal=np.cross(J[3:].T,actual_normal).T
        jac=np.vstack([Jpoint,.15*Jnormal]);error=np.r_[dp,.15*dn]
        delta=np.linalg.solve(jac.T@jac+np.eye(6)*1e-4,jac.T@error)
        q=np.clip(q+np.clip(delta,-.12,.12),leg.limits[:,0],leg.limits[:,1])
    pose,_=leg.forward(q,root);actual_normal=pose[:3,:3]@local_normal
    angle=float(np.arccos(np.clip(actual_normal@normal,-1,1)))
    return q,[float(np.linalg.norm(point-pose[:3,3]-pose[:3,:3]@local_point)),angle]


def summarize_intent(result):
    import numpy as np
    from lofted_shooting import summarize_loft
    out=[]
    for i,trial in enumerate(result['trials']):
        intent=trial['cartesian']['intent'];target=intent['target'];rows=result['traces'][i]
        crossing=None
        for a,b in zip(rows,rows[1:]):
            ay,by=a['ball_xyz'][1],b['ball_xyz'][1]
            if ay>target[1]>=by and by<ay:
                alpha=(target[1]-ay)/(by-ay)
                crossing=(np.array(a['ball_xyz'])*(1-alpha)+np.array(b['ball_xyz'])*alpha).tolist();break
        loft=summarize_loft(result,i);seq=result['sequence_records'][i]
        contact_times=[r['time_s'] for r in rows if r['contact_force_N']>2]
        release=next((r for r in rows if contact_times and r['time_s']>contact_times[-1]),None)
        speed=float(np.linalg.norm(release['ball_velocity'][:2])) if release else None
        errorx=abs(crossing[0]-target[0]) if crossing else None
        errorz=abs(crossing[2]-target[2]) if crossing else None
        passed=bool(seq['sequence_success'] and crossing and errorx<=.15 and errorz<=.10
            and speed is not None and abs(speed-intent['horizontal_speed_mps'])<=.6)
        out.append(dict(request=intent,goal_crossing_xyz=crossing,horizontal_error_m=errorx,height_error_m=errorz,
            release_horizontal_speed_mps=speed,flight=loft,full_sequence_success=seq['sequence_success'],intent_success=passed,
            scope='Single commanded physical trial; not a validated power/height calibration'))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',required=True);p.add_argument('--output',required=True)
    own,remaining=p.parse_known_args()
    trials=json.loads(Path(own.trials).read_text())
    if not 1<=len(trials)<=12:raise ValueError('Bounded probe needs1..12trials')
    if Path(own.output).exists():raise FileExistsError(own.output)
    sys.path.insert(0,str(ROOT/'tools'))
    spec=importlib.util.spec_from_file_location('_feedback_shot_evaluator',ROOT/'tools/eval_free_shooting.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);native_run=base.run

    def run_after_isaac(args,points):
        import gymnasium as gym
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        import cartesian_strike as module
        from shooting_control import StrikeResidual
        original_class=module.CartesianStrike;original_make=gym.make;created={}
        def make(*a,**kw):
            env=original_make(*a,**kw);created['env']=env.unwrapped;return env

        class FeedbackStrike:
            def __init__(self,trials,joint_names,scale,offset,limits,urdf):
                self.trials=trials;self.env=created['env'];self.robot=self.env.scene['robot']
                self.leg=module.LegKinematics(urdf);self.indices=[joint_names.index(n) for n in self.leg.names]
                self.envelope=StrikeResidual([t.get('cartesian',{}) for t in trials],joint_names,scale,offset,limits)
                self.last_errors=[None]*len(trials);self.plans=[];self.preparation=[None]*len(trials)
                for t in trials:
                    c=t['cartesian'];intent=c['intent'];v,duration=ballistic_intent([*t['ball'],.11],intent['target'],intent['horizontal_speed_mps'])
                    n=v/np.linalg.norm(v);normal=np.array(c['normal_local']);normal/=np.linalg.norm(normal)
                    R=Rotation.from_quat(c['orientation_xyzw']).as_matrix()
                    align,_=Rotation.align_vectors([n],[R@normal]);R=align.as_matrix()@R
                    local=np.array(c['point_local']);contact=np.r_[t['ball'],.11]-.11*n
                    # Upward launch comes from the below-centre surface normal. A straight
                    # upward approach would put the preceding foot trajectory under the floor.
                    velocity=np.r_[v[:2]/float(c.get('estimated_transfer_gain',1.4)),0.]
                    self.plans.append((R,local,contact,velocity))
                    intent.update(planned_launch_velocity_mps=v.tolist(),planned_flight_s=duration,estimated_foot_velocity_mps=velocity.tolist())
                sensor=self.env.scene['contact_forces']
                from pxr import UsdPhysics
                import omni.usd
                ground=[str(p.GetPath()) for p in omni.usd.get_context().get_stage().Traverse() if str(p.GetPath()).startswith('/World/ground/') and p.HasAPI(UsdPhysics.CollisionAPI)]
                self.support=sensor._physics_sim_view.create_rigid_contact_view('/World/envs/env_*/Robot/left_ankle_roll_link',ground,max_contact_data_count=64*len(trials))

            def apply(self,actions,frames,root_poses,joint_positions,dt):
                result=np.array(actions,copy=True);weights,_=self.envelope.envelope(frames)
                support=self.support.get_contact_force_matrix(self.env.physics_dt).reshape(len(self.trials),-1,3)[:,:,2].sum(1).cpu().numpy()
                velocity_targets=self.robot.data.joint_vel_target.clone()
                for i,t in enumerate(self.trials):
                    c=t['cartesian'];w=weights[i,0]
                    if w<=0 or support[i]<80:continue
                    pose=root_poses[i];root=np.eye(4);root[:3,:3]=Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix();root[:3,3]=pose[:3]
                    R,local,contact,base_velocity=self.plans[i]
                    actual,J=self.leg.forward(joint_positions[i,self.indices],root)
                    offset=actual[:3,:3]@local;Jpoint=J.copy();Jpoint[:3]+=np.cross(J[3:].T,offset).T
                    base_v=self.robot.data.root_lin_vel_w[i].cpu().numpy();base_w=self.robot.data.root_ang_vel_w[i].cpu().numpy()
                    point=actual[:3,3]+offset
                    if self.preparation[i] is None:
                        duration=(c['contact_frame']-frames[i])*dt
                        if duration<.04:continue
                        joint_v=self.robot.data.joint_vel[i,self.indices].cpu().numpy()
                        point_v=base_v+np.cross(base_w,point-root[:3,3])+Jpoint[:3]@joint_v
                        self.preparation[i]=(frames[i],point.copy(),point_v,actual[:3,:3].copy(),duration)
                    start,p0,v0,r0,duration=self.preparation[i]
                    target_point,velocity,target_rotation,angular=prepared_foot_path(
                        p0,v0,r0,contact,base_velocity,R,(frames[i]-start)*dt,duration)
                    local_normal=np.array(c['normal_local'],float);local_normal/=np.linalg.norm(local_normal)
                    desired_normal=target_rotation@local_normal
                    q,error=solve_contact_normal(self.leg,target_point,desired_normal,local,local_normal,root,joint_positions[i,self.indices]);self.last_errors[i]=error
                    if error[0]>.015 or error[1]>.15:continue
                    current=result[i,self.indices]*self.envelope.scale[i,self.indices]+self.envelope.offset[i,self.indices]
                    corrected=current+np.clip(q-current,-.4,.4)*w
                    bounds=self.envelope.limits[i,self.indices];corrected=np.clip(corrected,bounds[:,0],bounds[:,1])
                    result[i,self.indices]=(corrected-self.envelope.offset[i,self.indices])/self.envelope.scale[i,self.indices]
                    actual_normal=actual[:3,:3]@local_normal
                    twist=np.r_[velocity-base_v-np.cross(base_w,point-root[:3,3]),np.cross(angular,desired_normal)-np.cross(base_w,actual_normal)]
                    Jpoint[3:]=np.cross(J[3:].T,actual_normal).T
                    Jpoint[3:]*=.15;twist[3:]*=.15
                    dq=np.linalg.solve(Jpoint.T@Jpoint+np.eye(6)*.001,Jpoint.T@twist)
                    limit=self.robot.data.joint_vel_limits[i,self.indices].cpu().numpy()
                    dq=np.clip(dq,-np.minimum(limit,12.),np.minimum(limit,12.))*w
                    velocity_targets[i,self.indices]=torch.tensor(dq,device=self.robot.device,dtype=torch.float32)
                self.robot.set_joint_velocity_target(velocity_targets)
                return result

        gym.make=make;module.CartesianStrike=FeedbackStrike
        args.source_provenance.update(shot_feedback_source=str(Path(__file__).resolve()),
            shot_feedback_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            intent_status='experimental_unvalidated',physical_control='Position and velocity joint targets, left support gated80N; no impulses')
        try:
            value=native_run(args,points)
            result=json.loads(Path(args.output).read_text());result['intent_records']=summarize_intent(result)
            Path(args.output).write_text(json.dumps(result,indent=2))
            print('SHOT_INTENT_RESULT',json.dumps(result['intent_records']),flush=True)
            return value
        finally:gym.make=original_make;module.CartesianStrike=original_class
    base.run=run_after_isaac
    sys.argv=[str(ROOT/'tools/eval_free_shooting.py'),*remaining,'--trials',own.trials,'--output',own.output,'--sequence-evidence']
    base.main()


if __name__=='__main__':main()
