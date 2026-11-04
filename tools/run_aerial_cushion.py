#!/usr/bin/env python3
"""Predictive free-root aerial cushioning; physical ball initialized once.

The isolated harness reuses the frozen physics logger, not its shooting scores.
A prepared single-leg pose is the starting scope; approach is not demonstrated.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--video-dir',type=Path);p.add_argument('--steps',type=int)
    a=p.parse_args();plan=json.loads(a.plan.read_text());steps=a.steps or plan['steps']
    if a.output.exists():raise FileExistsError(a.output)
    if not 50<=steps<=300:raise ValueError('Pilot is bounded to 1..6 seconds')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('_aerial_base',ROOT/'tools/eval_free_shooting.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);native_run=base.run

    def aerial_run(args,points):
        import gymnasium as gym
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        from isaaclab.utils.math import quat_apply,quat_apply_inverse
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from isaaclab.envs import mdp as physical_mdp
        import soccer_onnx_policy as policy_module
        import free_shooting_metrics as contact_module
        from running_balance import nonfoot_contact_termination,ground_force
        from aerial_gravity import GravitySupport
        from cartesian_strike import LegKinematics
        from cushion_motion import CushionMotion,SideLegKinematics
        import strike_velocity as velocity_module
        NativeVelocity=velocity_module.StrikeVelocityPulse
        from contact_frame_sampler import ContactFrameSampler
        NativeActor=policy_module.OnnxSoccerPolicy;native_make=gym.make;native_events=contact_module.contact_point_events
        created={}
        def make(*pos,**kw):
            cfg=kw['cfg'];cfg.scene.contact_forces.history_length=4
            cfg.terminations.ee_body_pos=None;cfg.terminations.anchor_pos_z=None;cfg.terminations.anchor_ori=None
            cfg.terminations.physical_height=DoneTerm(func=physical_mdp.root_height_below_minimum,params={'minimum_height':.45})
            cfg.terminations.physical_orientation=DoneTerm(func=physical_mdp.bad_orientation,params={'limit_angle':math.radians(60)})
            cfg.terminations.physical_nonfoot=DoneTerm(func=nonfoot_contact_termination,params={'threshold':20.})
            cfg.viewer.eye=(2.1,1.1,1.45);cfg.viewer.lookat=(-.05,-.13,.65)
            env=native_make(*pos,**kw);created['env']=env.unwrapped;return env

        class AerialActor:
            def __init__(self,path,num_envs):
                if num_envs!=1:raise ValueError('One aerial pilot at a time')
                self.metadata=NativeActor(path,num_envs);self.observation_names=self.metadata.observation_names
                self.env=created['env'];self.robot=self.env.scene['robot'];self.ball=self.env.scene['soccer_ball']
                self.leg=LegKinematics(ROOT/'configs/paid_original/paid_main_resolved.urdf')
                self.ids=[self.robot.joint_names.index(n) for n in self.leg.names]
                self.left_leg=SideLegKinematics(ROOT/'configs/paid_original/paid_main_resolved.urdf','left')
                self.left_ids=[self.robot.joint_names.index(n) for n in self.left_leg.names]
                self.arm_ids=[i for i,n in enumerate(self.robot.joint_names) if any(part in n for part in ('shoulder','elbow','wrist'))]
                self.foot_id=self.robot.body_names.index('right_ankle_roll_link')
                self.contact_frames=ContactFrameSampler(self.env)
                self.action_term=self.env.action_manager.get_term('joint_pos')
                self.recovery_reference=np.load(ROOT/'third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-010_right.npz')['joint_pos'][210]
                self.q0=np.array(plan['ready_joint_pos']);self.right_target=np.array(plan['feet']['right'])
                self.gravity=GravitySupport(ROOT/'configs/paid_original/main.urdf',self.robot.joint_names)
                self.mass=self.robot.root_physx_view.get_masses().to(self.robot.device)
                self.weights=self.mass/self.mass.sum(dim=1,keepdim=True)
                import omni.usd
                from pxr import UsdPhysics
                ground=[str(p.GetPath()) for p in omni.usd.get_context().get_stage().Traverse() if str(p.GetPath()).startswith('/World/ground/') and p.HasAPI(UsdPhysics.CollisionAPI)]
                if not ground:
                    ground=[str(p.GetPath()) for p in omni.usd.get_context().get_stage().Traverse() if str(p.GetPath()).startswith('/World/terrain/') and p.HasAPI(UsdPhysics.CollisionAPI)]
                if not ground:raise RuntimeError('No physical ground found')
                self.ball_ground=self.env.scene['contact_forces']._physics_sim_view.create_rigid_contact_view('/World/envs/env_*/SoccerBall',ground,max_contact_data_count=64)
                self.initialization_writes=[];self.reset()
                pose_write=self.ball.write_root_pose_to_sim;vel_write=self.ball.write_root_velocity_to_sim
                def init_pose(pose,*aa,**kk):
                    if self.started:raise RuntimeError('Ball state write after first action')
                    value=pose.clone();value[:,:3]=torch.tensor([plan['ball_position']],device=self.robot.device)+self.env.scene.env_origins
                    self.initialization_writes.append('ball_pose');return pose_write(value,*aa,**kk)
                def init_vel(vel,*aa,**kk):
                    if self.started:raise RuntimeError('Ball velocity write after first action')
                    value=vel.clone();value[:,:3]=torch.tensor([plan['ball_velocity']],device=self.robot.device);value[:,3:]=0
                    self.initialization_writes.append('ball_velocity');return vel_write(value,*aa,**kk)
                self.ball.write_root_pose_to_sim=init_pose;self.ball.write_root_velocity_to_sim=init_vel
                scene_update=self.env.scene.update
                def sample(dt):
                    self.contact_frames.before_update();scene_update(dt);self.contact_frames.after_update(dt)
                    if not self.started:return
                    self.sensor_trace.append(dict(ball_ground_force_N=float(self.ball_ground.get_contact_force_matrix(dt).norm(dim=-1).sum()),
                        right_foot_position=self.robot.data.body_pos_w[0,self.foot_id].cpu().tolist(),
                        right_foot_velocity=self.robot.data.body_link_lin_vel_w[0,self.foot_id].cpu().tolist(),phase=self.phase))
                    force=self.env.scene['right_ball_contact'].data.force_matrix_w
                    contact=bool(force is not None and float(force.norm(dim=-1).max())>2.)
                    if contact and not self.touched:self.first_contact_time=len(self.sensor_trace)*dt
                    self.touched |= contact
                self.env.scene.update=sample
            def validate_joints(self,names):
                self.metadata.validate_joints(names)
                if list(names)!=plan['joint_names']:raise ValueError('Ready pose joint order differs')
            def reset(self):
                self.metadata.reset();self.touched=False;self.first_contact_time=None;self.recovery_started=None;self.recovery_cues=None;self.last_recovery_weight=0.
                self.started=False;self.t=0.;self.phase='receive';self.sensor_trace=[];self.events=[];self.seed=self.q0[self.ids].copy()
                self.motion=CushionMotion(plan);self.control_trace=[];self.velocity_targets=np.zeros((1,len(self.q0)),np.float32)
                self.last_left=self.q0[self.left_ids].copy();self.left_seed=self.last_left.copy()
            def step(self,obs,frame):
                self.started=True;now=self.t;self.t+=self.env.step_dt
                ball=self.ball.data.root_pos_w[0].cpu().numpy();vel=self.ball.data.root_lin_vel_w[0].cpu().numpy()
                self.motion.update_prediction(now,ball,vel,self.touched)
                target_time=now+plan['cushion'].get('actuator_lead_s',0.)
                target,desired_velocity,desired_root,self.phase=self.motion.targets(target_time)
                q=self.q0.copy()
                root=np.eye(4);root[:3,3]=self.robot.data.root_pos_w[0].cpu().numpy();qw=self.robot.data.root_quat_w[0].cpu().numpy()
                rot=Rotation.from_quat(qw[[1,2,3,0]]);root[:3,:3]=rot.as_matrix()
                self.left_seed,left_error=self.left_leg.solve(np.array(plan['feet']['left']),desired_root,self.left_seed)
                q[self.left_ids]=self.left_seed
                self.seed,error=self.leg.solve(target,root,self.seed);q[self.ids]=self.seed
                actual_q=self.robot.data.joint_pos[0].cpu().numpy();actual_dq=self.robot.data.joint_vel[0].cpu().numpy()
                actual_pose,jacobian=self.leg.forward(actual_q[self.ids],root)
                actual_velocity=self.robot.data.body_link_lin_vel_w[0,self.foot_id].cpu().numpy()
                actual_angular=self.robot.data.body_ang_vel_w[0,self.foot_id].cpu().numpy()
                base_linear=actual_velocity-jacobian[:3]@actual_dq[self.ids]
                base_angular=actual_angular-jacobian[3:]@actual_dq[self.ids]
                rotation_error=Rotation.from_matrix(target[:3,:3]@actual_pose[:3,:3].T).as_rotvec()
                task_velocity=np.r_[desired_velocity+plan['cushion']['position_feedback_gain']*(target[:3,3]-actual_pose[:3,3])-base_linear,self.motion.angular_velocity+5.*rotation_error-base_angular]
                weighted=jacobian.copy();weighted[3:]*=.15;task_velocity[3:]*=.15
                dq=np.linalg.solve(weighted.T@weighted+np.eye(6)*1e-4,weighted.T@task_velocity)
                self.velocity_targets[:]=0.
                self.velocity_targets[0,self.ids]=dq
                self.velocity_targets[0,self.left_ids]=(self.left_seed-self.last_left)/self.env.step_dt
                self.last_left=self.left_seed.copy()
                self.control_trace.append(dict(time_s=now,target_time_s=target_time,phase=self.phase,contact_prediction_s=self.motion.contact_time,
                    desired_foot_position=target[:3,3].tolist(),desired_foot_velocity=desired_velocity.tolist(),
                    actual_foot_position=actual_pose[:3,3].tolist(),actual_foot_velocity=actual_velocity.tolist(),
                    desired_root_height_m=float(desired_root[2,3]),ik_error=error,left_ik_error=left_error))
                com=(self.robot.data.body_com_pos_w*self.weights[:,:,None]).sum(1)[0].cpu().numpy()
                cv=(self.robot.data.body_com_lin_vel_w*self.weights[:,:,None]).sum(1)[0].cpu().numpy()
                yaw=Rotation.from_euler('z',-math.pi/2)
                support=np.array(plan['feet']['left'])[:3,3]+yaw.apply([.025,0,0])
                ce=yaw.inv().apply(com-support);local_v=yaw.inv().apply(cv)
                tilt=(yaw.inv()*rot).as_rotvec();omega=self.robot.data.root_ang_vel_b[0].cpu().numpy()
                correction=plan['balance_kp_com']*ce+plan['balance_kd_com']*local_v
                roll=-correction[1]+plan['balance_kp_tilt']*tilt[0]+plan['balance_kd_tilt']*omega[0]
                pitch=correction[0]+plan['balance_kp_tilt']*tilt[1]+plan['balance_kd_tilt']*omega[1]
                q[self.robot.joint_names.index('left_ankle_roll_joint')]+=np.clip(roll,-.18,.18)
                q[self.robot.joint_names.index('left_ankle_pitch_joint')]+=np.clip(pitch,-.18,.18)
                lim=self.robot.data.soft_joint_pos_limits[0].cpu().numpy();q=np.clip(q,lim[:,0],lim[:,1])
                support_force=ground_force(self.env)[0].cpu().numpy()
                fraction=float(support_force[0]/support_force.sum()) if support_force.sum()>30 else 1.
                tau=self.gravity.torques(q,root,left_fraction=fraction)
                recovery_action=None;weight=0.
                if plan.get('recovery'):
                    if self.recovery_cues is None:
                        cues=torch.tensor([[[1.32552,1.56739,-.7],[.47127,1.15748,-.7]]],dtype=torch.float32,device=self.robot.device)
                        quat=self.robot.data.root_quat_w[:,None,:].expand(-1,2,-1)
                        self.recovery_cues=self.robot.data.root_pos_w[:,None,:]+quat_apply(quat,cues)
                        self.recovery_cues[:,:,2]=.11
                    shot=np.array(obs,copy=True);shot[:,:29]=self.recovery_reference;shot[:,29:58]=0.
                    cues=quat_apply_inverse(self.robot.data.root_quat_w[:,None,:].expand(-1,2,-1),self.recovery_cues-self.robot.data.root_pos_w[:,None,:]).cpu().numpy()
                    shot[:,-6:-3]=cues[:,0];shot[:,-3:]=cues[:,1]
                    recovery_action=self.metadata.step(shot,229)
                    if self.recovery_started is None and self.touched and now>=max(plan['recovery']['earliest_s'],self.first_contact_time+plan['recovery'].get('contact_delay_s',0.)):
                        self.recovery_started=now;self.events.append(dict(kind='learned_standing_recovery',time_s=now,physical_state_write=False))
                    if self.recovery_started is not None:
                        u=min(1.,(now-self.recovery_started)/plan['recovery']['blend_seconds']);weight=.5-.5*math.cos(math.pi*u);self.phase='recover'
                feedback=np.array(plan['cushion'].get('velocity_effort_gains',[0.]*6))*(self.velocity_targets[0,self.ids]-actual_dq[self.ids])
                tau[self.ids]+=feedback
                self.control_trace[-1]['velocity_feedback_torque']=feedback.tolist()
                arm_gravity=tau[self.arm_ids].copy()
                tau*=1-weight
                if plan.get('keep_receive_arms'):tau[self.arm_ids]=arm_gravity
                self.last_recovery_weight=weight
                self.velocity_targets*=1-weight
                vmax=plan['cushion']['velocity_limit_rad_s']
                self.velocity_targets=np.clip(self.velocity_targets,-vmax,vmax)
                self.velocity_targets=np.clip(self.velocity_targets,(lim[:,0]-actual_q)[None]/.04,(lim[:,1]-actual_q)[None]/.04).astype(np.float32)
                limits=self.robot.data.joint_effort_limits[0].cpu().numpy()
                self.robot.set_joint_effort_target(torch.tensor(np.clip(tau,-limits,limits)[None],dtype=torch.float32,device=self.robot.device))
                scale=self.action_term._scale.cpu().numpy();offset=self.action_term._offset.cpu().numpy()
                action=((q[None]-offset)/scale).astype(np.float32)
                combined=action if recovery_action is None else (1-weight)*action+weight*recovery_action
                if plan.get('keep_receive_arms'):combined[:,self.arm_ids]=((self.q0[None]-offset)/scale)[:,self.arm_ids]
                return combined
        def aligned_events(*pos,**kw):return created['actor'].contact_frames.events(native_events,*pos,**kw)
        class RegisteredActor(AerialActor):
            def __init__(self,*p,**k):super().__init__(*p,**k);created['actor']=self
        class CushionVelocity:
            def __init__(self,entries,joint_names):pass
            def apply(self,frames,support):
                return created['actor'].velocity_targets.copy() if float(support[0])>=80. else np.zeros_like(created['actor'].velocity_targets)
        velocity_module.StrikeVelocityPulse=CushionVelocity
        gym.make=make;policy_module.OnnxSoccerPolicy=RegisteredActor;contact_module.contact_point_events=aligned_events
        args.source_provenance.update(aerial_control_source_sha256=sha(__file__),aerial_plan_sha256=sha(a.plan),
            incoming_ball_position=plan['ball_position'],incoming_ball_velocity=plan['ball_velocity'],
            aerial_initialization='Prepared single-leg receive pose; ball initialized once before the first action',
            aerial_control='Ballistic prediction; coordinated stance crouch; continuous foot position and velocity targets; normal joint actuators',
            cushion_motion_sha256=sha(ROOT/'tools/cushion_motion.py'))
        try:
            value=native_run(args,points);result=json.loads(a.output.read_text());actor=created['actor']
            result['aerial_sensor_trace']=actor.sensor_trace;result['aerial_control_events']=actor.events
            result['cushion_control_trace']=actor.control_trace;result['ball_predictions']=actor.motion.predictions
            result['aerial_initialization_writes']=actor.initialization_writes
            result['aerial_standing_conditioning_world']=actor.recovery_cues.cpu().tolist() if actor.recovery_cues is not None else None
            result['aerial_final_recovery_weight']=actor.last_recovery_weight
            result['contact_evidence_version']=3
            result['legacy_shooting_records_applicable']=False
            from aerial_control_metrics import summarize_aerial_control
            result['aerial_control']=summarize_aerial_control(result)
            a.output.write_text(json.dumps(result,indent=2));print('AERIAL_CONTROL_RESULT',json.dumps(result['aerial_control']),flush=True)
            return value
        finally:gym.make=native_make;policy_module.OnnxSoccerPolicy=NativeActor;contact_module.contact_point_events=native_events;velocity_module.StrikeVelocityPulse=NativeVelocity
    base.run=aerial_run
    trial=dict(name='aerial_foot_trap_pilot',motion_file=str(ROOT/plan['reference']),ball=plan['ball_position'][:2],goal=[0.,-3.7],goal_forward=[0,-1],style='instep',mode='hip')
    with tempfile.TemporaryDirectory(prefix='kinetickick-aerial-') as d:
        manifest=Path(d)/'trial.json';manifest.write_text(json.dumps([trial]))
        sys.argv=[str(ROOT/'tools/eval_free_shooting.py'),'--trials',str(manifest),'--output',str(a.output),'--sequence-evidence','--steps',str(steps),'--seed',str(plan['seed'])]
        if a.video_dir:sys.argv+=['--video-dir',str(a.video_dir)]
        base.main()

if __name__=='__main__':main()
