#!/usr/bin/env python3
"""Inside-foot receiving of ground passes and aerial balls; free robot and physical ball.

The isolated harness reuses the frozen physics logger, not its shooting scores.
Start with both feet down; live ball state sets interception height and timing.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--video-dir',type=Path);p.add_argument('--steps',type=int);p.add_argument('--scenario')
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    if 'scenarios' in plan:
        a.scenario=a.scenario or 'ground_center'
        if a.scenario not in plan['scenarios']:p.error('Unknown scenario: '+a.scenario)
        plan={**plan['controller'],**plan['scenarios'][a.scenario]}
    steps=a.steps or plan['steps']
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
        from support_balance import support_cop_correction
        from receiving_stance import grounded_support_correction
        from cartesian_strike import LegKinematics
        from cushion_motion import SideLegKinematics,smooth_step
        from reactive_receive_motion import ReactiveReceiveMotion as InsideReceiveMotion
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
            cfg.viewer.eye=tuple(plan.get('viewer',{}).get('eye',(2.1,1.1,1.45)));cfg.viewer.lookat=tuple(plan.get('viewer',{}).get('lookat',(-.05,-.13,.65)))
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
                    value=vel.clone();value[:,:3]=torch.tensor([plan['ball_velocity']],device=self.robot.device);value[:,3:]=torch.tensor([plan.get('ball_angular_velocity',[0.,0.,0.])],device=self.robot.device)
                    self.initialization_writes.append('ball_velocity');return vel_write(value,*aa,**kk)
                self.ball.write_root_pose_to_sim=init_pose;self.ball.write_root_velocity_to_sim=init_vel
                scene_update=self.env.scene.update
                def sample(dt):
                    self.contact_frames.before_update();scene_update(dt);self.contact_frames.after_update(dt)
                    if not self.started:return
                    foot_q=self.robot.data.body_quat_w[0,self.foot_id].cpu().numpy()
                    lever=Rotation.from_quat(foot_q[[1,2,3,0]]).apply(plan['contact_point_local'])
                    point_velocity=self.robot.data.body_link_lin_vel_w[0,self.foot_id].cpu().numpy()+np.cross(self.robot.data.body_ang_vel_w[0,self.foot_id].cpu().numpy(),lever)
                    self.sensor_trace.append(dict(inside_point_velocity=point_velocity.tolist(),ball_angular_velocity=self.ball.data.root_ang_vel_w[0].cpu().tolist(),ball_ground_force_N=float(self.ball_ground.get_contact_force_matrix(dt).norm(dim=-1).sum()),
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
                self.release_started=None;self.release_pose=None;self.release_delta=None;self.release_reason=None;self.place_pose=None;self.planted_pose=None;self.planted_at=None
                self.started=False;self.t=0.;self.phase='receive';self.sensor_trace=[];self.events=[];self.seed=self.q0[self.ids].copy()
                self.motion=InsideReceiveMotion(plan);self.control_trace=[];self.velocity_targets=np.zeros((1,len(self.q0)),np.float32)
                self.last_left=self.q0[self.left_ids].copy();self.left_seed=self.last_left.copy();self.ready_steps=0
            def step(self,obs,frame):
                self.started=True;now=self.t;self.t+=self.env.step_dt
                ball=self.ball.data.root_pos_w[0].cpu().numpy();vel=self.ball.data.root_lin_vel_w[0].cpu().numpy()
                self.motion.update_prediction(now,ball,vel,self.touched)
                target_time=now+plan['cushion'].get('actuator_lead_s',0.)
                target,desired_velocity,desired_root,self.phase=self.motion.targets(target_time)
                if self.release_started is not None:
                    duration=plan['recovery']['clearance_seconds']
                    u=np.clip((now-self.release_started)/duration,0.,1.)
                    target=self.release_pose.copy();target[:3,3]+=self.release_delta*smooth_step(u)
                    desired_velocity=self.release_delta*30*u*u*(1-u)**2/duration
                    self.motion.angular_velocity[:]=0.;self.phase='clear_ball'
                motor_stand=plan['recovery'].get('mode')=='motor_stand'
                load=0.
                if motor_stand and self.recovery_started is not None:
                    duration=plan['recovery'].get('place_seconds',.5)
                    u=np.clip((now-self.recovery_started)/duration,0.,1.);w=smooth_step(u)
                    if self.place_pose is None:self.place_pose=target.copy()
                    goal=np.array(plan['feet']['right']);goal[2,3]-=.002
                    target=self.place_pose.copy();delta=goal[:3,3]-self.place_pose[:3,3]
                    target[:3,3]+=w*delta;rate=30*u*u*(1-u)**2/duration
                    desired_velocity=rate*delta
                    dr=Rotation.from_matrix(goal[:3,:3]@self.place_pose[:3,:3].T).as_rotvec()
                    target[:3,:3]=Rotation.from_rotvec(w*dr).as_matrix()@self.place_pose[:3,:3]
                    self.motion.angular_velocity=rate*dr;self.phase='place_foot' if u<1. else 'stand'
                    if self.planted_pose is not None:
                        target=self.planted_pose.copy();desired_velocity=np.zeros(3)
                        self.motion.angular_velocity[:]=0.;self.phase='stand'
                        load=.10*smooth_step((now-self.planted_at)/.6)
                q=self.q0.copy()
                root=np.eye(4);root[:3,3]=self.robot.data.root_pos_w[0].cpu().numpy();qw=self.robot.data.root_quat_w[0].cpu().numpy()
                rot=Rotation.from_quat(qw[[1,2,3,0]]);root[:3,:3]=rot.as_matrix()
                self.left_seed,left_error=self.left_leg.solve(np.array(plan['feet']['left']),desired_root,self.left_seed)
                q[self.left_ids]=self.left_seed
                actual_q=self.robot.data.joint_pos[0].cpu().numpy();actual_dq=self.robot.data.joint_vel[0].cpu().numpy()
                actual_pose,jacobian=self.leg.forward(actual_q[self.ids],root)
                self.motion.actual_foot=actual_pose.copy()
                if (motor_stand and self.recovery_started is not None and self.planted_pose is None
                        and now-self.recovery_started>.25 and float(ground_force(self.env)[0,1])>10.):
                    self.planted_pose=actual_pose.copy();self.planted_at=now
                    self.events.append(dict(kind='measured_foot_plant',time_s=now,
                        actual_foot_position=actual_pose[:3,3].tolist(),physical_state_write=False))
                if self.touched and self.recovery_started is None and self.release_started is None:
                    force=self.env.scene['right_ball_contact'].data.force_matrix_w
                    magnitude=float(force.norm(dim=-1).max()) if force is not None else 0.
                    target=self.motion.yield_to_contact(target,actual_pose[:3,3],magnitude)
                soft_brake=self.motion.aerial and self.phase=='brake_ground' and plan['reaction'].get('soft_aerial_brake',False)
                if soft_brake:
                    normal=target[:3,1]
                    lever=np.array(plan['contact_center_local'])+np.array([0.,.12,0.])
                    desired_center=target[:3,3]+target[:3,:3]@lever
                    actual_center=actual_pose[:3,3]+actual_pose[:3,:3]@lever
                    closing=float((desired_center-actual_center)@normal)
                    target[:3,3]-=normal*max(0.,closing-.02*self.env.step_dt)
                self.seed,error=self.leg.solve(target,root,self.seed);q[self.ids]=self.seed
                actual_velocity=self.robot.data.body_link_lin_vel_w[0,self.foot_id].cpu().numpy()
                actual_angular=self.robot.data.body_ang_vel_w[0,self.foot_id].cpu().numpy()
                base_linear=actual_velocity-jacobian[:3]@actual_dq[self.ids]
                base_angular=actual_angular-jacobian[3:]@actual_dq[self.ids]
                rotation_error=Rotation.from_matrix(target[:3,:3]@actual_pose[:3,:3].T).as_rotvec()
                task_velocity=np.r_[desired_velocity+plan['cushion']['position_feedback_gain']*(target[:3,3]-actual_pose[:3,3])-base_linear,self.motion.angular_velocity+plan['cushion'].get('rotation_feedback_gain',5.)*rotation_error-base_angular]
                if soft_brake:
                    normal=target[:3,1]
                    relative=float((task_velocity[:3]+base_linear-vel)@normal)
                    task_velocity[:3]-=normal*max(0.,relative-.02)
                weighted=jacobian.copy();weighted[3:]*=.15;task_velocity[3:]*=.15
                dq=np.linalg.solve(weighted.T@weighted+np.eye(6)*1e-4,weighted.T@task_velocity)
                self.velocity_targets[:]=0.
                self.velocity_targets[0,self.ids]=dq
                self.velocity_targets[0,self.left_ids]=(self.left_seed-self.last_left)/self.env.step_dt
                self.last_left=self.left_seed.copy()
                self.control_trace.append(dict(time_s=now,target_time_s=target_time,phase=self.phase,contact_prediction_s=self.motion.contact_time,
                    desired_foot_position=target[:3,3].tolist(),desired_foot_velocity=desired_velocity.tolist(),
                    actual_foot_position=actual_pose[:3,3].tolist(),actual_foot_velocity=actual_velocity.tolist(),
                    absorbed_displacement_m=self.motion.absorbed_displacement,desired_root_height_m=float(desired_root[2,3]),ik_error=error,left_ik_error=left_error))
                com=(self.robot.data.body_com_pos_w*self.weights[:,:,None]).sum(1)[0].cpu().numpy()
                cv=(self.robot.data.body_com_lin_vel_w*self.weights[:,:,None]).sum(1)[0].cpu().numpy()
                yaw=Rotation.from_matrix(np.array(plan['feet']['left'])[:3,:3])
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
                # Keep one coherent support wrench while the receiving foot unloads.
                stance_fraction=1.-load
                tau=self.gravity.torques(actual_q,root,left_fraction=stance_fraction)
                left_pose,left_jacobian=self.left_leg.forward(actual_q[self.left_ids],root)
                if load>0.:
                    correction_torque,balance_data=grounded_support_correction(com,cv,left_pose,actual_pose,left_jacobian,float(self.mass.sum()),stance_fraction,plan['support_balance'])
                else:
                    correction_torque,balance_data=support_cop_correction(com,cv,left_pose,left_jacobian,float(self.mass.sum()),plan['support_balance'])
                if support_force[0]>80.:tau[self.left_ids]+=correction_torque
                # A moving receive leg also excites torso rotation, which CoP position
                # feedback alone does not damp. Apply the reaction through stance hips.
                angular=self.robot.data.root_ang_vel_w[0].cpu().numpy()
                rotation_error=Rotation.from_matrix(root[:3,:3]@desired_root[:3,:3].T).as_rotvec()
                stabilize=np.clip(plan['support_balance'].get('root_rotation_gain',0.)*rotation_error+
                    plan['support_balance'].get('root_angular_damping',0.)*angular,-12.,12.)
                if support_force[0]>80.:tau[self.left_ids[:3]]+=left_jacobian[3:,:3].T@stabilize
                balance_data['root_rotation_error']=rotation_error.tolist()
                balance_data['root_angular_velocity']=angular.tolist()
                balance_data['hip_stabilizing_moment']=stabilize.tolist()
                self.control_trace[-1]['support_balance']=balance_data
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
                    slow=(np.linalg.norm(vel)<plan['recovery']['maximum_ball_speed'] and float(self.ball.data.root_ang_vel_w[0].norm())<plan['recovery']['maximum_ball_spin'])
                    self.ready_steps=self.ready_steps+1 if self.touched and slow else 0
                    timeout=bool(self.touched and now>=self.first_contact_time+plan['recovery']['maximum_wait_s'])
                    captured=self.ready_steps>=plan['recovery']['ready_steps']
                    release_due=self.touched and (captured or timeout) and now>=max(plan['recovery']['earliest_s'],self.first_contact_time+plan['recovery'].get('contact_delay_s',0.))
                    if self.recovery_started is None and self.release_started is None and release_due:
                        reason='ball_slow_and_low_spin' if captured else 'balance_timeout'
                        if plan['recovery'].get('clearance_seconds',0.) > 0:
                            self.release_started=now;self.release_pose=actual_pose.copy();self.release_reason=reason
                            direction=actual_pose[:2,3]-ball[:2]
                            if np.linalg.norm(direction)<.01:direction=-self.motion.normal[:2]
                            direction/=np.linalg.norm(direction)
                            self.release_delta=np.r_[direction*plan['recovery']['clearance_horizontal_m'],plan['recovery']['clearance_up_m']]
                            self.events.append(dict(kind='separate_foot_before_recovery',reason=reason,time_s=now,ball_speed_mps=float(np.linalg.norm(vel)),physical_state_write=False))
                        else:
                            self.recovery_started=now;self.events.append(dict(kind='cartesian_standing_recovery' if motor_stand else 'learned_standing_recovery',reason=reason,time_s=now,physical_state_write=False))
                    if self.recovery_started is None and self.release_started is not None and now>=self.release_started+plan['recovery']['clearance_seconds']:
                        a0,b0,radii=self.contact_frames.capsules['right']
                        local_ball=actual_pose[:3,:3].T@(ball-actual_pose[:3,3]);segments=b0-a0
                        uu=np.clip(((local_ball-a0)*segments).sum(1)/(segments*segments).sum(1),0.,1.)
                        clearance=float((np.linalg.norm(local_ball-(a0+uu[:,None]*segments),axis=1)-radii-.11).min())
                        self.recovery_started=now;self.events.append(dict(kind='cartesian_standing_recovery' if motor_stand else 'learned_standing_recovery',reason=self.release_reason,time_s=now,ball_foot_clearance_m=clearance,physical_state_write=False))
                    if self.recovery_started is not None and not motor_stand:
                        u=min(1.,(now-self.recovery_started)/plan['recovery']['blend_seconds']);weight=.5-.5*math.cos(math.pi*u);self.phase='recover'
                feedback=np.array(plan['cushion'].get('velocity_effort_gains',[0.]*6))*(self.velocity_targets[0,self.ids]-actual_dq[self.ids])
                tau[self.ids]+=feedback
                lateral_gain=plan['cushion'].get('transverse_damping_Ns_m',0.) if self.motion.aerial else 0.
                lateral=np.cross(np.array([0.,0.,1.]),self.motion.normal)
                lateral/=max(1e-9,np.linalg.norm(lateral))
                lever=actual_pose[:3,:3]@np.array(plan['contact_point_local'])
                point_velocity=actual_velocity+np.cross(actual_angular,lever)
                desired_point_velocity=desired_velocity+np.cross(self.motion.angular_velocity,target[:3,:3]@np.array(plan['contact_point_local']))
                damping_force=lateral_gain*float((desired_point_velocity-point_velocity)@lateral)*lateral
                point_jacobian=jacobian[:3]+np.cross(jacobian[3:].T,lever).T
                if self.recovery_started is None:tau[self.ids]+=point_jacobian.T@damping_force
                self.control_trace[-1]['transverse_damping_force']=damping_force.tolist()
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
        args.source_provenance.update(inside_control_source_sha256=sha(__file__),inside_plan_sha256=sha(a.plan),reactive_scenario=a.scenario,
            incoming_ball_position=plan['ball_position'],incoming_ball_velocity=plan['ball_velocity'],
            inside_initialization='Common grounded stance; live ball observation triggers foot motion; ball initialized once',
            inside_control='Measured arrival prediction; continuous inside-foot withdrawal; ball-speed and spin gated recovery; normal joint actuators',
            receive_mode=plan['receive_mode'],incoming_ball_angular_velocity=plan.get('ball_angular_velocity',[0.,0.,0.]),
            inside_motion_sha256=sha(ROOT/'tools/inside_receive_motion.py'),support_balance_sha256=sha(ROOT/'tools/support_balance.py'),reactive_motion_sha256=sha(ROOT/'tools/reactive_receive_motion.py'),reactive_metrics_sha256=sha(ROOT/'tools/reactive_receive_metrics.py'),receiving_stance_sha256=sha(ROOT/'tools/receiving_stance.py'))
        try:
            value=native_run(args,points);result=json.loads(a.output.read_text());actor=created['actor']
            result['inside_sensor_trace']=actor.sensor_trace;result['inside_control_events']=actor.events
            result['inside_control_trace']=actor.control_trace;result['ball_predictions']=actor.motion.predictions;result['reactive_events']=actor.motion.events;result['reactive_resolved_plan']=plan
            result['inside_initialization_writes']=actor.initialization_writes
            result['inside_standing_conditioning_world']=actor.recovery_cues.cpu().tolist() if actor.recovery_cues is not None else None
            result['inside_final_recovery_weight']=actor.last_recovery_weight
            result['contact_evidence_version']=3
            result['legacy_shooting_records_applicable']=False
            from reactive_receive_metrics import summarize_reactive_receive
            result['inside_receive']=summarize_reactive_receive(result,plan['receive_mode'])
            a.output.write_text(json.dumps(result,indent=2));print('INSIDE_RECEIVE_RESULT',json.dumps(result['inside_receive']),flush=True)
            return value
        finally:gym.make=native_make;policy_module.OnnxSoccerPolicy=NativeActor;contact_module.contact_point_events=native_events;velocity_module.StrikeVelocityPulse=NativeVelocity
    base.run=aerial_run
    trial=dict(name='inside_'+plan['receive_mode']+'_trap_pilot',motion_file=str(ROOT/plan['reference']),ball=plan['ball_position'][:2],goal=[plan['ball_position'][0],plan['ball_position'][1]-3.7],goal_forward=[0,-1],style='inside',mode='hip')
    with tempfile.TemporaryDirectory(prefix='kinetickick-aerial-') as d:
        manifest=Path(d)/'trial.json';manifest.write_text(json.dumps([trial]))
        sys.argv=[str(ROOT/'tools/eval_free_shooting.py'),'--trials',str(manifest),'--output',str(a.output),'--sequence-evidence','--steps',str(steps),'--seed',str(plan['seed'])]
        if a.video_dir:sys.argv+=['--video-dir',str(a.video_dir)]
        base.main()

if __name__=='__main__':main()
