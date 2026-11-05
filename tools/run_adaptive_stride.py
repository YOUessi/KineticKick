#!/usr/bin/env python3
"""Measured-state adaptive stride pilot with frozen physical acceptance rules.

A reference-index change is a command change. Robot/ball states are never written
after initialization. Existing foot/ball, goal, whole-body and running evidence
uses the existing acceptance rules with collision-time contact geometry.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=None);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1201);p.add_argument('--steps',type=int,default=700)
    p.add_argument('--video-dir',type=Path);p.add_argument('--blend-seconds',type=float,default=.1)
    p.add_argument('--style',choices=('instep','outside'),default='instep')
    p.add_argument('--ball-offset',type=float,nargs=2,default=[0.,0.])
    p.add_argument('--root-velocity-y-delta',type=float,default=0.)
    p.add_argument('--disable-feedback',action='store_true')
    a=p.parse_args()
    if not all(math.isfinite(v) for v in [*a.ball_offset,a.root_velocity_y_delta]) or max(map(abs,a.ball_offset))>.15 or abs(a.root_velocity_y_delta)>.3:
        p.error('Bounded initial perturbations: ball +/-0.15m, velocity +/-0.3m/s')
    if a.plan is None:a.plan=ROOT/'configs/adaptive_strike_v1.json'
    if not 1<=a.steps<=800:raise ValueError('Full-sequence pilot is bounded to800steps/16seconds')
    if a.output.exists():raise FileExistsError(a.output)
    if not .1<=a.blend_seconds<=.8:raise ValueError('Blend must be0.1..0.8seconds')
    plan=json.loads(a.plan.read_text())
    nominal_ball=list(plan['ball_xy'])
    plan['ball_xy']=[v+d for v,d in zip(nominal_ball,a.ball_offset)]
    window=plan['stride']['window_s']
    if (len(window)!=4 or not all(math.isfinite(v) for v in window)
            or not 0<=window[0]<window[1]<=window[2]<window[3]<3.):
        raise ValueError('Ordered, finite stride window before recovery required')
    for key in ('reference','running_checkpoint'):plan[key]=str((ROOT/plan[key]).resolve())
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if sha(ROOT/plan['feedback']['template'])!=plan['feedback']['template_sha256']:raise ValueError('Feedback calibration fingerprint mismatch')
    if sha(plan['reference'])!=plan['reference_sha256']:raise ValueError('Reference fingerprint mismatch')
    if plan.get('running_checkpoint_sha256') and sha(plan['running_checkpoint'])!=plan['running_checkpoint_sha256']:raise ValueError('Running checkpoint fingerprint mismatch')
    for name,digest in plan.get('controller_sha256',{}).items():
        if sha(ROOT/name)!=digest:raise ValueError('Controller changed since the recorded replay: '+name)
    spec=importlib.util.spec_from_file_location('_run_to_shot_base',ROOT/'tools/eval_free_shooting.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);native_run=base.run

    def run_after_isaac(args,points):
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab.utils.math import quat_apply_inverse,quat_apply
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from isaaclab.envs import mdp as physical_mdp
        import soccer_onnx_policy as soccer_module
        import free_shooting_metrics as contact_module
        from contact_frame_sampler import ContactFrameSampler
        native_events=contact_module.contact_point_events
        def aligned_events(*pos,**kw):
            return created['actor'].contact_frames.events(native_events,*pos,**kw)
        from running_balance import RunningBalanceActor,ground_force,support_observation,reference_contact,nonfoot_contact_termination
        from adaptive_strike import StrikeFeedback, retarget_swing_translation
        from cartesian_strike import LegKinematics
        from scipy.spatial.transform import Rotation
        import shooting_control, strike_velocity
        NativeResidual=shooting_control.StrikeResidual
        NativePulse=strike_velocity.StrikeVelocityPulse
        NativeActor=soccer_module.OnnxSoccerPolicy;native_make=gym.make;created={}

        class MeasuredResidual(NativeResidual):
            def __init__(self, entries, *aa, **kk):
                super().__init__(entries,*aa,**kk)
                self.is_strike=bool(entries and entries[0].get('targets'))
            def apply(self,actions,frames):
                actor=created.get('actor')
                if not self.is_strike or a.disable_feedback or actor is None:
                    return super().apply(actions,frames)
                adaptive_frames=np.array([actor.feedback_state['phase_s']/actor.env.step_dt+1.])
                result=super().apply(actions,adaptive_frames)
                weight=float(self.envelope(adaptive_frames)[0][0,0])
                if weight<=0 or actor.recovery_started is not None:return result
                root=np.eye(4);pose=actor.robot.data.root_state_w[0,:7].cpu().numpy()
                root[:3,:3]=Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix();root[:3,3]=pose[:3]
                ids=actor.leg_ids
                original=result[0,ids]*self.scale[0,ids]+self.offset[0,ids]
                target,_=actor.leg.forward(original,root)
                target[:2,3]+=np.asarray(actor.feedback_state['foot_delta_xy'])
                q,error=retarget_swing_translation(actor.leg,original,root,actor.feedback_state['foot_delta_xy'])
                actor.feedback_state['ik_error']=error
                if error>.025:
                    actor.feedback_state['ik_rejected']=True
                    return result
                correction=np.clip(q-original,-.25,.25)*weight
                modified=np.clip(original+correction,self.limits[0,ids,0],self.limits[0,ids,1])
                result[0,ids]=(modified-self.offset[0,ids])/self.scale[0,ids]
                actor.feedback_state['joint_correction_rad']=correction.tolist()
                return result

        class MeasuredPulse(NativePulse):
            def apply(self,frames,support):
                actor=created.get('actor')
                if not a.disable_feedback and actor is not None and actor.recovery_started is None:
                    frames=np.array([actor.feedback_state['phase_s']/actor.env.step_dt+1.])
                return super().apply(frames,support)

        def make(*pos,**kw):
            cfg=kw['cfg'];cfg.scene.contact_forces.history_length=4
            d=a.root_velocity_y_delta;cfg.commands.motion.velocity_range['y']=(d,d)
            cfg.terminations.ee_body_pos=None;cfg.terminations.anchor_pos_z=None;cfg.terminations.anchor_ori=None
            cfg.terminations.physical_height=DoneTerm(func=physical_mdp.root_height_below_minimum,params={'minimum_height':.45})
            cfg.terminations.physical_orientation=DoneTerm(func=physical_mdp.bad_orientation,params={'limit_angle':math.radians(60)})
            cfg.terminations.physical_nonfoot=DoneTerm(func=nonfoot_contact_termination,params={'threshold':20.})
            env=native_make(*pos,**kw);created['env']=env.unwrapped;return env

        class HybridActor:
            def __init__(self,path,num_envs):
                if num_envs!=1:raise ValueError('This integration pilot uses one physical environment')
                self.recovery=NativeActor(path,num_envs)
                self.swing=NativeActor(path,num_envs)
                self.reference=np.load(plan['reference'])
                self.observation_names=self.recovery.observation_names
                self.env=created['env'];self.robot=self.env.scene['robot'];self.motion=self.env.command_manager.get_term('motion')
                self.leg_ids=[self.robot.joint_names.index('right_'+n+'_joint') for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
                self.running=RunningBalanceActor(176,286,29,warmstart_path=plan['running_checkpoint']).to(self.robot.device)
                self.leg=LegKinematics(ROOT/'configs/paid_original/paid_main_resolved.urdf')
                template=json.loads((ROOT/plan['feedback']['template']).read_text())
                self.template=template
                self.running.eval();created['actor']=self;self.reset()
                # Sample the filtered force every physics step. A 5 ms impact
                # can disappear before the next 20 ms policy observation.
                self.contact_frames=ContactFrameSampler(self.env)
                scene_update=self.env.scene.update
                def update_and_latch(dt):
                    self.contact_frames.before_update()
                    scene_update(dt)
                    self.contact_frames.after_update(dt)
                    sensor=self.env.scene['contact_forces'];forces=sensor.data.net_forces_w[0].norm(dim=-1)
                    bodies={n:float(forces[i]) for i,n in enumerate(sensor.body_names)
                        if n not in ('left_ankle_roll_link','right_ankle_roll_link') and float(forces[i])>5.}
                    if bodies:self.nonfoot_events.append(dict(policy_step=self.step_count,forces_N=bodies))
                    forces=self.env.scene['right_ball_contact'].data.force_matrix_w
                    self.kicked |= bool(forces is not None and float(forces.norm(dim=-1).max())>2.)
                self.env.scene.update=update_and_latch
            def validate_joints(self,names):self.recovery.validate_joints(names)
            def reset(self):
                self.recovery.reset();self.swing.reset();self.running.reset(torch.ones(1,dtype=torch.bool,device=self.robot.device))
                self.step_count=0;self.ready_steps=0;self.recovery_started=None;self.events=[]
                self.feedback=StrikeFeedback(self.template['times'],self.template['root_xy'],nominal_ball,plan['feedback'])
                self.feedback_trace=[];self.feedback_state=None;self.nonfoot_events=[]
                self.recovery_goal_w=None;self.recovery_ball_w=None;self.last_running_action=None;self.kicked=False;self.last_visible_ball=None;self.perception_events=[]
            def running_obs(self,obs):
                from soccer.tasks.tracking import mdp
                x=torch.as_tensor(obs,dtype=torch.float32,device=self.robot.device).clone()
                for sl,y in [(slice(-6,-3),-8.),(slice(-3,None),-10.)]:
                    point=torch.tensor([[0.,y,.11]],device=self.robot.device)+self.env.scene.env_origins
                    x[:,sl]=quat_apply_inverse(self.robot.data.root_quat_w,point-self.robot.data.root_pos_w)
                anchor=mdp.motion_anchor_pos_b(self.env,'motion')
                if not a.disable_feedback and self.feedback_state is not None:
                    shift=torch.tensor([[*self.feedback_state['placement_xy'],0.]],device=self.robot.device,dtype=torch.float32)
                    u=min(1.,self.step_count*self.env.step_dt/1.2);shift*=u*u*(3.-2.*u)
                    if plan['feedback'].get('calibrated_approach_path',False):
                        nominal=torch.tensor(self.feedback_state['nominal_root_xy'],device=self.robot.device,dtype=torch.float32)
                        ref_root=self.motion.body_pos_w[0,0,:2]-self.env.scene.env_origins[0,:2]
                        shift[0,:2]+=(nominal-ref_root).clamp(-.4,.4)
                    anchor=anchor+quat_apply_inverse(self.motion.robot_anchor_quat_w,shift)
                return torch.cat([x,anchor,mdp.motion_anchor_ori_b(self.env,'motion'),
                    self.robot.data.root_lin_vel_b,support_observation(self.env),reference_contact(self.env)],dim=-1)
            def step(self,observations,frame):
                now=self.step_count*self.env.step_dt
                if self.step_count==0:
                    self.initial_root_velocity=self.robot.data.root_lin_vel_w[0].cpu().tolist()
                    # Match the trained running actor's one-frame reference lead.
                    self.motion._update_command()
                    observations=self.env.observation_manager.compute()['policy'].cpu().numpy()
                support=ground_force(self.env)[0]
                self.feedback_state=self.feedback.update(now,
                    (self.robot.data.root_pos_w[0,:2]-self.env.scene.env_origins[0,:2]).cpu().tolist(),
                    self.robot.data.root_lin_vel_w[0,:2].cpu().tolist(),
                    (self.env.scene['soccer_ball'].data.root_pos_w[0,:2]-self.env.scene.env_origins[0,:2]).cpu().tolist(),
                    float(support[0]),self.kicked)
                self.feedback_trace.append(self.feedback_state)
                strike_time=now if a.disable_feedback else self.feedback_state['phase_s']
                if not a.disable_feedback and plan['feedback'].get('synchronize_gait_phase',False) and self.recovery_started is None:
                    self.motion.time_steps[:]=min(round(strike_time/self.env.step_dt)+1,int(self.motion.motion_length[0])-1)
                    observations=self.env.observation_manager.compute()['policy'].cpu().numpy()
                if self.recovery_started is None:
                    speed=float(self.robot.data.root_lin_vel_w[0,:2].norm())
                    tilt=math.degrees(math.acos(float((-self.robot.data.projected_gravity_b[0,2]).clamp(-1,1))))
                    support=ground_force(self.env)[0]
                    distance=float((self.env.scene['soccer_ball'].data.root_pos_w[0,:2]-self.robot.data.root_pos_w[0,:2]).norm())
                    forward_speed=-float(self.robot.data.root_lin_vel_w[0,1])
                    gate=plan.get('recovery_gate',{})
                    ready=bool(now>=gate.get('earliest_s',3.) and float(self.motion.anchor_lin_vel_w[0,:2].norm())<.1
                        and speed<.45 and tilt<15 and float(support.min())>=30
                        and float(self.robot.data.root_pos_w[0,2])>=gate.get('minimum_root_height_m',0.))
                    self.ready_steps=self.ready_steps+1 if ready else 0
                    if self.ready_steps>=gate.get('consecutive_policy_steps',1):
                        self.recovery_started=now
                        self.events.append(dict(kind='post_shot_standing_recovery',time_s=now,root_xyz=self.robot.data.root_pos_w[0].cpu().tolist(),
                            root_speed_m_s=speed,root_tilt_deg=tilt,support_N=support.cpu().tolist(),ball_distance_m=distance,
                            command_frame_from=int(self.motion.time_steps[0]),forward_speed_m_s=forward_speed,command_frame_to=plan['shot_start_frame']+plan['recovery_entry_frame'],physical_state_write=False))
                        self.motion.time_steps[:]=plan['shot_start_frame']+plan['recovery_entry_frame']-1
                        self.motion._update_command()
                        observations=self.env.observation_manager.compute()['policy'].cpu().numpy()
                if self.recovery_started is None and (not self.kicked or distance<=2.):self.last_visible_ball=np.array(observations[:,-6:-3],copy=True)
                if a.video_dir:
                    ball=self.env.scene['soccer_ball'].data.root_pos_w.clone();root=self.robot.data.root_pos_w
                    ball[:,1]=ball[:,1].clamp_min(plan['goal_xy'][1]-.5)
                    ball[:,0]=ball[:,0].clamp(plan['goal_xy'][0]-1.,plan['goal_xy'][0]+1.)
                    target=root.clone()
                    if now>=plan['stride']['window_s'][0]:target=.5*(root+ball)
                    target[:,2]=.55
                    self.env.scene['review_camera'].set_world_poses_from_view(
                        eyes=target+torch.tensor([[3.8,1.5,2.1]],device=self.robot.device),targets=target)
                with torch.inference_mode():
                    if self.recovery_started is None:
                        running=self.running.act_inference(self.running_obs(observations)).cpu().numpy()
                        self.last_running_action=running.copy();action=running.copy()
                        entry=plan['stride'];start,full,release,end=entry['window_s']
                        if start<=strike_time<end:
                            frame=plan['shot_start_frame']+entry['entry_frame']+round((strike_time-start)/self.env.step_dt)
                            shot_obs=np.array(observations,copy=True)
                            shot_obs[:,:29]=self.reference['joint_pos'][frame]
                            shot_obs[:,29:58]=self.reference['joint_vel'][frame]
                            shot_obs[:,61:64]=self.reference['body_ang_vel_w'][frame,self.motion.robot_anchor_body_index]
                            swing=self.swing.step(shot_obs,frame-plan['shot_start_frame'])
                            phase=min((strike_time-start)/(full-start),1.) if strike_time<release else (end-strike_time)/(end-release)
                            weight=.5-.5*math.cos(math.pi*phase)
                            action[:,self.leg_ids]=(1-weight)*running[:,self.leg_ids]+weight*swing[:,self.leg_ids]
                    else:
                        running=self.last_running_action
                        shot_obs=np.array(observations,copy=True)
                        distance=float((self.env.scene['soccer_ball'].data.root_pos_w[0,:2]-self.robot.data.root_pos_w[0,:2]).norm())
                        if not self.kicked or distance<=2.:
                            self.last_visible_ball=shot_obs[:,-6:-3].copy()
                        elif self.last_visible_ball is not None:
                            # Use the upstream last-visible-target convention for a ball
                            # outside the actor's2m working range. The real ball is untouched.
                            shot_obs[:,-6:-3]=self.last_visible_ball
                            if not self.perception_events:self.perception_events.append(dict(time_s=now,kind='post_contact_ball_out_of_range',range_m=2.,actual_distance_m=distance,held_local_target=self.last_visible_ball.tolist()))
                        if self.kicked and distance>2. and plan.get('recovery_cues_local'):
                            # Once the shot has departed, request a stationary skill.
                            # These are explicit policy conditioning cues, not ball measurements.
                            if self.recovery_goal_w is None:
                                goal=torch.tensor([plan['recovery_cues_local']['goal']],device=self.robot.device)
                                self.recovery_goal_w=self.robot.data.root_pos_w.clone()+quat_apply(self.robot.data.root_quat_w,goal)
                                ball=torch.tensor([plan['recovery_cues_local']['ball']],device=self.robot.device)
                                self.recovery_ball_w=self.robot.data.root_pos_w.clone()+quat_apply(self.robot.data.root_quat_w,ball)
                                # A tilted torso must not turn a ground-target cue into a high ball.
                                ground_z=self.env.scene.env_origins[:,2]+.11
                                self.recovery_ball_w[:,2]=ground_z;self.recovery_goal_w[:,2]=ground_z
                                self.perception_events.append(dict(time_s=now,kind='stationary_virtual_world_targets',
                                    world_ball=self.recovery_ball_w.cpu().tolist(),world_goal=self.recovery_goal_w.cpu().tolist(),
                                    physical_ball_and_goal_unchanged=True))
                            shot_obs[:,-6:-3]=quat_apply_inverse(self.robot.data.root_quat_w,
                                self.recovery_ball_w-self.robot.data.root_pos_w).cpu().numpy()
                            shot_obs[:,-3:]=quat_apply_inverse(self.robot.data.root_quat_w,
                                self.recovery_goal_w-self.robot.data.root_pos_w).cpu().numpy()
                        shot=self.recovery.step(shot_obs,max(0,int(self.motion.time_steps[0])-plan['shot_start_frame']))
                        u=min((now-self.recovery_started)/a.blend_seconds,1.);weight=.5-.5*math.cos(math.pi*u)
                        action=shot if weight>=1 else (1-weight)*running+weight*shot
                self.step_count+=1
                return action

        gym.make=make;soccer_module.OnnxSoccerPolicy=HybridActor;contact_module.contact_point_events=aligned_events
        shooting_control.StrikeResidual=MeasuredResidual;strike_velocity.StrikeVelocityPulse=MeasuredPulse
        args.source_provenance.update(stride_shot_source_sha256=sha(__file__),running_checkpoint_sha256=sha(plan['running_checkpoint']),
            stride_shot_plan=str(a.plan.resolve()),stride_shot_plan_sha256=sha(a.plan),running_virtual_points_world=[[0,-8,.11],[0,-10,.11]],
            physical_ball_initialized_once=True,reference_switch_writes_physical_state=False,
            recovery_blend_seconds=a.blend_seconds,trial_steps=a.steps,
            recovery_conditioning_cues_local=plan.get('recovery_cues_local'),recovery_gate=plan.get('recovery_gate',{}),
            contact_frame_sampler_sha256=sha(ROOT/'tools/contact_frame_sampler.py'),
            stride_window_seconds=plan['stride']['window_s'],stride_control='Running actor continuously controls support and body; soccer actor blends only six right-leg actions',
            termination_policy='complete_run_shot_height0.45m_tilt60deg_nonfoot20N',
            shot_perception='Actual ball until legal-foot contact and2m departure; then last-visible local cue, physical ball still measured')
        try:
            value=native_run(args,points)
            result=json.loads(a.output.read_text());actor=created['actor']
            from running_metrics import summarize_running
            run=summarize_running(result['traces'][0],initial_root=result['initial_robot_state']['root_position'][0],forward=[0,-1])
            result['running_evidence']=run;result['recovery_handoff_events']=actor.events;result['physics_contact_latched']=actor.kicked;result['perception_events']=actor.perception_events
            from stride_shot_metrics import summarize_stride_shot
            stride=summarize_stride_shot(result['traces'][0],contact_time=result['records'][0]['contact_time_s'],running=run)
            result['stride_shot_evidence']=stride
            result['stride_shot_success']=bool(run['running_success'] and actor.events and stride['continuous_running_contact'] and result['sequence_records'][0]['sequence_success'])
            result['contact_evidence_version']=3
            result['contact_geometry']='Sphere-side raw contact projected to capsule surface with signed separation, transformed using pre-integration foot pose; raw records retained'
            result['adaptive_control']=dict(enabled=not a.disable_feedback,feedback=actor.feedback_trace,
                nonfoot_events=actor.nonfoot_events,initial_ball_offset_xy=a.ball_offset,
                initial_root_velocity_y_delta=a.root_velocity_y_delta,nominal_ball_xy=nominal_ball,
                observed_initial_root_velocity=actor.initial_root_velocity,
                source_sha256={n:sha(ROOT/'tools'/n) for n in ['run_adaptive_stride.py','adaptive_strike.py']},
                template_sha256=sha(ROOT/plan['feedback']['template']),resolved_plan=plan,
                interpretation='Learned running policy with explicit measured-state placement/phase feedback; no learned human coordination claim')
            result['scope']=plan['scope'];a.output.write_text(json.dumps(result,indent=2))
            print('STRIDE_SHOT_RESULT',json.dumps(dict(success=result['stride_shot_success'],stride=stride,running=run,
                handoff=actor.events,shot=result['records'][0],sequence=result['sequence_records'][0])),flush=True)
            return value
        finally:
            gym.make=native_make;soccer_module.OnnxSoccerPolicy=NativeActor;contact_module.contact_point_events=native_events
            shooting_control.StrikeResidual=NativeResidual;strike_velocity.StrikeVelocityPulse=NativePulse
    base.run=run_after_isaac
    trial=dict(name='run_stride_'+plan['style']+'_shot',motion_file=plan['reference'],ball=plan['ball_xy'],goal=plan['goal_xy'],
        policy_goal=plan.get('policy_goal_xy',plan['goal_xy']),style=plan['style'],mode=plan['mode'],goal_forward=plan.get('goal_forward',[0,-1]),strike=plan.get('strike',{}),strike_velocity=plan.get('strike_velocity'))
    with tempfile.TemporaryDirectory(prefix='kinetickick-run-shot-') as d:
        manifest=Path(d)/'trial.json';manifest.write_text(json.dumps([trial]))
        sys.argv=[str(ROOT/'tools/eval_free_shooting.py'),'--trials',str(manifest),'--output',str(a.output),
            '--sequence-evidence','--steps',str(a.steps),'--seed',str(a.seed),'--style',plan['style']]
        if a.video_dir:sys.argv+=['--video-dir',str(a.video_dir)]
        base.main()


if __name__=='__main__':main()
