#!/usr/bin/env python3
"""Evaluate fitted state-conditioned placement using the unchanged physical recorder."""
import argparse,hashlib,importlib.util,json,math,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/state_conditioned_strike_v5.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ball-offset',nargs=2,type=float,default=[0.,0.])
    p.add_argument('--velocity-delta',type=float,default=0.)
    p.add_argument('--video-dir',type=Path)
    p.add_argument('--contact-diagnostics',action='store_true',help='Record non-foot collision bodies without changing termination')
    p.add_argument('--pair-dynamics-v055',action='store_true',help='Read native 200Hz G1 rigid-pair/wholebody physical diagnostics')
    p.add_argument('--native-readonly-pair-v056',action='store_true',help='Subscribe to existing PhysX collision reports without applying USD schema')
    p.add_argument('--perturb-reset-v056',action='store_true',help='Invoke upstream true seeded root and joint initial-condition perturbations')
    p.add_argument('--eval-seed',type=int,default=1201,help='Pre-registered PhysX seed; default original 1201')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if not all(math.isfinite(v) for v in [*a.ball_offset,a.velocity_delta]) or max(map(abs,a.ball_offset))>.15 or abs(a.velocity_delta)>.3:p.error('Perturbation outside bounded study')
    cfgdata=json.loads(a.config.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    from strike_baseline_protocol import verify_files
    verify_files(ROOT,cfgdata['frozen_files_sha256'])
    for k in ['template','reference','expert_plan']:
        if sha(ROOT/cfgdata[k])!=cfgdata[k+'_sha256']:raise ValueError('Changed '+k)
    spec=importlib.util.spec_from_file_location('_joint_strike_base',ROOT/'tools/eval_free_shooting.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);native_run=base.run

    def run_after_isaac(args,points):
        import gymnasium as gym
        import torch
        from isaaclab.envs import mdp as physical_mdp
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from running_balance import nonfoot_contact_termination
        if cfgdata.get('coordination_style_v041'):
            from coordination_style_v041 import get_task
        elif cfgdata.get('human_coordination_reference'):
            from coordination_reference_v040 import get_task
        elif cfgdata.get('foot_sweep_direction'):
            from foot_sweep_direction_v038 import get_task
        elif cfgdata.get('prestrike_feedback'):
            from prestrike_feedback import get_task
        elif cfgdata.get('early_path_feedback'):
            from early_path_controller import get_task
        elif cfgdata.get('post_shot_recovery'):
            from post_shot_recovery import get_task
        else:
            from placement_strike import get_task
        from contact_frame_sampler import ContactFrameSampler
        from physx_pair_dynamics_v055 import PhysxPairDynamicsObserver
        import soccer_onnx_policy,free_shooting_metrics,strike_velocity
        native_make=gym.make;NativeActor=soccer_onnx_policy.OnnxSoccerPolicy
        native_events=free_shooting_metrics.contact_point_events;NativePulse=strike_velocity.StrikeVelocityPulse
        created={}
        def make(*pos,**kw):
            cfg=kw['cfg'];cfg.joint_strike=cfgdata;cfg.scene.contact_forces.history_length=4
            cfg.commands.motion.velocity_range['y']=(a.velocity_delta,a.velocity_delta)
            cfg.terminations.ee_body_pos=None;cfg.terminations.anchor_pos_z=None;cfg.terminations.anchor_ori=None
            cfg.terminations.physical_height=DoneTerm(func=physical_mdp.root_height_below_minimum,params={'minimum_height':.45})
            cfg.terminations.physical_orientation=DoneTerm(func=physical_mdp.bad_orientation,params={'limit_angle':math.radians(60.)})
            cfg.terminations.physical_nonfoot=DoneTerm(func=nonfoot_contact_termination,params={'threshold':20.})
            env=native_make(*pos,**kw);created['env']=env.unwrapped;return env
        class Actor:
            def __init__(self,path,num_envs):
                if num_envs!=1:raise ValueError('Recorded physical evaluation uses one environment')
                self.env=created['env'];self.task=get_task(self.env);created['actor']=self
                self.observation_names=NativeActor(path,num_envs).observation_names
                self.nonfoot_events=[];self.state_inputs=[]
                self.contact_frames=ContactFrameSampler(self.env)
                self.pair_dynamics=None  # do not create native view until first stable step after env.reset
                self.pair_reader_v056=None
                native_update=self.env.scene.update
                def update(dt):
                    self.contact_frames.before_update();native_update(dt);self.contact_frames.after_update(dt)
                    self.task.after_physics(dt)
                    if self.pair_dynamics is not None:self.pair_dynamics.capture(dt)
                    if a.contact_diagnostics and dt>0:
                        sensor=self.env.scene['contact_forces'];force=sensor.data.net_forces_w[0]
                        for index,name in enumerate(sensor.body_names):
                            if name in ('left_ankle_roll_link','right_ankle_roll_link') or float(force[index].norm())<5.:continue
                            body=self.task.robot.body_names.index(name)
                            self.nonfoot_events.append(dict(time_s=float(self.env.sim.current_time),body=name,force_w_N=force[index].cpu().tolist(),position_w=self.task.robot.data.body_pos_w[0,body].cpu().tolist()))
                self.env.scene.update=update;self.first=True;self.initial=None
            def validate_joints(self,names):
                if list(names)!=self.task.meta['joint_names']:raise ValueError('Joint order changed')
            def reset(self):
                self.first=True
                if a.native_readonly_pair_v056 and self.pair_reader_v056 is None:
                    from native_contact_reader_v056 import G1ReadOnlyContactSubscriber
                    self.pair_reader_v056=G1ReadOnlyContactSubscriber(self.env)
                if a.pair_dynamics_v055 and self.pair_dynamics is None:
                    self.pair_dynamics=PhysxPairDynamicsObserver(self.env)
            def step(self,observations,frame):
                if self.first:
                    self.task.reset_fields(torch.tensor([0],device=self.task.device))
                    self.task.shift[:]=self.task.ball.data.root_pos_w[:,:2]-self.env.scene.env_origins[:,:2]-torch.tensor(cfgdata['ball_xy'],device=self.task.device)
                    self.initial=self.task.robot.data.root_lin_vel_w[0].cpu().tolist();self.first=False
                if a.video_dir:
                    ball=self.task.ball.data.root_pos_w.clone();root=self.task.robot.data.root_pos_w
                    ball[:,1]=ball[:,1].clamp_min(cfgdata['goal_xy'][1]-.5)
                    ball[:,0]=ball[:,0].clamp(cfgdata['goal_xy'][0]-1.,cfgdata['goal_xy'][0]+1.)
                    target=root.clone() if float(self.task.elapsed[0])*.02<1.3 else .5*(root+ball)
                    target[:,2]=.55
                    self.env.scene['review_camera'].set_world_poses_from_view(eyes=target+torch.tensor([[3.8,1.5,2.1]],device=self.task.device),targets=target)
                self.state_inputs.append(dict(time_s=float(self.task.elapsed[0])*.02,root_velocity_w=self.task.robot.data.root_lin_vel_w[0].cpu().tolist()))
                with torch.inference_mode():
                    actions=None if cfgdata.get('post_shot_recovery') else torch.zeros(1,30,device=self.task.device)
                    return self.task.commands(actions).cpu().numpy()
        class Pulse:
            def __init__(self,*pos,**kw):pass
            def apply(self,frames,support):return created['actor'].task.step_velocity.cpu().numpy()
        def events(*pos,**kw):return created['actor'].contact_frames.events(native_events,*pos,**kw)
        gym.make=make;soccer_onnx_policy.OnnxSoccerPolicy=Actor;strike_velocity.StrikeVelocityPulse=Pulse;free_shooting_metrics.contact_point_events=events
        try:
            value=native_run(args,points)
            result=json.loads(a.output.read_text());actor=created['actor']
            from closed_loop_acceptance import full_task_success
            from running_metrics import summarize_running
            from stride_shot_metrics import summarize_stride_shot
            run=summarize_running(result['traces'][0],initial_root=result['initial_robot_state']['root_position'][0],forward=[0,-1])
            stride=summarize_stride_shot(result['traces'][0],contact_time=result['records'][0]['contact_time_s'],running=run)
            result.update(running_evidence=run,stride_shot_evidence=stride,
                stride_shot_success=full_task_success(run,stride,result['sequence_records'][0],actor.task.expert.events),
                contact_evidence_version=3,contact_geometry='Sphere-side raw contact projected to capsule surface with signed separation, transformed using pre-integration foot pose; raw records retained')
            result['termination_policy']='complete_run_shot_height0.45m_tilt60deg_nonfoot20N'
            result['legacy_observation_metadata_policy']=dict(path=result['policy'],sha256=result['policy_sha256'])
            result['policy']=str(a.config.resolve())
            result['policy_sha256']=sha(a.config)
            result['joint_strike']=dict(mode='coordination_style_v041' if cfgdata.get('coordination_style_v041') else ('human_coordination_reference' if cfgdata.get('human_coordination_reference') else ('foot_sweep_direction' if cfgdata.get('foot_sweep_direction') else ('prestrike_feedback' if cfgdata.get('prestrike_feedback') else ('early_path_feedback' if cfgdata.get('early_path_feedback') else ('post_shot_residual' if cfgdata.get('post_shot_recovery') else 'fitted_placement'))))),
                checkpoint=cfgdata.get('post_shot_recovery',{}).get('checkpoint'),checkpoint_sha256=cfgdata.get('post_shot_recovery',{}).get('checkpoint_sha256'),
                prestrike_checkpoint=cfgdata.get('prestrike_feedback',{}).get('checkpoint'),prestrike_checkpoint_sha256=cfgdata.get('prestrike_feedback',{}).get('checkpoint_sha256'),
                template_sha256=cfgdata['template_sha256'],config=cfgdata,config_sha256=sha(a.config),
                source_sha256={n:sha(ROOT/'tools'/n) for n in ['run_placement_strike.py','placement_strike.py','closed_loop_strike.py','closed_loop_expert.py','joint_strike_core.py','paid_actor_warmstart.py','closed_loop_acceptance.py','hip_clearance.py','shank_clearance.py','approach_adapter.py']+(['landing_release.py'] if cfgdata.get('landing_release') else [])+(['support_reference_frame.py','support_reference_observation.py'] if cfgdata.get('support_reference_frame') else [])+(['post_shot_recovery.py','post_shot_objective.py'] if cfgdata.get('post_shot_recovery') else [])+(['early_path_controller.py','early_path_feedback.py'] if cfgdata.get('early_path_feedback') else [])+(['prestrike_feedback.py','prestrike_objective.py'] if cfgdata.get('prestrike_feedback') else [])+(['foot_sweep_direction_v038.py'] if cfgdata.get('foot_sweep_direction') else [])+(['coordination_reference_v040.py'] if cfgdata.get('human_coordination_reference') else [])+(['coordination_style_v041.py'] if cfgdata.get('coordination_style_v041') else [])},
                initial_root_velocity=actor.initial,ball_offset_xy=a.ball_offset,velocity_y_delta=a.velocity_delta,
                feedback=actor.task.debug,state_inputs=actor.state_inputs,actual_policy='Frozen feedback experts with fitted state-conditioned placement/phase, foot-pose IK and bounded early actuator adapter'+(' and measured post-contact landing release' if cfgdata.get('landing_release') else '')+(' and post-shot support-reference SE(2) handoff' if cfgdata.get('support_reference_frame') else '')+(' and shared learned post-shot whole-body residual' if cfgdata.get('post_shot_recovery') else '')+(' and fixed measured lateral path feedback' if cfgdata.get('early_path_feedback') else '')+(' and shared learned pre-strike whole-body residual' if cfgdata.get('prestrike_feedback') else '')+(' and fixed material-point sweep-velocity feedback (v38)' if cfgdata.get('foot_sweep_direction') else '')+(' and offline human-proxy 29DoF coordination reference (v40)' if cfgdata.get('human_coordination_reference') else '')+(' and independent nominal style activation (v41)' if cfgdata.get('coordination_style_v041') else ''),expert_backend=actor.task.expert.backend,expert_handoffs=actor.task.expert.events)
            if a.contact_diagnostics:result['nonfoot_contact_diagnostics']=actor.nonfoot_events
            if actor.pair_dynamics is not None:result['physx_pair_dynamics_v055']=actor.pair_dynamics.result()
            if actor.pair_reader_v056 is not None:result['physx_native_report_v056']=actor.pair_reader_v056.result()
            a.output.write_text(json.dumps(result,indent=2)+'\n')
            print('CLOSED_LOOP_STRIKE_RESULT',json.dumps(dict(success=result['stride_shot_success'],shot=result['records'][0],termination=result['termination_reasons'])),flush=True)
            return value
        finally:
            gym.make=native_make;soccer_onnx_policy.OnnxSoccerPolicy=NativeActor;strike_velocity.StrikeVelocityPulse=NativePulse;free_shooting_metrics.contact_point_events=native_events
    base.run=run_after_isaac
    trial=dict(name='state_conditioned_strike',motion_file=str(ROOT/cfgdata['reference']),
        ball=[v+d for v,d in zip(cfgdata['ball_xy'],a.ball_offset)],goal=cfgdata['goal_xy'],goal_forward=[0,-1],style='instep',mode='hip')
    with tempfile.TemporaryDirectory(prefix='kk-joint-strike-') as folder:
        path=Path(folder)/'trial.json';path.write_text(json.dumps([trial]))
        sys.argv=[str(ROOT/'tools/eval_free_shooting.py'),'--trials',str(path),'--output',str(a.output),'--sequence-evidence','--steps','700','--seed',str(a.eval_seed),'--style','instep']
        if a.video_dir:sys.argv+=['--video-dir',str(a.video_dir)]
        if a.perturb_reset_v056:sys.argv+=['--perturb-reset']
        base.main()

if __name__=='__main__':main()
