#!/usr/bin/env python3
"""Evaluate zero or learned residual control using the unchanged physical recorder."""
import argparse,hashlib,importlib.util,json,math,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/joint_strike_learning_v1.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--zero',action='store_true')
    p.add_argument('--ball-offset',nargs=2,type=float,default=[0.,0.])
    p.add_argument('--velocity-delta',type=float,default=0.)
    p.add_argument('--video-dir',type=Path)
    a=p.parse_args()
    if a.zero==bool(a.checkpoint):p.error('Choose exactly one of --zero or --checkpoint')
    if a.output.exists():raise FileExistsError(a.output)
    if not all(math.isfinite(v) for v in [*a.ball_offset,a.velocity_delta]) or max(map(abs,a.ball_offset))>.15 or abs(a.velocity_delta)>.3:p.error('Perturbation outside bounded study')
    cfgdata=json.loads(a.config.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    from strike_baseline_protocol import verify_files
    verify_files(ROOT,cfgdata['frozen_files_sha256'])
    for k in ['template','reference']:
        if sha(ROOT/cfgdata[k])!=cfgdata[k+'_sha256']:raise ValueError('Changed '+k)
    if a.checkpoint and not a.checkpoint.is_file():p.error('Checkpoint does not exist')
    spec=importlib.util.spec_from_file_location('_joint_strike_base',ROOT/'tools/eval_free_shooting.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);native_run=base.run

    def run_after_isaac(args,points):
        import gymnasium as gym
        import torch
        from isaaclab.envs import mdp as physical_mdp
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from running_balance import nonfoot_contact_termination
        from joint_strike_core import get_task,JointStrikeActor
        from contact_frame_sampler import ContactFrameSampler
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
                self.model=None
                if a.checkpoint:
                    nobs=self.task.features().shape[-1]
                    self.model=JointStrikeActor(nobs,nobs,30).to(self.task.device)
                    self.model.load_state_dict(torch.load(a.checkpoint,map_location=self.task.device,weights_only=False)['model_state_dict']);self.model.eval()
                self.contact_frames=ContactFrameSampler(self.env)
                native_update=self.env.scene.update
                def update(dt):
                    self.contact_frames.before_update();native_update(dt);self.contact_frames.after_update(dt)
                    self.task.after_physics(dt)
                self.env.scene.update=update;self.first=True;self.initial=None
            def validate_joints(self,names):
                if list(names)!=self.task.meta['joint_names']:raise ValueError('Joint order changed')
            def reset(self):self.first=True
            def step(self,observations,frame):
                if self.first:
                    self.task.reset_fields(torch.tensor([0],device=self.task.device))
                    self.task.shift[:]=self.task.ball.data.root_pos_w[:,:2]-self.env.scene.env_origins[:,:2]-torch.tensor(cfgdata['ball_xy'],device=self.task.device)
                    self.initial=self.task.robot.data.root_lin_vel_w[0].cpu().tolist();self.first=False
                with torch.inference_mode():
                    actions=torch.zeros(1,30,device=self.task.device) if a.zero else self.model.act_inference(self.task.features())
                    return self.task.commands(actions).cpu().numpy()
        class Pulse:
            def __init__(self,*pos,**kw):pass
            def apply(self,frames,support):return created['actor'].task.step_velocity.cpu().numpy()
        def events(*pos,**kw):return created['actor'].contact_frames.events(native_events,*pos,**kw)
        gym.make=make;soccer_onnx_policy.OnnxSoccerPolicy=Actor;strike_velocity.StrikeVelocityPulse=Pulse;free_shooting_metrics.contact_point_events=events
        try:
            value=native_run(args,points)
            result=json.loads(a.output.read_text());actor=created['actor']
            from running_metrics import summarize_running
            from stride_shot_metrics import summarize_stride_shot
            run=summarize_running(result['traces'][0],initial_root=result['initial_robot_state']['root_position'][0],forward=[0,-1])
            stride=summarize_stride_shot(result['traces'][0],contact_time=result['records'][0]['contact_time_s'],running=run)
            result.update(running_evidence=run,stride_shot_evidence=stride,
                stride_shot_success=bool(run['running_success'] and stride['continuous_running_contact'] and result['sequence_records'][0]['sequence_success']),
                contact_evidence_version=3,contact_geometry='Sphere-side raw contact projected to capsule surface with signed separation, transformed using pre-integration foot pose; raw records retained')
            result['termination_policy']='complete_run_shot_height0.45m_tilt60deg_nonfoot20N'
            result['legacy_observation_metadata_policy']=dict(path=result['policy'],sha256=result['policy_sha256'])
            result['policy']=str(a.checkpoint.resolve()) if a.checkpoint else cfgdata['template']
            result['policy_sha256']=sha(a.checkpoint) if a.checkpoint else cfgdata['template_sha256']
            result['joint_strike']=dict(mode='zero_reference' if a.zero else 'learned_residual',
                checkpoint=None if a.zero else str(a.checkpoint.resolve()),checkpoint_sha256=None if a.zero else sha(a.checkpoint),
                template_sha256=cfgdata['template_sha256'],config=cfgdata,config_sha256=sha(a.config),
                source_sha256={n:sha(ROOT/'tools'/n) for n in ['run_joint_strike.py','joint_strike_core.py']},
                initial_root_velocity=actor.initial,ball_offset_xy=a.ball_offset,velocity_y_delta=a.velocity_delta,
                feedback=actor.task.debug,actual_policy='Zero correction or PPO residual MLP over frozen actuator trajectory; legacy ONNX metadata only')
            a.output.write_text(json.dumps(result,indent=2)+'\n')
            print('JOINT_STRIKE_RESULT',json.dumps(dict(success=result['stride_shot_success'],shot=result['records'][0],termination=result['termination_reasons'])),flush=True)
            return value
        finally:
            gym.make=native_make;soccer_onnx_policy.OnnxSoccerPolicy=NativeActor;strike_velocity.StrikeVelocityPulse=NativePulse;free_shooting_metrics.contact_point_events=native_events
    base.run=run_after_isaac
    trial=dict(name='joint_strike_learning',motion_file=str(ROOT/cfgdata['reference']),
        ball=[v+d for v,d in zip(cfgdata['ball_xy'],a.ball_offset)],goal=cfgdata['goal_xy'],goal_forward=[0,-1],style='instep',mode='hip')
    with tempfile.TemporaryDirectory(prefix='kk-joint-strike-') as folder:
        path=Path(folder)/'trial.json';path.write_text(json.dumps([trial]))
        sys.argv=[str(ROOT/'tools/eval_free_shooting.py'),'--trials',str(path),'--output',str(a.output),'--sequence-evidence','--steps','700','--seed','1201','--style','instep']
        if a.video_dir:sys.argv+=['--video-dir',str(a.video_dir)]
        base.main()

if __name__=='__main__':main()
