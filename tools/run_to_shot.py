#!/usr/bin/env python3
"""One physical episode: running actor -> actual-ball soccer actor -> recovery.

A reference-index change is a command change. Robot/ball states are never written
after initialization. Existing foot/ball, goal, whole-body and running evidence
is captured by the unchanged physical shooting evaluator.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=ROOT/'configs/run_to_shot_v1.json');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1201);p.add_argument('--steps',type=int,default=700)
    p.add_argument('--video-dir',type=Path);p.add_argument('--blend-seconds',type=float,default=.4)
    a=p.parse_args()
    if not 1<=a.steps<=800:raise ValueError('Full-sequence pilot is bounded to800steps/16seconds')
    if a.output.exists():raise FileExistsError(a.output)
    if not .1<=a.blend_seconds<=.8:raise ValueError('Blend must be0.1..0.8seconds')
    plan=json.loads(a.plan.read_text())
    for key in ('reference','running_checkpoint'):plan[key]=str((ROOT/plan[key]).resolve())
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
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
        from isaaclab.utils.math import quat_apply_inverse
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from isaaclab.envs import mdp as physical_mdp
        import soccer_onnx_policy as soccer_module
        from running_balance import RunningBalanceActor,ground_force,support_observation,reference_contact,nonfoot_contact_termination
        NativeActor=soccer_module.OnnxSoccerPolicy;native_make=gym.make;created={}
        def make(*pos,**kw):
            cfg=kw['cfg'];cfg.scene.contact_forces.history_length=4
            cfg.terminations.ee_body_pos=None;cfg.terminations.anchor_pos_z=None;cfg.terminations.anchor_ori=None
            cfg.terminations.physical_height=DoneTerm(func=physical_mdp.root_height_below_minimum,params={'minimum_height':.45})
            cfg.terminations.physical_orientation=DoneTerm(func=physical_mdp.bad_orientation,params={'limit_angle':math.radians(60)})
            cfg.terminations.physical_nonfoot=DoneTerm(func=nonfoot_contact_termination,params={'threshold':20.})
            env=native_make(*pos,**kw);created['env']=env.unwrapped;return env

        class HybridActor:
            def __init__(self,path,num_envs):
                if num_envs!=1:raise ValueError('This integration pilot uses one physical environment')
                self.soccer=NativeActor(path,num_envs)
                self.observation_names=self.soccer.observation_names
                self.env=created['env'];self.robot=self.env.scene['robot'];self.motion=self.env.command_manager.get_term('motion')
                self.running=RunningBalanceActor(176,286,29,warmstart_path=plan['running_checkpoint']).to(self.robot.device)
                self.running.eval();created['actor']=self;self.reset()
            def validate_joints(self,names):self.soccer.validate_joints(names)
            def reset(self):
                self.soccer.reset();self.running.reset(torch.ones(1,dtype=torch.bool,device=self.robot.device))
                self.step_count=0;self.ready_steps=0;self.switched=None;self.events=[]
                self.kicked=False;self.last_visible_ball=None;self.perception_events=[]
            def running_obs(self,obs):
                from soccer.tasks.tracking import mdp
                x=torch.as_tensor(obs,dtype=torch.float32,device=self.robot.device).clone()
                for sl,y in [(slice(-6,-3),-8.),(slice(-3,None),-10.)]:
                    point=torch.tensor([[0.,y,.11]],device=self.robot.device)+self.env.scene.env_origins
                    x[:,sl]=quat_apply_inverse(self.robot.data.root_quat_w,point-self.robot.data.root_pos_w)
                return torch.cat([x,mdp.motion_anchor_pos_b(self.env,'motion'),mdp.motion_anchor_ori_b(self.env,'motion'),
                    self.robot.data.root_lin_vel_b,support_observation(self.env),reference_contact(self.env)],dim=-1)
            def step(self,observations,frame):
                now=self.step_count*self.env.step_dt
                if self.step_count==0:
                    # Match the trained running actor's one-frame reference lead.
                    self.motion._update_command()
                    observations=self.env.observation_manager.compute()['policy'].cpu().numpy()
                if self.switched is None:
                    speed=float(self.robot.data.root_lin_vel_w[0,:2].norm())
                    tilt=math.degrees(math.acos(float((-self.robot.data.projected_gravity_b[0,2]).clamp(-1,1))))
                    support=ground_force(self.env)[0]
                    distance=float((self.env.scene['soccer_ball'].data.root_pos_w[0,:2]-self.robot.data.root_pos_w[0,:2]).norm())
                    ready=bool(now>=3 and float(self.motion.anchor_lin_vel_w[0,:2].norm())<.1
                        and speed<.45 and tilt<15 and float(support.min())>=30 and .75<=distance<=1.5)
                    self.ready_steps=self.ready_steps+1 if ready else 0
                    if self.ready_steps>=4:
                        self.switched=now
                        self.events.append(dict(time_s=now,root_xyz=self.robot.data.root_pos_w[0].cpu().tolist(),
                            root_speed_m_s=speed,root_tilt_deg=tilt,support_N=support.cpu().tolist(),ball_distance_m=distance,
                            command_frame_from=int(self.motion.time_steps[0]),command_frame_to=plan['shot_start_frame'],physical_state_write=False))
                        self.motion.time_steps[:]=plan['shot_start_frame']-1
                        self.motion._update_command()
                        observations=self.env.observation_manager.compute()['policy'].cpu().numpy()
                if a.video_dir:
                    ball=self.env.scene['soccer_ball'].data.root_pos_w.clone();root=self.robot.data.root_pos_w
                    ball[:,1]=ball[:,1].clamp_min(plan['goal_xy'][1]-.5)
                    ball[:,0]=ball[:,0].clamp(plan['goal_xy'][0]-1.,plan['goal_xy'][0]+1.)
                    target=root.clone()
                    if self.switched is not None:target=.5*(root+ball)
                    target[:,2]=.55
                    self.env.scene['review_camera'].set_world_poses_from_view(
                        eyes=target+torch.tensor([[3.8,1.5,2.1]],device=self.robot.device),targets=target)
                with torch.inference_mode():
                    if self.switched is None or now-self.switched<a.blend_seconds:
                        running=self.running.act_inference(self.running_obs(observations)).cpu().numpy()
                    if self.switched is None:action=running
                    else:
                        shot_obs=np.array(observations,copy=True)
                        forces=self.env.scene['right_ball_contact'].data.force_matrix_w
                        self.kicked |= bool(forces is not None and float(forces.norm(dim=-1).max())>2.)
                        distance=float((self.env.scene['soccer_ball'].data.root_pos_w[0,:2]-self.robot.data.root_pos_w[0,:2]).norm())
                        if not self.kicked or distance<=2.:
                            self.last_visible_ball=shot_obs[:,-6:-3].copy()
                        elif self.last_visible_ball is not None:
                            # Use the upstream last-visible-target convention for a ball
                            # outside the actor's2m working range. The real ball is untouched.
                            shot_obs[:,-6:-3]=self.last_visible_ball
                            if not self.perception_events:self.perception_events.append(dict(time_s=now,kind='post_contact_ball_out_of_range',range_m=2.,actual_distance_m=distance,held_local_target=self.last_visible_ball.tolist()))
                        shot=self.soccer.step(shot_obs,max(0,int(self.motion.time_steps[0])-plan['shot_start_frame']))
                        u=min((now-self.switched)/a.blend_seconds,1.);weight=.5-.5*math.cos(math.pi*u)
                        action=shot if weight>=1 else (1-weight)*running+weight*shot
                self.step_count+=1
                return action

        gym.make=make;soccer_module.OnnxSoccerPolicy=HybridActor
        args.source_provenance.update(run_to_shot_source_sha256=sha(__file__),running_checkpoint_sha256=sha(plan['running_checkpoint']),
            run_to_shot_plan=str(a.plan.resolve()),run_to_shot_plan_sha256=sha(a.plan),running_virtual_points_world=[[0,-8,.11],[0,-10,.11]],
            physical_ball_initialized_once=True,reference_switch_writes_physical_state=False,
            handoff_blend_seconds=a.blend_seconds,trial_steps=a.steps,
            termination_policy='complete_run_shot_height0.45m_tilt60deg_nonfoot20N',
            shot_perception='Actual ball until legal-foot contact and2m departure; then last-visible local cue, physical ball still measured')
        try:
            value=native_run(args,points)
            result=json.loads(a.output.read_text());actor=created['actor']
            from running_metrics import summarize_running
            run=summarize_running(result['traces'][0],initial_root=result['initial_robot_state']['root_position'][0],forward=[0,-1])
            result['running_evidence']=run;result['handoff_events']=actor.events;result['perception_events']=actor.perception_events
            result['run_to_shot_success']=bool(run['running_success'] and actor.events and result['sequence_records'][0]['sequence_success'])
            result['scope']=plan['scope'];a.output.write_text(json.dumps(result,indent=2))
            print('RUN_TO_SHOT_RESULT',json.dumps(dict(success=result['run_to_shot_success'],running=run,
                handoff=actor.events,shot=result['records'][0],sequence=result['sequence_records'][0])),flush=True)
            return value
        finally:gym.make=native_make;soccer_module.OnnxSoccerPolicy=NativeActor
    base.run=run_after_isaac
    trial=dict(name='run_then_inside_shot',motion_file=plan['reference'],ball=plan['ball_xy'],goal=plan['goal_xy'],
        policy_goal=plan['goal_xy'],style='inside',mode='hip',goal_forward=[0,-1],strike={})
    with tempfile.TemporaryDirectory(prefix='kinetickick-run-shot-') as d:
        manifest=Path(d)/'trial.json';manifest.write_text(json.dumps([trial]))
        sys.argv=[str(ROOT/'tools/eval_free_shooting.py'),'--trials',str(manifest),'--output',str(a.output),
            '--sequence-evidence','--steps',str(a.steps),'--seed',str(a.seed),'--style','inside']
        if a.video_dir:sys.argv+=['--video-dir',str(a.video_dir)]
        base.main()


if __name__=='__main__':main()
