#!/usr/bin/env python3
"""One bounded prestrike PPO trial over the frozen v32 post-shot recovery model.

Model selection is the final finite checkpoint at the declared time/update
budget. Training reset proxies never replace the independent 14 s evaluator.
"""
import argparse,hashlib,json,math,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

FROZEN_RECOVERY_SHA256='ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326'


def validate_checkpoint_config(config):
    early=config.get('prestrike_feedback');recovery=config.get('post_shot_recovery')
    if not isinstance(early,dict) or not isinstance(recovery,dict):
        raise ValueError('Separate prestrike and frozen post-shot configurations are required')
    if config.get('support_reference_frame'):
        raise ValueError('Original v6 spatial references are required')
    if early.get('checkpoint') is not None or early.get('checkpoint_sha256') is not None:
        raise ValueError('The new prestrike actor must start without a checkpoint')
    if not recovery.get('checkpoint') or recovery.get('checkpoint_sha256')!=FROZEN_RECOVERY_SHA256:
        raise ValueError('The original pinned v32 recovery checkpoint must remain enabled')


def validate_development_cases(config,cases,declared_cases):
    names=config.get('prestrike_training',{}).get('case_ids')
    declared={case['id']:case for case in declared_cases}
    if (not isinstance(names,list) or len(names)!=18 or not all(isinstance(name,str) for name in names)
            or len(declared_cases)!=18 or len(declared)!=18 or len(set(names))!=18 or set(names)!=set(declared)):
        raise ValueError('Use exactly the declared v32 old18 development cases; no fresh or duplicate entries')
    if any(cases.get(name)!=declared[name] for name in names):
        raise ValueError('A declared development condition changed; no jitter or implicit case expansion')
    return names


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=256)
    p.add_argument('--iters',type=int,default=400)
    p.add_argument('--max-seconds',type=float,default=600.)
    p.add_argument('--seed',type=int,default=3501)
    a=p.parse_args()
    if not (1<=a.num_envs<=256 and 1<=a.iters<=400 and math.isfinite(a.max_seconds) and 0<a.max_seconds<=600.):p.error('Bounded first trial: <=256 envs, <=400 updates, <=600 optimizer/rollout seconds')
    if a.output.exists():raise FileExistsError(a.output)
    config=json.loads(a.config.read_text())
    validate_checkpoint_config(config)
    from strike_baseline_protocol import verify_files
    verify_files(ROOT,config['frozen_files_sha256'])
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(ROOT/config['post_shot_recovery']['checkpoint'])!=FROZEN_RECOVERY_SHA256:
        raise ValueError('Frozen v32 recovery checkpoint file SHA256 mismatch')
    for key in ['template','reference','expert_plan']:
        if sha(ROOT/config[key])!=config[key+'_sha256']:raise ValueError('Changed '+key)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True).app
    try:train(a,config,sha)
    except BaseException:
        import traceback
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();raise
    finally:app.close()


def train(a,config,sha):
    import gymnasium as gym
    import torch
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.managers import ObservationGroupCfg,ObservationTermCfg as ObsTerm,RewardTermCfg as RewTerm,TerminationTermCfg as DoneTerm
    from isaaclab.utils import configclass
    from isaaclab.envs import mdp as physical_mdp
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import rsl_rl.runners.on_policy_runner as runners
    from prestrike_feedback import get_task,task_features,task_reward,PreStrikeActor,FEATURE_WIDTH,FEATURE_LAYOUT
    from running_balance import nonfoot_contact_termination
    upstream=ROOT/'third_party/HumanoidSoccer/source/whole_body_tracking';sys.path.insert(0,str(upstream))
    shim=types.ModuleType('soccer.assets');shim.ASSET_DIR=str(upstream/'soccer/assets');sys.modules['soccer.assets']=shim
    import soccer.tasks
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    @configclass
    class StateObs(ObservationGroupCfg):
        state=ObsTerm(func=task_features)
        enable_corruption=False
        concatenate_terms=True
    cfg=G1FlatEnvCfg();cfg.joint_strike=config;cfg.scene.num_envs=a.num_envs;cfg.scene.env_spacing=8.
    cfg.seed=a.seed;cfg.sim.device='cuda:0';cfg.episode_length_s=7.
    cfg.scene.robot.spawn.asset_path=str(ROOT/'configs/paid_original/paid_main_resolved.urdf');cfg.scene.robot.spawn.fix_base=False
    cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn=sim.DomeLightCfg(intensity=1100)
    cfg.scene.contact_forces.debug_vis=False;cfg.scene.contact_forces.history_length=4
    cfg.commands.motion.motion_files=[str(ROOT/config['reference'])];cfg.commands.motion.debug_vis=False
    for key in cfg.commands.motion.pose_range:cfg.commands.motion.pose_range[key]=(0.,0.)
    for key in cfg.commands.motion.velocity_range:cfg.commands.motion.velocity_range[key]=(0.,0.)
    cfg.commands.motion.joint_position_range=(0.,0.)
    for key in ['physics_material','add_joint_default_pos','base_com','base_external_force_torque','push_robot']:setattr(cfg.events,key,None)
    cfg.scene.soccer_ball=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/SoccerBall',
        spawn=sim.SphereCfg(radius=.11,activate_contact_sensors=True,collision_props=sim.CollisionPropertiesCfg(),
            rigid_props=sim.RigidBodyPropertiesCfg(),mass_props=sim.MassPropertiesCfg(mass=.43),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.94,.94,.90)),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=.7,dynamic_friction=.55,restitution=.3)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.8,2.8,.113)))
    for side in ['left','right']:
        setattr(cfg.scene,side+'_ball_contact',ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+side+'_ankle_roll_link',update_period=0.,history_length=1,filter_prim_paths_expr=['{ENV_REGEX_NS}/SoccerBall'],debug_vis=False))
    cfg.observations.policy=StateObs();cfg.observations.critic=StateObs()
    for key,value in vars(cfg.rewards).items():
        if isinstance(value,RewTerm):setattr(cfg.rewards,key,None)
    cfg.rewards.joint_strike=RewTerm(func=task_reward,weight=1.)
    cfg.terminations.ee_body_pos=None;cfg.terminations.anchor_pos_z=None;cfg.terminations.anchor_ori=None
    cfg.terminations.physical_height=DoneTerm(func=physical_mdp.root_height_below_minimum,params={'minimum_height':.45})
    cfg.terminations.physical_orientation=DoneTerm(func=physical_mdp.bad_orientation,params={'limit_angle':math.radians(60.)})
    cfg.terminations.physical_nonfoot=DoneTerm(func=nonfoot_contact_termination,params={'threshold':20.})
    budget_start=[None]
    class BudgetReached(Exception):pass
    def check_budget():
        if budget_start[0] is None:return
        if (a.output/'STOP').exists():raise BudgetReached('diagnostic_stop')
        if time.monotonic()-budget_start[0]>=a.max_seconds:raise BudgetReached('wall_time_limit')
    class ResidualEnv(RslRlVecEnvWrapper):
        def step(self,actions):
            check_budget()
            task=get_task(self.unwrapped);physical=task.commands(actions)
            task.robot.set_joint_velocity_target(task.step_velocity)
            obs,reward,done,extras=super().step(physical)
            if 'time_outs' in extras:
                extras['time_outs']=extras['time_outs'] & ~self.unwrapped.termination_manager.terminated
            return obs,reward,done,extras
    env=ResidualEnv(gym.make('Tracking-Flat-G1-v0',cfg=cfg),clip_actions=None);env.num_actions=29
    try:
        from strike_cases import load_cases
        cases=load_cases(ROOT)
        development_manifest=ROOT/'configs/post_shot_recovery_study_v1.json'
        declared_cases=json.loads(development_manifest.read_text())['existing_cases']
        names=validate_development_cases(config,cases,declared_cases)
        contexts=torch.tensor([[*cases[name]['ball_offset_xy_m'],cases[name]['root_velocity_y_delta_mps']] for name in names],device='cuda:0')
        nominal_index=names.index('center');other_indices=torch.tensor([i for i,name in enumerate(names) if name!='center'],device='cuda:0')
        raw=env.unwrapped;task=get_task(raw);motion=raw.command_manager.get_term('motion')
        assert list(raw.reward_manager.active_terms)==['joint_strike']
        if abs(raw.physics_dt-.005)>1e-9 or abs(raw.step_dt-.02)>1e-9:raise ValueError('Original physical/control intervals required')
        native_update=motion._update_command
        def exact_reference(command,ids):
            command.motion_idx[ids]=0;command.motion_length[ids]=command.motion.file_lengths[0];command.time_steps[ids]=0
        motion._uniform_sampling=types.MethodType(exact_reference,motion)
        def no_wrap(command):
            command.time_steps[:]=torch.minimum(command.time_steps,command.motion_length-2)
            return native_update()
        motion._update_command=types.MethodType(no_wrap,motion)
        scene_update=raw.scene.update
        episode_qd_peak=torch.zeros(a.num_envs,device=task.device)
        def physics(dt):
            scene_update(dt);task.after_physics(dt)
            if dt>0:
                ratio=(task.robot.data.joint_vel.abs()/task.robot.data.joint_vel_limits).amax(-1)
                episode_qd_peak.copy_(torch.maximum(episode_qd_peak,ratio))
        raw.scene.update=physics
        native_reset=raw._reset_idx
        counts=torch.zeros(7,device=task.device);sampled_case=torch.zeros(a.num_envs,device=task.device,dtype=torch.long)
        reset_generator=torch.Generator(device=task.device).manual_seed(a.seed+1)
        def reset(ids):
            ids=torch.as_tensor(ids,device=task.device,dtype=torch.long)
            if not len(ids):return
            valid=task.elapsed[ids]>1
            nominal=sampled_case[ids]==nominal_index
            r=task.robot.data
            from running_balance import ground_force
            stable=(r.root_pos_w[:,2]-raw.scene.env_origins[:,2]>.55)&(r.root_lin_vel_w[:,:2].norm(dim=-1)<.3)&(-r.projected_gravity_b[:,2]>.9063)&(ground_force(raw).amin(-1)>30.)
            counts.add_(torch.stack([valid.sum(),(valid&nominal).sum(),(valid&task.contact[ids]).sum(),(valid&task.scored[ids]).sum(),(valid&raw.termination_manager.terminated[ids]).sum(),(valid&(episode_qd_peak[ids]>1.+1e-6)).sum(),(valid&stable[ids]&~raw.termination_manager.terminated[ids]).sum()]))
            native_reset(ids);task.reset_fields(ids);episode_qd_peak[ids]=0.
            choice=other_indices[torch.randint(len(other_indices),(len(ids),),device=task.device,generator=reset_generator)]
            choice[ids%4==0]=nominal_index;sampled_case[ids]=choice
            condition=contexts[choice];task.shift[ids]=condition[:,:2]
            state=task.ball.data.default_root_state[ids].clone();state[:,7:]=0.
            state[:,:2]=torch.tensor(config['ball_xy'],device=task.device)+condition[:,:2]+raw.scene.env_origins[ids,:2]
            state[:,2]=.113+raw.scene.env_origins[ids,2]
            task.ball.write_root_state_to_sim(state,env_ids=ids);task.ball.reset(ids);task.previous_ball[ids]=state[:,:3]
            velocity=task.robot.data.root_state_w[ids,7:13].clone();velocity[:,1]+=condition[:,2]
            task.robot.write_root_velocity_to_sim(velocity,env_ids=ids)
        raw._reset_idx=reset
        env.reset();obs,extra=env.get_observations()
        if obs.shape!=(a.num_envs,FEATURE_WIDTH) or not bool(torch.isfinite(obs).all()):raise ValueError('Invalid initial prestrike features')
        runner_cfg=G1FlatPPORunnerCfg().to_dict();runner_cfg.update(num_steps_per_env=64,save_interval=400,empirical_normalization=False)
        runner_cfg['policy']=dict(class_name='PreStrikeActor',init_noise_std=.25)
        runner_cfg['algorithm'].update(learning_rate=1e-4,schedule='fixed',num_learning_epochs=4,num_mini_batches=4,entropy_coef=.001,gamma=.995,lam=.97)
        runners.PreStrikeActor=PreStrikeActor
        a.output.mkdir(parents=True)
        source_names=['train_prestrike_feedback.py','prestrike_feedback.py','prestrike_objective.py','post_shot_recovery.py','post_shot_objective.py','placement_strike.py','closed_loop_strike.py','closed_loop_expert.py','joint_strike_core.py','paid_actor_warmstart.py','hip_clearance.py','shank_clearance.py','approach_adapter.py','landing_release.py']
        manifest=dict(version='0.1.35',config=config,config_sha256=sha(a.config),seed=a.seed,num_envs=a.num_envs,
            max_seconds=a.max_seconds,max_updates=a.iters,training_episode_seconds=7.,physics_dt=raw.physics_dt,policy_dt=raw.step_dt,
            actor_observations=FEATURE_WIDTH,feature_layout=FEATURE_LAYOUT,actor_actions=29,sources={name:sha(ROOT/'tools'/name) for name in source_names},
            initial_policy='Zero-mean bounded prestrike residual; original expert and pinned v32 post-shot actor invoked by PreStrikeTask once per policy step',
            frozen_recovery_checkpoint_sha256=FROZEN_RECOVERY_SHA256,development_manifest_sha256=sha(development_manifest),
            training_contexts=[cases[name] for name in names],nominal_fraction=.25,context_sampling='Remaining environments uniformly sample the 17 already observed nonnominal cases at reset; no fresh landing_eval cases',
            physical_writes='Robot/ball initialization at episode resets only; no mid-episode ball impulse or pose reset',
            selection='Final finite checkpoint at the one declared wall-time/update limit; no per-case selection or best-reward sweep',
            teacher_backend=task.expert.backend,backend_scope='Batched frozen experts use Torch; independent single-environment evaluator retains original auto/ONNX soccer expert',
            runner=runner_cfg,proxy_metrics='Reset counts and instantaneous terminal stability are not physical full-task acceptance')
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for name in source_names:(a.output/name).write_bytes((ROOT/'tools'/name).read_bytes())
        runner=runners.OnPolicyRunner(env,runner_cfg,log_dir=str(a.output),device=task.device)
        start=time.monotonic();budget_start[0]=start;native_alg_update=runner.alg.update;updates=[0];stop='iteration_limit'
        def finite_policy():
            policy=runner.alg.policy
            finite=all(bool(torch.isfinite(value).all()) for value in policy.state_dict().values())
            if getattr(policy,'noise_std_type',None)=='scalar':finite &= bool((policy.std>0.).all())
            return finite
        def update():
            check_budget()
            answer=native_alg_update();updates[0]+=1
            if not finite_policy():raise FloatingPointError('Nonfinite learned state or nonpositive exploration std')
            if updates[0]%10==0:
                progress=dict(updates=updates[0],training_seconds=time.monotonic()-start,reset_counts=counts.tolist(),latest_step_peak_qd_ratio=float(episode_qd_peak.max()))
                (a.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            return answer
        runner.alg.update=update
        try:runner.learn(num_learning_iterations=a.iters,init_at_random_ep_len=False)
        except BudgetReached as exc:stop=str(exc)
        except BaseException as exc:
            failure=dict(status='failed',error_type=type(exc).__name__,error=str(exc),training_seconds=time.monotonic()-start,
                updates=updates[0],finite_policy=finite_policy(),candidate_checkpoint=None)
            (a.output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
            runner.save(str(a.output/'failed_state.pt'))
            raise
        if not finite_policy():raise FloatingPointError('Final checkpoint failed finite/positive-std contract')
        runner.save(str(a.output/'last_model.pt'))
        result=dict(stop=stop,training_seconds=time.monotonic()-start,budget_overrun_seconds=max(0.,time.monotonic()-start-a.max_seconds),budget_boundary='Checked before every policy step and optimizer update; one already executing call and final serialization are not preempted',updates=updates[0],checkpoint=str(a.output/'last_model.pt'),checkpoint_sha256=sha(a.output/'last_model.pt'),
            reset_statistics=dict(zip(['episodes','nominal_episodes','contact_episodes','goal_episodes','physical_termination_episodes','observed_overspeed_episodes','instantaneous_terminal_stable_episodes'],counts.tolist())),
            acceptance='Requires separate original 14 s physical evaluation of legal contact, goal, actual joint limits and full recovery; these counts are training proxies')
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PRESTRIKE_TRAINING_FINISHED',json.dumps(result),flush=True)
    finally:env.close()

if __name__=='__main__':main()
