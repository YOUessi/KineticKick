#!/usr/bin/env python3
"""Bounded PPO on an actual approach-kick-recovery episode with a frozen prior."""
import argparse,hashlib,json,math,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/joint_strike_learning_v1.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--iters',type=int,default=1000)
    p.add_argument('--max-seconds',type=float,default=900.)
    p.add_argument('--seed',type=int,default=2501)
    p.add_argument('--resume',type=Path)
    p.add_argument('--zero-probe',action='store_true',help='No optimizer: probe the shared training environment for one nominal horizon')
    a=p.parse_args()
    if not (1<=a.num_envs<=1024 and 1<=a.iters<=1000 and 0<a.max_seconds<=900 and math.isfinite(a.max_seconds)):p.error('Bounds: <=1024 envs, <=1000 updates, <=900 training seconds')
    if a.output.exists():raise FileExistsError(a.output)
    config=json.loads(a.config.read_text())
    from strike_baseline_protocol import verify_files
    verify_files(ROOT,config['frozen_files_sha256'])
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for key in ['template','reference']:
        if sha(ROOT/config[key])!=config[key+'_sha256']:raise ValueError('Changed '+key)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True).app
    try:train(a,config,sha)
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
    from joint_strike_core import get_task,task_features,task_reward,JointStrikeActor
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
    cfg.seed=a.seed;cfg.sim.device='cuda:0';cfg.episode_length_s=14.
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
    class ResidualEnv(RslRlVecEnvWrapper):
        def step(self,actions):
            task=get_task(self.unwrapped);physical=task.commands(actions)
            task.robot.set_joint_velocity_target(task.step_velocity)
            return super().step(physical)
    env=ResidualEnv(gym.make('Tracking-Flat-G1-v0',cfg=cfg),clip_actions=None);env.num_actions=30
    try:
        raw=env.unwrapped;task=get_task(raw);motion=raw.command_manager.get_term('motion')
        assert list(raw.reward_manager.active_terms)==['joint_strike']
        native_update=motion._update_command
        def no_wrap(command):
            command.time_steps[:]=torch.minimum(command.time_steps,command.motion_length-2)
            return native_update()
        motion._update_command=types.MethodType(no_wrap,motion)
        scene_update=raw.scene.update
        def physics(dt):scene_update(dt);task.after_physics(dt)
        raw.scene.update=physics
        native_reset=raw._reset_idx
        counts=torch.zeros(6,device=task.device);curriculum=[.25];probe_resets=[]
        def reset(ids):
            ids=torch.as_tensor(ids,device=task.device,dtype=torch.long)
            valid=task.elapsed[ids]>1
            nm=task.nominal_mask[ids]&valid;rm=~task.nominal_mask[ids]&valid
            success=task.scored[ids]&(task.contact_quality[ids]>=.75)&(task.stable_seconds[ids]>=1.)&~raw.termination_manager.terminated[ids]
            if a.zero_probe and bool((ids==0).any()) and float(task.elapsed[0])>1:
                probe_resets.append(dict(step=float(task.elapsed[0]),root=(task.robot.data.root_pos_w[0]-raw.scene.env_origins[0]).cpu().tolist(),tilt_cos=float(-task.robot.data.projected_gravity_b[0,2]),scored=bool(task.scored[0]),contact=bool(task.contact[0]),termination={n:bool(raw.termination_manager.get_term(n)[0]) for n in raw.termination_manager.active_terms}))
            counts.add_(torch.stack([nm.sum(),(nm&success).sum(),rm.sum(),(rm&success).sum(),(valid&task.contact[ids]).sum(),(valid&task.scored[ids]).sum()]))
            native_reset(ids);task.reset_fields(ids)
            nominal=ids%2==0;task.nominal_mask[ids]=nominal
            amplitude=torch.tensor(config['training']['ball_jitter_xy_m'],device=task.device)*curriculum[0]
            offset=(torch.rand(len(ids),2,device=task.device)*2-1)*amplitude;offset[nominal]=0.
            task.shift[ids]=offset
            state=task.ball.data.default_root_state[ids].clone();state[:,7:]=0.
            state[:,:2]=torch.tensor(config['ball_xy'],device=task.device)+offset+raw.scene.env_origins[ids,:2]
            state[:,2]=.113+raw.scene.env_origins[ids,2]
            task.ball.write_root_state_to_sim(state,env_ids=ids);task.ball.reset(ids);task.previous_ball[ids]=state[:,:3]
            velocity=task.robot.data.root_state_w[ids,7:13].clone()
            dy=(torch.rand(len(ids),device=task.device)*2-1)*config['training']['initial_velocity_y_jitter_mps']*curriculum[0];dy[nominal]=0.
            velocity[:,1]+=dy;task.robot.write_root_velocity_to_sim(velocity,env_ids=ids)
        raw._reset_idx=reset
        env.reset();obs,extra=env.get_observations();assert torch.isfinite(obs).all()
        runner_cfg=G1FlatPPORunnerCfg().to_dict();runner_cfg.update(num_steps_per_env=24,save_interval=100,empirical_normalization=False)
        runner_cfg['policy']=dict(class_name='JointStrikeActor',init_noise_std=.15)
        runner_cfg['algorithm'].update(learning_rate=1e-4,schedule='fixed',num_learning_epochs=4,num_mini_batches=4,entropy_coef=.002,gamma=.995)
        runners.JointStrikeActor=JointStrikeActor
        a.output.mkdir(parents=True)
        manifest=dict(version='0.1.25',config=config,config_sha256=sha(a.config),seed=a.seed,num_envs=a.num_envs,
            max_seconds=a.max_seconds,max_updates=a.iters,actor_observations=obs.shape[-1],actor_actions=30,
            sources={n:sha(ROOT/'tools'/n) for n in ['train_joint_strike.py','joint_strike_core.py']},
            initial_policy='zero mean residual over recorded successful actuator commands',
            physical_reset_only=True,nominal_fraction=.5,curriculum='randomized half: scale .25 to1 over first400 PPO updates; nominal half remains exact',
            reward_surface='Filtered force-normal local +Z proxy; exact collision-point surface reserved for physical acceptance',
            proxy_metrics='Training reset statistics are not full physical acceptance',runner=runner_cfg)
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for name in manifest['sources']:(a.output/name).write_bytes((ROOT/'tools'/name).read_bytes())
        if a.zero_probe:
            initial=dict(root=(task.robot.data.root_pos_w-raw.scene.env_origins).cpu().tolist(),velocity=task.robot.data.root_lin_vel_w.cpu().tolist(),joint_pos=task.robot.data.joint_pos.cpu().tolist())
            start=time.monotonic()
            with torch.inference_mode():
                for _ in range(699):env.step(torch.zeros(a.num_envs,30,device=task.device))
            result=dict(mode='zero_training_probe',seconds=time.monotonic()-start,initial=initial,
                elapsed=task.elapsed.cpu().tolist(),contact=task.contact.cpu().tolist(),quality=task.contact_quality.cpu().tolist(),goals=task.scored.cpu().tolist(),stable_seconds=task.stable_seconds.cpu().tolist(),reset_counts=counts.cpu().tolist(),first_env_resets=probe_resets)
            (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('ZERO_TRAINING_PROBE',json.dumps(result),flush=True)
            return
        runner=runners.OnPolicyRunner(env,runner_cfg,log_dir=str(a.output),device=task.device)
        if a.resume:runner.load(str(a.resume))
        start=time.monotonic();native_alg_update=runner.alg.update;updates=[0]
        class BudgetReached(Exception):pass
        def update():
            if (a.output/'STOP').exists():raise BudgetReached('diagnostic_stop')
            if time.monotonic()-start>=a.max_seconds:raise BudgetReached('wall_time_limit')
            result=native_alg_update();updates[0]+=1;curriculum[0]=.25+.75*min(1.,updates[0]/400.)
            if updates[0]%50==0:
                (a.output/'progress.json').write_text(json.dumps(dict(updates=updates[0],seconds=time.monotonic()-start,curriculum=curriculum[0],reset_statistics=counts.cpu().tolist())))
            return result
        runner.alg.update=update;stop='iteration_limit'
        try:runner.learn(num_learning_iterations=a.iters,init_at_random_ep_len=False)
        except BudgetReached as exc:stop=str(exc)
        finally:runner.save(str(a.output/'last_model.pt'))
        result=dict(stop=stop,training_seconds=time.monotonic()-start,updates=updates[0],curriculum_scale=curriculum[0],checkpoint=str(a.output/'last_model.pt'),checkpoint_sha256=sha(a.output/'last_model.pt'),
                    reset_statistics=dict(zip(['nominal_episodes','nominal_proxy_successes','perturbed_episodes','perturbed_proxy_successes','right_contact_episodes','goal_episodes'],counts.cpu().tolist())),acceptance='Requires independent unchanged physical evaluator')
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('JOINT_STRIKE_TRAINING_FINISHED',json.dumps(result),flush=True)
    finally:env.close()

if __name__=='__main__':main()
