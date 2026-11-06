#!/usr/bin/env python3
"""Fit a measured capture-error actuator adapter to restore the frozen locomotion state.

State matching uses1.26s episodes and a paired, unperturbed frozen expert.
--full-task uses14s physical shot/goal/recovery search proxies. Independent
collision-point full-shot acceptance remains unchanged.
"""
import argparse,hashlib,json,math,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--population',type=int,default=128);p.add_argument('--generations',type=int,default=24);p.add_argument('--max-seconds',type=float,default=180.);p.add_argument('--seed',type=int,default=2604)
    p.add_argument('--full-task',action='store_true',help='Optimize unchanged shot/goal/recovery proxies over14s')
    a=p.parse_args();a.num_envs=a.population*3+1;a.shared_origin=True
    if not 16<=a.population<=256 or not 1<=a.generations<=32 or not 0<a.max_seconds<=300:p.error('Bounded prefix fit:<=256 candidates,32generations,300seconds')
    if a.output.exists():raise FileExistsError(a.output)
    if not a.full_task:p.error('Approach learning requires the full physical task')
    config=json.loads(a.config.read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    from strike_baseline_protocol import verify_files
    verify_files(ROOT,config['frozen_files_sha256'])
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True).app
    try:fit(a,config,sha)
    except BaseException:
        import traceback
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();raise
    finally:app.close()

def fit(a,config,sha):
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
    from placement_strike import get_task,DEFAULT_PARAMETERS,PARAMETER_NAMES
    task_features=lambda env:get_task(env).features()
    task_reward=lambda env:get_task(env).reward()
    from running_balance import nonfoot_contact_termination
    from task_strike_rewards import task_focused_reward,missed_strike,missed_goal
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
    cfg=G1FlatEnvCfg();cfg.joint_strike=config;cfg.scene.num_envs=a.num_envs;cfg.scene.env_spacing=0. if a.shared_origin else 8.
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
        raw.scene.update=physics;native_reset=raw._reset_idx;raw._reset_idx=lambda selected:None
        ids=torch.arange(a.num_envs,device=task.device)
        cases=[[.015,.015,.15],[0.,0.,.20],[-.025,0.,.10]]
        contexts=torch.cat([torch.zeros(1,3,device=task.device),torch.tensor(cases,device=task.device).repeat(a.population,1)],0)
        def reset_population():
            native_reset(ids);task.reset_fields(ids)
            state=task.ball.data.default_root_state.clone();state[:,7:]=0.
            state[:,:2]=torch.tensor(config['ball_xy'],device=task.device)+raw.scene.env_origins[:,:2]+contexts[:,:2];state[:,2]=.113
            task.ball.write_root_state_to_sim(state,env_ids=ids);task.ball.reset(ids);task.previous_ball[:]=state[:,:3]
            velocity=task.robot.data.root_state_w[:,7:13].clone();velocity[:,1]+=contexts[:,2]
            task.robot.write_root_velocity_to_sim(velocity,env_ids=ids)
            raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.)
        rng=torch.Generator(device=task.device).manual_seed(a.seed)
        mean=torch.tensor(config['approach_adapter']['coefficients'],device=task.device);std=torch.ones_like(mean)*.06
        best=mean.clone();best_loss=float('inf');start=time.monotonic();records=[]
        legs=[task.robot.joint_names.index(side+'_'+joint+'_joint') for side in ['left','right'] for joint in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]
        a.output.mkdir(parents=True)
        sources={name:sha(ROOT/'tools'/name) for name in ['fit_approach_adapter.py','placement_strike.py','closed_loop_expert.py','closed_loop_strike.py','joint_strike_core.py','approach_adapter.py','shank_clearance.py','hip_clearance.py']}
        manifest=dict(config=config,config_sha256=sha(a.config),seed=a.seed,population=a.population,generations=a.generations,max_seconds=a.max_seconds,sources=sources,
                      conditions=cases,condition='Measured lateral capture error; three development cases with paired nominal frozen expert',physical_episode_seconds=14. if a.full_task else 1.26,
                      authority='Actuator targets only; physical writes at reset only',acceptance='Parallel task proxies only; independent collision-point full acceptance required' if a.full_task else 'Prefix state restoration objective; full shot not inferred')
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for name in sources:(a.output/name).write_bytes((ROOT/'tools'/name).read_bytes())
        zero=torch.zeros(a.num_envs,30,device=task.device)
        for generation in range(a.generations):
            if generation and time.monotonic()-start>=a.max_seconds:break
            candidates=(mean+std*torch.randn(a.population,3,10,device=task.device,generator=rng)).clamp(-.16,.16)
            candidates[0]=best;candidates[1]=mean;task.approach_parameters[0]=0.;task.approach_parameters[1:]=candidates.repeat_interleave(3,dim=0)
            reset_population();initial=dict(root_position=task.robot.data.root_pos_w.tolist(),root_velocity=task.robot.data.root_lin_vel_w.tolist(),joint_pos=task.robot.data.joint_pos.tolist(),joint_vel=task.robot.data.joint_vel.tolist());samples=[];failed=torch.zeros(a.num_envs,dtype=torch.bool,device=task.device);loss=torch.zeros(a.population*3,device=task.device)
            with torch.no_grad():
                for step in range(699 if a.full_task else 63):
                    env.step(zero);failed|=raw.termination_manager.terminated
                    if step in [0,9,24,49,62,69,77,80,85]:samples.append(dict(step=step,root=task.robot.data.root_pos_w.tolist(),velocity=task.robot.data.root_lin_vel_w.tolist(),joint_pos=task.robot.data.joint_pos.tolist(),actions=raw.action_manager.action.tolist(),contacts=task.contact.tolist()))
                    if 59<=step<63:
                        r=task.robot.data
                        position=((r.root_pos_w[1:]-r.root_pos_w[:1]-torch.cat([contexts[1:,:2],torch.zeros(a.population*3,1,device=task.device)],-1))/.025).square().sum(-1)
                        orientation=(1.-(r.root_quat_w[1:]*r.root_quat_w[:1]).sum(-1).square()).clamp_min(0.)*1000.
                        speed=((r.root_lin_vel_w[1:]-r.root_lin_vel_w[:1])/.15).square().sum(-1)
                        angular=((r.root_ang_vel_w[1:]-r.root_ang_vel_w[:1])/.5).square().sum(-1)
                        joints=((r.joint_pos[1:][:,legs]-r.joint_pos[:1][:,legs])/.10).square().mean(-1)
                        rates=((r.joint_vel[1:][:,legs]-r.joint_vel[:1][:,legs])/2.).square().mean(-1)
                        loss+=(position+orientation+.5*speed+.3*angular+2.*joints+.2*rates)/4.
            if bool(failed[0]) and not a.full_task:raise RuntimeError('Paired nominal reference failed before strike')
            if a.full_task:
                quality=task.contact_quality[1:].clamp(0,1)
                success=task.scored[1:]&(quality>=.80)&(task.stable_seconds[1:]>=1.)&~failed[1:]&(task.expert.start[1:]>=0)
                loss=.05*loss-1000.*success.float()-100.*task.scored[1:].float()*quality-25.*task.contact[1:].float()*quality-10.*(~failed[1:]).float()
                loss+=100.*failed[1:].float()
            per_case=loss.reshape(a.population,3)
            loss=per_case.mean(1)+10.*candidates.square().mean(dim=(1,2))
            elite=loss.topk(max(12,a.population//8),largest=False).indices;index=int(loss.argmin());value=float(loss[index])
            if value<best_loss:best_loss=value;best=candidates[index].clone()
            mean=.2*mean+.8*candidates[elite].mean(0);std=torch.maximum(.2*std+.8*candidates[elite].std(0),torch.full_like(std,.004))
            r=task.robot.data
            record=dict(generation=generation,seconds=time.monotonic()-start,loss=value,best_loss=best_loss,
                        initial_zero_or_previous_best_loss=float(loss[0]),root_error_m=(r.root_pos_w[index*3+1]-r.root_pos_w[0]).tolist(),
                        velocity_error_mps=(r.root_lin_vel_w[index*3+1]-r.root_lin_vel_w[0]).tolist(),failed_candidates=int(failed[1:].sum()),coefficients=candidates[index].tolist())
            if a.full_task:record.update(proxy_success=bool(success.reshape(a.population,3)[index].all()),proxy_successes=int(success.sum()),
                success_by_case=success.reshape(a.population,3)[index].tolist(),goal_by_case=task.scored[1:].reshape(a.population,3)[index].tolist(),
                quality_by_case=quality.reshape(a.population,3)[index].tolist(),standing_by_case=task.stable_seconds[1:].reshape(a.population,3)[index].tolist(),
                failed_by_case=failed[1:].reshape(a.population,3)[index].tolist())
            (a.output/f'diagnostic_{generation}.json').write_text(json.dumps(dict(initial=initial,samples=samples,best_index=index,coefficients=candidates.tolist()))+'\n')
            records.append(record);print('PREFIX_GENERATION',json.dumps(record),flush=True)
            (a.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
            model=dict(config,approach_adapter=dict(config['approach_adapter'],coefficients=best.tolist()),approach_training='Bounded full-task CEM on measured capture error; independent physical acceptance required')
            (a.output/'best_config.json').write_text(json.dumps(model,indent=2)+'\n')
            (a.output/f'generation_{generation}.json').write_text(json.dumps(dict(model,approach_adapter=dict(config['approach_adapter'],coefficients=candidates[index].tolist())),indent=2)+'\n')
        result=dict(seconds=time.monotonic()-start,generations=len(records),best_loss=best_loss,acceptance='Independent full14s episode required')
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PREFIX_FIT_FINISHED',json.dumps(result),flush=True)
    finally:env.close()

if __name__=='__main__':main()
