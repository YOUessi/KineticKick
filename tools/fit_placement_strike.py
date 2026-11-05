#!/usr/bin/env python3
"""Bounded cross-entropy fitting of measured-state placement/phase gains.

Population scores are search proxies; final models require unchanged physical
single-environment acceptance. A robot or ball is written only at episode reset.
"""
import argparse,hashlib,json,math,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASES=[[-.05,0.,0.],[.05,0.,0.],[0.,.10,0.],[0.,-.10,0.],[0.,0.,.20],[0.,0.,-.20]]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/closed_loop_strike_v1.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--population',type=int,default=96)
    p.add_argument('--generations',type=int,default=8)
    p.add_argument('--seed',type=int,default=2603)
    p.add_argument('--max-seconds',type=float,default=600)
    p.add_argument('--shared-origin',action='store_true',help='Collision-isolated environments at identical world coordinates for numerical parity probe')
    a=p.parse_args();a.num_envs=a.population*len(CASES)
    if not 8<=a.population<=128 or not 1<=a.generations<=12 or not 0<a.max_seconds<=1200:p.error('Bounded search:8..128 candidates,1..12 generations,<=1200seconds')
    if a.output.exists():raise FileExistsError(a.output)
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
        plane_error=torch.full((a.num_envs,),10.,device=task.device)
        def physics(dt):
            previous=task.previous_ball.clone();scene_update(dt);task.after_physics(dt)
            current=task.ball.data.root_pos_w;plane=task.goal[:,1]-.11
            crossing=(previous[:,1]>plane)&(current[:,1]<=plane)&(plane_error==10.)
            fraction=((previous[:,1]-plane)/(previous[:,1]-current[:,1]).clamp_min(1e-8)).clamp(0,1)
            lateral=previous[:,0]+fraction*(current[:,0]-previous[:,0])-task.goal[:,0]
            plane_error[crossing]=lateral[crossing].abs()
        raw.scene.update=physics
        native_reset=raw._reset_idx
        ids=torch.arange(a.num_envs,device=task.device)
        contexts=torch.tensor(CASES,device=task.device).repeat(a.population,1)
        def reset_population():
            native_reset(ids);task.reset_fields(ids)
            state=task.ball.data.default_root_state.clone();state[:,7:]=0.
            state[:,:2]=torch.tensor(config['ball_xy'],device=task.device)+contexts[:,:2]+raw.scene.env_origins[:,:2]
            state[:,2]=.113+raw.scene.env_origins[:,2]
            task.ball.write_root_state_to_sim(state,env_ids=ids);task.ball.reset(ids);task.previous_ball[:]=state[:,:3]
            velocity=task.robot.data.root_state_w[:,7:13].clone();velocity[:,1]+=contexts[:,2]
            task.robot.write_root_velocity_to_sim(velocity,env_ids=ids)
            raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.)
        # Terminations are latched as failure; no mid-episode automatic respawn.
        raw._reset_idx=lambda selected:None
        rng=torch.Generator(device=task.device).manual_seed(a.seed)
        values=list(config.get('placement_parameters',DEFAULT_PARAMETERS))
        if len(values)==13:values+=4*[0.]
        mean=torch.tensor(values,device=task.device)
        std=torch.tensor([.4,.4,.5,.5,.5,.5,.3,.3,.3,.06,.06,.06,.06,.7,.5,.7,.5],device=task.device)
        low=torch.tensor([-2.,-2.,-3.,-3.,-3.,-3.,0.,0.,0.,-.2,-.2,-.2,-.2,-3.,-3.,-3.,-3.],device=task.device)
        high=torch.tensor([3.,3.,3.,3.,3.,3.,2.,2.,2.,.2,.2,.2,.2,3.,3.,3.,3.],device=task.device)
        floor=std*.12
        a.output.mkdir(parents=True)
        sources={n:sha(ROOT/'tools'/n) for n in ['fit_placement_strike.py','placement_strike.py','closed_loop_strike.py','closed_loop_expert.py','joint_strike_core.py']}
        manifest=dict(seed=a.seed,cases=CASES,population=a.population,generations=a.generations,max_seconds=a.max_seconds,sources=sources,shared_origin=a.shared_origin,
                      parameter_names=PARAMETER_NAMES,config=config,definition='Search proxies only; final independent physical acceptance required',physical_writes='Reset only; free-root original asset and actuator limits')
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for n in sources:(a.output/n).write_bytes((ROOT/'tools'/n).read_bytes())
        best_score=-float('inf');best=mean.clone();records=[];start=time.monotonic()
        zero=torch.zeros(a.num_envs,30,device=task.device)
        for generation in range(a.generations):
            if generation and time.monotonic()-start>a.max_seconds:break
            population=(mean+std*torch.randn(a.population,len(mean),generator=rng,device=task.device)).clamp(min=low,max=high)
            population[0]=best;population[1]=mean
            if generation==0:population[2]=0.
            task.parameters[:]=population.repeat_interleave(len(CASES),dim=0)
            reset_population();plane_error.fill_(10.);sampled=[];initial=dict(root=(task.robot.data.root_pos_w-raw.scene.env_origins).tolist(),velocity=task.robot.data.root_lin_vel_w.tolist());failed=torch.zeros(a.num_envs,device=task.device,dtype=torch.bool)
            minimum_distance=torch.full((a.num_envs,),10.,device=task.device);approach_error=torch.zeros_like(minimum_distance)
            with torch.no_grad():
                for step in range(699):
                    env.step(zero);failed|=raw.termination_manager.terminated
                    if step in [0,49,69,77,80,698]:sampled.append(dict(step=step,root=(task.robot.data.root_pos_w-raw.scene.env_origins).tolist(),ball=(task.ball.data.root_pos_w-raw.scene.env_origins).tolist()))
                    distance=(task.robot.data.body_pos_w[:,task.feet[1]]-task.ball.data.root_pos_w).norm(dim=-1)
                    minimum_distance=torch.minimum(minimum_distance,distance)
                    if step==69:
                        desired=task.sample('root_pose',torch.full_like(task.elapsed,70.))[:,:2]+contexts[:,:2]
                        approach_error=(task.robot.data.root_pos_w[:,:2]-raw.scene.env_origins[:,:2]-desired).norm(dim=-1)
            quality=task.contact_quality.clamp(0,1);survived=~failed
            success=task.scored&(quality>=.75)&(task.stable_seconds>=1.)&survived&(task.expert.start>=0)
            score=1000.*success.float()+100.*task.scored.float()*quality+20.*task.contact.float()*quality+10.*survived.float()
            score+=5.*torch.exp(-minimum_distance/.15)-80.*approach_error.clamp_max(2.)+100.*torch.exp(-(plane_error/.35).square())
            per_case=score.reshape(a.population,len(CASES));fitness=per_case.mean(-1)
            elite=fitness.topk(max(8,a.population//8)).indices
            current=int(fitness.argmax());current_score=float(fitness[current])
            if current_score>best_score:best_score=current_score;best=population[current].clone()
            mean=.25*mean+.75*population[elite].mean(0)
            std=torch.maximum(.25*std+.75*population[elite].std(0),floor)
            record=dict(generation=generation,seconds=time.monotonic()-start,score=current_score,best_score=best_score,
                        parameters=population[current].tolist(),success_by_case=success.reshape(a.population,-1)[current].tolist(),
                        goal_by_case=task.scored.reshape(a.population,-1)[current].tolist(),contact_by_case=task.contact.reshape(a.population,-1)[current].tolist(),
                        quality_by_case=quality.reshape(a.population,-1)[current].tolist(),goal_lateral_error_m=plane_error.reshape(a.population,-1)[current].tolist(),standing_by_case=task.stable_seconds.reshape(a.population,-1)[current].tolist(),
                        maximum_success_cases=int(success.reshape(a.population,-1).sum(-1).max()))
            record['first_candidate_success']=success.reshape(a.population,-1)[0].tolist()
            record['first_candidate_goals']=task.scored.reshape(a.population,-1)[0].tolist()
            (a.output/f'diagnostic_{generation}.json').write_text(json.dumps(dict(initial=initial,samples=sampled,best_index=current))+'\n')
            records.append(record);print('GENERATION',json.dumps(record),flush=True)
            (a.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
            candidate=dict(config,placement_parameters=population[current].tolist(),placement_parameter_names=PARAMETER_NAMES,
                           fitted_model='Measured-state command/phase gains with orientation-preserving swing IK',search_generation=generation)
            (a.output/f'generation_{generation}.json').write_text(json.dumps(candidate,indent=2)+'\n')
            final=dict(config,placement_parameters=best.tolist(),placement_parameter_names=PARAMETER_NAMES,
                       fitted_model='Measured-state command/phase gains with orientation-preserving swing IK')
            (a.output/'best_config.json').write_text(json.dumps(final,indent=2)+'\n')
        result=dict(seconds=time.monotonic()-start,generations=len(records),best_score=best_score,parameters=best.tolist(),acceptance='Search only; independent evaluation pending')
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('SEARCH_FINISHED',json.dumps(result),flush=True)
    finally:env.close()

if __name__=='__main__':main()
