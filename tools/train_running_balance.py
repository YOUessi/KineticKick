#!/usr/bin/env python3
"""Bounded 176-observation running feedback training and physical evaluation.

A warm-start actor learns from a finite running or run-to-stop command. Only training
resets write states. Original asset, actuator limits and gains are unchanged.
The reference ends with an episode timeout, never a running-state teleport.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--iters', type=int, default=800)
    p.add_argument('--max-seconds', type=float, default=900.)
    p.add_argument('--seed', type=int, default=1010)
    p.add_argument('--initial-velocity-jitter', type=float, default=0., help='Uniform planar root-velocity reset perturbation in m/s, 0..0.3; training only')
    p.add_argument('--resume', type=Path)
    p.add_argument('--init-checkpoint', type=Path, default=Path(__file__).resolve().parents[1]/'experiments/running_v010/capsule_resume_seed1012_n96_i400/last_model.pt')
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--video-dir', type=Path)
    p.add_argument('--arm-clearance-rad',type=float,default=0.)
    p.add_argument('--strict-contact',action='store_true')
    p.add_argument('--stand-handoff',action='store_true',help='Experimental measured-support handoff to original standing actor; evaluation only')
    p.add_argument('--full-phase-sampling',action='store_true',help='Sample half the training resets throughout the complete reference')
    p.add_argument('--recovery-window',type=float,default=0.,help='Evaluate final standing window in seconds; zero disables this gate')
    a = p.parse_args()
    if not (1 <= a.num_envs <= 1024 and 1 <= a.iters <= 2000 and 0 < a.max_seconds <= 1800):
        p.error('Pilot bounds: 1..1024 envs, 1..2000 iterations, 0..1800 seconds')
    if not math.isfinite(a.initial_velocity_jitter) or not 0 <= a.initial_velocity_jitter <= .3:
        p.error('Initial velocity jitter must be finite and within 0..0.3 m/s')
    if a.evaluate and a.initial_velocity_jitter:
        p.error('Use fixed-condition evaluation; randomized resets are training only')
    if not 0 <= a.recovery_window <= 2.:p.error('Recovery window must be between 0 and 2 seconds')
    if not a.motion.is_file():
        p.error('Motion file must exist')
    if a.output.exists() and (a.evaluate or any(a.output.iterdir())):
        p.error('Preserve existing experiment; choose an empty output directory')
    if a.evaluate and (a.num_envs != 1 or a.resume is None):
        p.error('Physical evaluation requires --num-envs 1 and --resume CHECKPOINT')
    if a.stand_handoff and not a.evaluate:p.error('Standing handoff is evaluation-only')
    if a.video_dir and not a.evaluate:p.error('Only physical evaluation renders frames')
    if not 0<=a.arm_clearance_rad<=.12 or (a.arm_clearance_rad and not a.evaluate):p.error('Arm guard is an explicit evaluation-only offset0..0.12rad')
    return a


def main():
    args = parse_args()
    # Isaac must be initialized before importing Torch or simulator modules.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=bool(args.video_dir)).app
    try:
        train(args)
    finally:
        app.close()


def train(args):
    import gymnasium as gym
    import torch
    import isaaclab.sim as sim
    from isaaclab.envs import mdp as physical_mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
    from isaaclab.utils.math import quat_apply_inverse
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import rsl_rl.runners.on_policy_runner as runner_module
    from running_balance import RunningBalanceActor, support_observation, reference_contact, contact_timing_reward, foot_clearance_reward, root_angular_tracking, nonfoot_contact_termination
    from paid_stage1_root_rewards import global_torso_linear_velocity_tracking, global_torso_planar_tracking_exp

    root = Path(__file__).resolve().parents[1]
    upstream = root / 'third_party/HumanoidSoccer/source/whole_body_tracking'
    sys.path.insert(0, str(upstream))
    shim = types.ModuleType('soccer.assets')
    shim.ASSET_DIR = str(upstream / 'soccer/assets')
    sys.modules['soccer.assets'] = shim
    import soccer.tasks
    from soccer.tasks.tracking import mdp
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

    def point_observation(env, y):
        robot = env.scene['robot']
        point = torch.tensor([0., y, .11], device=robot.device) + env.scene.env_origins
        return quat_apply_inverse(robot.data.root_quat_w, point - robot.data.root_pos_w)

    cfg = G1FlatEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 8.
    if args.video_dir:
        from isaaclab.sensors import CameraCfg
        cfg.scene.review_camera = CameraCfg(prim_path='{ENV_REGEX_NS}/ReviewCamera', update_period=.04,
            height=720,width=1280,data_types=['rgb'],spawn=sim.PinholeCameraCfg(focal_length=24.,horizontal_aperture=36.,clipping_range=(.05,100.)))
    cfg.sim.device = 'cuda:0'
    cfg.seed = args.seed
    import numpy as np
    motion_duration=(len(np.load(args.motion)['joint_pos'])-1)/50.
    cfg.episode_length_s = motion_duration+.04
    cfg.commands.motion.motion_files = [str(args.motion.resolve())]
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    cfg.scene.contact_forces.history_length = 4
    urdf = root / 'configs/paid_original/paid_main_resolved.urdf'
    cfg.scene.robot.spawn.asset_path = str(urdf)
    cfg.scene.robot.spawn.fix_base = False
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn = sim.DomeLightCfg(intensity=1100)
    for key in cfg.commands.motion.pose_range:
        cfg.commands.motion.pose_range[key] = (0., 0.)
    for key in cfg.commands.motion.velocity_range:
        cfg.commands.motion.velocity_range[key] = (0., 0.)
    for key in ('x', 'y'):
        cfg.commands.motion.velocity_range[key] = (-args.initial_velocity_jitter, args.initial_velocity_jitter)
    cfg.commands.motion.joint_position_range = (0., 0.)
    for key in ('physics_material', 'add_joint_default_pos', 'base_com', 'base_external_force_torque', 'push_robot'):
        setattr(cfg.events, key, None)
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.target_point_pos = ObsTerm(func=point_observation, params={'y': -8.})
    cfg.observations.policy.target_destination_pos_local = ObsTerm(func=point_observation, params={'y': -10.})
    cfg.observations.policy.motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b,params={'command_name':'motion'})
    cfg.observations.policy.motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b,params={'command_name':'motion'})
    cfg.observations.policy.base_lin_vel = ObsTerm(func=physical_mdp.base_lin_vel)
    cfg.observations.policy.feet_support = ObsTerm(func=support_observation)
    cfg.observations.policy.reference_contact = ObsTerm(func=reference_contact)

    # Absolute safety gates are distinct from the soccer wrist/reference-error gates.
    cfg.terminations.ee_body_pos = None
    cfg.terminations.anchor_pos_z = None
    cfg.terminations.anchor_ori = None
    cfg.terminations.physical_height = DoneTerm(func=physical_mdp.root_height_below_minimum, params={'minimum_height': .45})
    cfg.terminations.physical_orientation = DoneTerm(func=physical_mdp.bad_orientation, params={'limit_angle': math.radians(60.)})
    if args.strict_contact:
        cfg.terminations.physical_nonfoot = DoneTerm(func=nonfoot_contact_termination,params={'threshold':20.})
    cfg.terminations.reference_finished = DoneTerm(func=mdp.motion_finished, params={'command_name': 'motion'}, time_out=True)
    cfg.rewards.running_velocity = RewTerm(func=global_torso_linear_velocity_tracking, weight=3., params={'command_name': 'motion', 'std': 1.2})
    cfg.rewards.running_position = RewTerm(func=global_torso_planar_tracking_exp, weight=3., params={'command_name': 'motion', 'std': .8})
    cfg.rewards.alive = RewTerm(func=physical_mdp.is_alive, weight=1.)
    cfg.rewards.fall = RewTerm(func=physical_mdp.is_terminated, weight=-10.)
    cfg.rewards.running_contact_timing = RewTerm(func=contact_timing_reward,weight=2.)
    cfg.rewards.running_clearance = RewTerm(func=foot_clearance_reward,weight=3.)
    cfg.rewards.running_angular_tracking = RewTerm(func=root_angular_tracking,weight=2.)
    cfg.rewards.undesired_contacts.weight = -1.
    if args.strict_contact:
        cfg.rewards.undesired_contacts.params['sensor_cfg'].body_names = [r'^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$']


    runner_cfg = G1FlatPPORunnerCfg().to_dict()
    runner_cfg.update(num_steps_per_env=24, save_interval=100, empirical_normalization=False)
    source = root / '.pretrained_checkpoints/paid_official/policy_30000.onnx'
    runner_module.RunningBalanceActor = RunningBalanceActor
    runner_cfg['policy'] = dict(class_name='RunningBalanceActor', warmstart_path=str(args.init_checkpoint), init_noise_std=.18)
    runner_cfg['algorithm'].update(learning_rate=3e-4, schedule='adaptive', num_learning_epochs=4, num_mini_batches=4, entropy_coef=.002)
    run_dir=args.output if not args.evaluate else args.output.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = dict(purpose='176-observation contact and velocity feedback running adaptation',
        scope='Finite locomotion reference; no shot and no claim of standing start unless explicitly measured',
        seed=args.seed, num_envs=args.num_envs, max_iterations=args.iters, max_training_seconds=args.max_seconds,
        source_onnx=str(source), source_onnx_sha256=sha(source), motion=str(args.motion.resolve()), motion_sha256=sha(args.motion),
        urdf_sha256=sha(urdf), training_script_sha256=sha(__file__), actor_converter_sha256=sha(root / 'tools/running_balance.py'), warmstart_sha256=sha(args.init_checkpoint), warmstart=str(args.init_checkpoint), actor_observations=176,
        init='50% frame0; 50% uniform ' + ('complete reference' if args.full_phase_sampling else 'frames0..90') + '; physical state writes only on episode resets',
        full_phase_sampling=args.full_phase_sampling,recovery_window_s=args.recovery_window,
        initial_planar_velocity_jitter_mps=args.initial_velocity_jitter,
        observation_points_world=[[0,-8,.11],[0,-10,.11]], original_actuators_unchanged=True,
        reference_end='Timeout; no loop or state write within an episode', runner=runner_cfg,
        resume=str(args.resume) if args.resume else None)
    if not args.evaluate:(run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    env = RslRlVecEnvWrapper(gym.make('Tracking-Flat-G1-v0', cfg=cfg), clip_actions=100.)
    try:
        raw = env.unwrapped
        robot = raw.scene['robot']
        assert len(robot.joint_names) == 29 and not cfg.scene.robot.spawn.fix_base
        motion = raw.command_manager.get_term('motion')
        def sample_phase(command, env_ids):
            ids = torch.as_tensor(env_ids, device=command.device, dtype=torch.long)
            command.motion_idx[ids] = 0
            command.motion_length[ids] = command.motion.file_lengths[0]
            last = int(command.motion.file_lengths[0])-1
            upper = last if args.full_phase_sampling else min(91,last)
            phase = torch.randint(0, upper, (len(ids),), device=command.device)
            command.time_steps[ids] = 0 if args.evaluate else torch.where(torch.rand(len(ids), device=command.device) < .5, 0, phase)
        motion._uniform_sampling = types.MethodType(sample_phase, motion)
        native_update = motion._update_command
        def no_reference_wrap(command):
            cap = command.motion_length - 1
            command.time_steps[:] = torch.minimum(command.time_steps, cap - 1)
            return native_update()
        motion._update_command = types.MethodType(no_reference_wrap, motion)
        env.reset()
        phase = motion.time_steps.clone()
        motion.time_steps[:] = phase
        motion._update_command()
        assert torch.equal(motion.time_steps, phase+1)
        obs = raw.observation_manager.compute()['policy']
        assert obs.shape == (args.num_envs, 176), obs.shape
        import onnx
        metadata = {p.key: p.value for p in onnx.load(str(source)).metadata_props}
        assert robot.joint_names == metadata['joint_names'].split(',')
        assert list(raw.observation_manager.active_terms['policy'])[:9] == metadata['observation_names'].split(',')
        runner = runner_module.OnPolicyRunner(env, runner_cfg, log_dir=None if args.evaluate else str(run_dir), device='cuda:0')
        if args.resume:
            runner.load(str(args.resume), load_optimizer=not args.evaluate)
        if args.evaluate:
            evaluate(env,runner.alg.policy,args,manifest)
            return
        start = time.monotonic()
        original_update = runner.alg.update
        class BudgetReached(Exception):
            pass
        def bounded_update():
            if time.monotonic() - start >= args.max_seconds:
                raise BudgetReached()
            return original_update()
        runner.alg.update = bounded_update
        print('RUNNING_BALANCE_START', json.dumps({k: manifest[k] for k in ('num_envs','max_iterations','max_training_seconds','scope')}), flush=True)
        stop = 'iteration_limit'
        try:
            runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=False)
        except BudgetReached:
            stop = 'wall_time_limit'
        finally:
            runner.save(str(args.output / 'last_model.pt'))
        result = dict(stop=stop, training_seconds=time.monotonic()-start,
            last_iteration=runner.current_learning_iteration, checkpoint=str(args.output / 'last_model.pt'),
            checkpoint_sha256=sha(args.output / 'last_model.pt'), note='Training completion is not running acceptance; evaluate separately.')
        (args.output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print('RUNNING_BALANCE_FINISHED', json.dumps(result), flush=True)
    finally:
        env.close()



def evaluate(env,policy,args,manifest):
    import json,math
    import numpy as np
    import torch
    from running_balance import setup_feedback,ground_force
    from running_metrics import summarize_running
    raw=env.unwrapped;robot=raw.scene['robot'];motion=raw.command_manager.get_term('motion')
    policy.eval();policy.reset(torch.ones(1,device=robot.device,dtype=torch.bool))
    feedback=setup_feedback(raw);sensor=raw.scene['contact_forces']
    nonfoot=[i for i,n in enumerate(sensor.body_names) if n not in ('left_ankle_roll_link','right_ankle_roll_link')]
    initial=(robot.data.root_pos_w-raw.scene.env_origins)[0].cpu().tolist()
    rows=[];kinematics=[];substep=0
    native_update=raw.scene.update;native_reset=raw._reset_idx
    raw._reset_idx=lambda ids: None
    def capture(dt):
        nonlocal substep
        native_update(dt);substep+=1
        pos=(robot.data.root_pos_w-raw.scene.env_origins)[0].cpu().tolist()
        tilt=math.degrees(math.acos(float((-robot.data.projected_gravity_b[0,2]).clamp(-1,1))))
        force=ground_force(raw)[0].cpu().tolist()
        non=float(sensor.data.net_forces_w[0,nonfoot].norm(dim=-1).max())
        rows.append(dict(time_s=substep*dt,physics_dt_s=dt,root_xyz=pos,root_tilt_deg=tilt,
            root_linear_velocity=robot.data.root_lin_vel_w[0].cpu().tolist(),
            root_angular_velocity=robot.data.root_ang_vel_w[0].cpu().tolist(),
            nonfoot_contact_N=non,support_foot_force_N=dict(zip(('left','right'),force)),terminated=False))
    raw.scene.update=capture
    if args.video_dir:
        from PIL import Image
        if args.video_dir.exists() and any(args.video_dir.iterdir()):raise FileExistsError(args.video_dir)
        args.video_dir.mkdir(parents=True,exist_ok=True)
        camera=raw.scene['review_camera']
        def follow_camera():
            target=robot.data.root_pos_w.clone();target[:,2]=.65
            eye=target+torch.tensor([[3.2,.6,1.6]],device=robot.device)
            camera.set_world_poses_from_view(eyes=eye,targets=target)
        follow_camera()
        for _ in range(4):raw.sim.render()
    obs=raw.observation_manager.compute()['policy'];reasons=[]
    standing_actor=None;handoff_start=None;ready_steps=0;handoff_events=[]
    if args.stand_handoff:
        from soccer_onnx_policy import OnnxSoccerPolicy
        standing_actor=OnnxSoccerPolicy(manifest['source_onnx'],1)
        standing_actor.validate_joints(robot.joint_names)
        if list(raw.observation_manager.active_terms['policy'])[:9]!=standing_actor.observation_names:raise ValueError('Standing handoff observation schema mismatch')
    try:
        with torch.inference_mode():
            for step in range(int(motion.motion_length[0])-1):
                if args.video_dir:follow_camera()
                action=policy.act_inference(obs)
                if standing_actor is not None:
                    speed=float(robot.data.root_lin_vel_w[0,:2].norm())
                    tilt=math.degrees(math.acos(float((-robot.data.projected_gravity_b[0,2]).clamp(-1,1))))
                    support=ground_force(raw)[0]
                    ready=bool(float(motion.anchor_lin_vel_w[0,:2].norm())<.1 and speed<.45 and tilt<15 and float(support.min())>=30)
                    ready_steps=ready_steps+1 if ready else 0
                    if handoff_start is None and ready_steps>=4:
                        handoff_start=step*raw.step_dt
                        handoff_events.append(dict(time_s=handoff_start,root_speed_m_s=speed,tilt_deg=tilt,foot_force_N=support.cpu().tolist(),physical_state_write=False))
                    if handoff_start is not None:
                        standing=torch.tensor(standing_actor.step(obs[:,:160].cpu().numpy(),motion.time_steps.cpu().numpy()),device=robot.device)
                        u=min((step*raw.step_dt-handoff_start)/.4,1.)
                        weight=.5-.5*math.cos(math.pi*u)
                        action=(1-weight)*action+weight*standing
                if args.arm_clearance_rad:
                    term=raw.action_manager.get_term('joint_pos')
                    targets=action*term._scale+term._offset
                    ramp=min((step+1)*raw.step_dt/.2,1.)
                    for name,sign in [('left_shoulder_roll_joint',1),('right_shoulder_roll_joint',-1)]:
                        j=robot.joint_names.index(name)
                        targets[:,j]+=sign*args.arm_clearance_rad*ramp
                    bounds=robot.data.soft_joint_pos_limits
                    targets=torch.clamp(targets,bounds[:,:,0],bounds[:,:,1])
                    action=(targets-term._offset)/term._scale
                obs,_,done,_=env.step(action)
                kinematics.append(dict(time_s=(step+1)*raw.step_dt,root_pose=robot.data.root_state_w[0,:7].cpu().tolist(),
                    joint_pos=robot.data.joint_pos[0].cpu().tolist(),joint_vel=robot.data.joint_vel[0].cpu().tolist(),
                    action=action[0].cpu().tolist()))
                if args.video_dir and step%2==0:
                    Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(args.video_dir/f'frame_{step//2:04d}.png')
                if bool(done[0]):
                    reasons=[n for n in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(n)[0])]
                    if any(n.startswith('physical_') for n in reasons):rows[-1]['terminated']=True
                    break
        summary=summarize_running(rows,initial_root=initial,forward=[0,-1])
        summary.update(duration_s=rows[-1]['time_s'],forward_displacement_m=initial[1]-rows[-1]['root_xyz'][1],
            lateral_displacement_m=rows[-1]['root_xyz'][0]-initial[0],termination_reasons=reasons)
        if args.recovery_window:
            from running_sequence_metrics import summarize_recovery
            summary['recovery']=summarize_recovery(rows,expected_duration_s=(int(motion.motion_length[0])-1)*raw.step_dt,window_s=args.recovery_window)
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        result=dict(summary=summary,traces=[rows],kinematics=[kinematics],initial_root=initial,
            checkpoint=str(args.resume),checkpoint_sha256=sha(args.resume),source_sha256=sha(__file__),
            feedback_sha256=sha(Path(__file__).with_name('running_balance.py')),motion=str(args.motion),motion_sha256=sha(args.motion),
            stand_handoff=args.stand_handoff,handoff_events=handoff_events,
            standing_policy=manifest['source_onnx'] if args.stand_handoff else None,
            standing_policy_sha256=manifest['source_onnx_sha256'] if args.stand_handoff else None,
            actor_observations=176,observation_names=list(raw.observation_manager.active_terms['policy']),
            root_fixed=False,post_initialization_state_writes=False,reference_lead_frames=1,arm_clearance_rad=args.arm_clearance_rad,strict_contact=args.strict_contact,
            ground_collision_paths=feedback['ground'],joint_names=robot.joint_names,
            joint_effort_limits=robot.data.joint_effort_limits[0].cpu().tolist(),
            joint_velocity_limits=robot.data.joint_vel_limits[0].cpu().tolist(),scope=manifest['scope'])
        args.output.write_text(json.dumps(result,indent=2))
        print('RUNNING_BALANCE_EVALUATION',json.dumps(summary),flush=True)
    finally:raw.scene.update=native_update;raw._reset_idx=native_reset


if __name__ == '__main__':
    main()
