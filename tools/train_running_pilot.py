#!/usr/bin/env python3
"""Bounded original-29DoF running research; moving-phase resets, not standing start.

A warm-start actor learns from a finite 2.2 s running command. Only training
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
    p.add_argument('--num-envs', type=int, default=96)
    p.add_argument('--iters', type=int, default=400)
    p.add_argument('--max-seconds', type=float, default=600.)
    p.add_argument('--seed', type=int, default=1010)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    if not (1 <= a.num_envs <= 160 and 1 <= a.iters <= 2000 and 0 < a.max_seconds <= 1800):
        p.error('Pilot bounds: 1..160 envs, 1..2000 iterations, 0..1800 seconds')
    if not a.motion.is_file():
        p.error('Motion file must exist')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Preserve existing experiment; choose an empty output directory')
    return a


def main():
    args = parse_args()
    # Isaac must be initialized before importing Torch or simulator modules.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app
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
    from paid_actor_warmstart import PaidActorCriticRecurrent
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
    cfg.sim.device = 'cuda:0'
    cfg.seed = args.seed
    cfg.episode_length_s = 2.2
    cfg.commands.motion.motion_files = [str(args.motion.resolve())]
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    urdf = root / 'configs/paid_original/paid_main_resolved.urdf'
    cfg.scene.robot.spawn.asset_path = str(urdf)
    cfg.scene.robot.spawn.fix_base = False
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn = sim.DomeLightCfg(intensity=1100)
    for key in cfg.commands.motion.pose_range:
        cfg.commands.motion.pose_range[key] = (0., 0.)
    for key in cfg.commands.motion.velocity_range:
        cfg.commands.motion.velocity_range[key] = (0., 0.)
    cfg.commands.motion.joint_position_range = (0., 0.)
    for key in ('physics_material', 'add_joint_default_pos', 'base_com', 'base_external_force_torque', 'push_robot'):
        setattr(cfg.events, key, None)
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.target_point_pos = ObsTerm(func=point_observation, params={'y': -8.})
    cfg.observations.policy.target_destination_pos_local = ObsTerm(func=point_observation, params={'y': -10.})
    # Absolute safety gates are distinct from the soccer wrist/reference-error gates.
    cfg.terminations.ee_body_pos = None
    cfg.terminations.anchor_pos_z = None
    cfg.terminations.anchor_ori = None
    cfg.terminations.physical_height = DoneTerm(func=physical_mdp.root_height_below_minimum, params={'minimum_height': .45})
    cfg.terminations.physical_orientation = DoneTerm(func=physical_mdp.bad_orientation, params={'limit_angle': math.radians(60.)})
    cfg.terminations.reference_finished = DoneTerm(func=mdp.motion_finished, params={'command_name': 'motion'}, time_out=True)
    cfg.rewards.running_velocity = RewTerm(func=global_torso_linear_velocity_tracking, weight=3., params={'command_name': 'motion', 'std': 1.2})
    cfg.rewards.running_position = RewTerm(func=global_torso_planar_tracking_exp, weight=3., params={'command_name': 'motion', 'std': .8})
    cfg.rewards.alive = RewTerm(func=physical_mdp.is_alive, weight=1.)
    cfg.rewards.fall = RewTerm(func=physical_mdp.is_terminated, weight=-10.)

    runner_cfg = G1FlatPPORunnerCfg().to_dict()
    runner_cfg.update(num_steps_per_env=24, save_interval=100, empirical_normalization=False)
    source = root / '.pretrained_checkpoints/paid_official/policy_30000.onnx'
    runner_module.PaidActorCriticRecurrent = PaidActorCriticRecurrent
    runner_cfg['policy'] = dict(class_name='PaidActorCriticRecurrent', onnx_path=str(source), init_noise_std=.15)
    runner_cfg['algorithm'].update(learning_rate=1e-4, schedule='fixed', num_learning_epochs=4, num_mini_batches=4, entropy_coef=.002)
    args.output.mkdir(parents=True, exist_ok=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = dict(purpose='Bounded running adaptation of the original 29DoF soccer actor',
        scope='Moving-phase running only; not standing start, run-to-shot or full recovery',
        seed=args.seed, num_envs=args.num_envs, max_iterations=args.iters, max_training_seconds=args.max_seconds,
        source_onnx=str(source), source_onnx_sha256=sha(source), motion=str(args.motion.resolve()), motion_sha256=sha(args.motion),
        urdf_sha256=sha(urdf), training_script_sha256=sha(__file__), actor_converter_sha256=sha(root / 'tools/paid_actor_warmstart.py'),
        init='50% frame0; 50% uniformly frames0..60, reference root pose/velocity only on episode resets',
        observation_points_world=[[0,-8,.11],[0,-10,.11]], original_actuators_unchanged=True,
        reference_end='Timeout; no loop or state write within an episode', runner=runner_cfg,
        resume=str(args.resume) if args.resume else None)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
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
            phase = torch.randint(0, 61, (len(ids),), device=command.device)
            command.time_steps[ids] = torch.where(torch.rand(len(ids), device=command.device) < .5, 0, phase)
        motion._uniform_sampling = types.MethodType(sample_phase, motion)
        native_update = motion._update_command
        def no_reference_wrap(command):
            cap = command.motion_length - 1
            command.time_steps[:] = torch.minimum(command.time_steps, cap - 1)
            return native_update()
        motion._update_command = types.MethodType(no_reference_wrap, motion)
        env.reset()
        phase = motion.time_steps.clone()
        motion.time_steps[:] = phase - 1
        motion._update_command()
        assert torch.equal(motion.time_steps, phase)
        obs = raw.observation_manager.compute()['policy']
        assert obs.shape == (args.num_envs, 160), obs.shape
        import onnx
        metadata = {p.key: p.value for p in onnx.load(str(source)).metadata_props}
        assert robot.joint_names == metadata['joint_names'].split(',')
        assert list(raw.observation_manager.active_terms['policy']) == metadata['observation_names'].split(',')
        runner = runner_module.OnPolicyRunner(env, runner_cfg, log_dir=str(args.output), device='cuda:0')
        if args.resume:
            runner.load(str(args.resume), load_optimizer=True)
        start = time.monotonic()
        original_update = runner.alg.update
        class BudgetReached(Exception):
            pass
        def bounded_update():
            if time.monotonic() - start >= args.max_seconds:
                raise BudgetReached()
            return original_update()
        runner.alg.update = bounded_update
        print('RUNNING_PILOT_START', json.dumps({k: manifest[k] for k in ('num_envs','max_iterations','max_training_seconds','scope')}), flush=True)
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
        print('RUNNING_PILOT_FINISHED', json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
