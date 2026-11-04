#!/usr/bin/env python3
"""Measured free-root shooting with the upstream ball-conditioned policy.

One simulator; physics-substep contact evidence; no state writes during a trial.
This command reports failed and ambiguous trials rather than relabeling them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
import time

ROOT = Path(__file__).resolve().parents[1]


def motion_provenance(source_provenance, motion_files):
    """Describe the resolved inputs, including multi-reference diagnostics."""
    paths = [Path(path).resolve() for path in motion_files]
    if not paths:
        raise ValueError('At least one motion reference is required')
    result = dict(source_provenance)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result['motion_sources_sha256'] = hashes
    expected = (ROOT / result['motion_file']).resolve()
    if paths != [expected]:
        result.pop('motion_file')
        result.pop('motion_sha256')
    elif hashes[str(expected)] != result['motion_sha256']:
        raise ValueError('Motion reference changed after provenance preflight')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--steps", type=int, default=410)
    parser.add_argument("--ball-x", type=float, default=.14)
    parser.add_argument("--ball-y", type=float, default=-.93)
    parser.add_argument("--goal-x", type=float, default=0.)
    parser.add_argument("--goal-y", type=float, default=-3.7)
    parser.add_argument("--style", choices=("instep", "inside", "outside"), default="inside")
    parser.add_argument("--config", default=str(ROOT / "configs/free_shooting_v1.json"))
    parser.add_argument('--target-config', default=str(ROOT / 'configs/targeted_shooting_v1.json'))
    parser.add_argument('--target-x', type=float, help='Calibrated fixed-goal horizontal target in meters')
    strength = parser.add_mutually_exclusive_group()
    strength.add_argument('--power', choices=('low', 'medium', 'high'))
    strength.add_argument('--speed-mps', type=float, help='A calibrated outgoing speed for the selected style')
    parser.add_argument('--targeted-acceptance', action='store_true', help='Run all nine aim/power commands for --style with seeded perturbations')
    parser.add_argument("--baseline", action="store_true", help="Evaluate the original policy without style control")
    parser.add_argument("--acceptance", action="store_true", help="Run the frozen 15-trial matrix with seeded reset perturbations")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold", action="store_true", help="Keep the finished GUI open until it is closed")
    parser.add_argument("--video-dir", help="Save 25 fps camera frames for one trial")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--reference-sweep", action="store_true",
                        help="Evaluate all local soccer references at their upstream nominal ball positions")
    parser.add_argument("--trials", help="JSON manifest of independent ball, goal, style and strike corrections")
    parser.add_argument('--sequence-evidence', action='store_true',
                        help='Record whole-body and hip/knee evidence throughout the physical sequence')
    parser.add_argument('--running-diagnostic', action='store_true',
                        help='Locomotion probe: replace reference-error termination with absolute fall limits; never certifies shooting')
    parser.add_argument("--perturb-reset", action="store_true",
                        help="Seeded initial root xy +/-1 cm, yaw +/-0.01 rad and joint positions +/-0.005 rad")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(ROOT / ".pretrained_checkpoints/paid_official/policy_30000.onnx"))
    args = parser.parse_args()
    from targeted_shooting import controller_provenance, verify_target_provenance
    args.source_provenance = controller_provenance(ROOT)
    base_config = json.loads(Path(args.config).read_text())
    if 'controller_sources_sha256' in base_config:
        try:
            verify_target_provenance(base_config, ROOT)
        except ValueError as error:
            parser.error(str(error))
    if not 1 <= args.num_envs <= 24 or not 1 <= args.steps <= 1000:
        parser.error("Use 1..24 environments and 1..1000 policy steps")
    points = [(args.ball_x, args.ball_y)] * args.num_envs
    if args.grid:
        points = [(x, y) for y in (-.65, -.8, -.95, -1.1) for x in (.0, .14, .28)][:args.num_envs]
        if len(points) != args.num_envs:
            parser.error("The initial grid contains at most 12 positions")
    args.motion_files = [str(ROOT / "third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz")]
    args.trials_data = None
    args.targeted_config_sha256 = None
    args.targeted_provenance = None
    targeted = args.targeted_acceptance or args.target_x is not None or args.power is not None or args.speed_mps is not None
    if targeted:
        from targeted_shooting import resolve_targeted_shot, validate_target_config, verify_target_provenance, POWERS
        if args.trials or args.grid or args.reference_sweep or args.baseline or args.acceptance:
            parser.error('Targeted commands cannot be combined with other experiment modes')
        try:
            target_config = json.loads(Path(args.target_config).read_text())
            validate_target_config(target_config)
            args.targeted_provenance = verify_target_provenance(target_config, ROOT)
            if [args.ball_x, args.ball_y] != target_config['ball'] or [args.goal_x, args.goal_y] != target_config['goal']:
                raise ValueError('Targeted shooting pins the ball and physical goal to the calibrated geometry')
            if hashlib.sha256(Path(args.model).read_bytes()).hexdigest() != target_config['policy_sha256']:
                raise ValueError('Targeted shooting requires its pinned policy checkpoint')
            if args.targeted_acceptance:
                if args.target_x is not None or args.power is not None or args.speed_mps is not None:
                    raise ValueError('Targeted acceptance uses the entire frozen grid, without custom commands')
                if args.steps != target_config['trial_steps'] or args.seed not in target_config['acceptance_seeds']:
                    raise ValueError('Use the frozen duration and one of the acceptance seeds from --target-config')
                args.trials_data = [resolve_targeted_shot(target_config, args.style, x, power=p)
                                   for p in POWERS for x in target_config['target_x_m']]
                args.num_envs = len(args.trials_data)
                args.perturb_reset = True
            else:
                request = dict(speed_mps=args.speed_mps) if args.speed_mps is not None else dict(power=args.power or 'medium')
                trial = resolve_targeted_shot(target_config, args.style, args.target_x if args.target_x is not None else 0., **request)
                args.trials_data = [trial for _ in range(args.num_envs)]
            points = [t['ball'] for t in args.trials_data]
            args.targeted_config_sha256 = hashlib.sha256(Path(args.target_config).read_bytes()).hexdigest()
        except (ValueError, OSError) as error:
            parser.error(str(error))
    elif args.acceptance:
        if args.trials or args.grid or args.reference_sweep or args.baseline:
            parser.error("Acceptance uses the frozen config without other experiment flags")
        config = json.loads(Path(args.config).read_text())
        if hashlib.sha256(Path(args.model).read_bytes()).hexdigest() != config['policy_sha256']:
            parser.error('Acceptance requires the checkpoint pinned in the controller config')
        if args.steps != config['trial_steps']:
            parser.error('Acceptance requires the frozen trial duration')
        args.trials_data = [dict(name=f'{style}_c{i}', style=style, **condition, **control)
            for style, control in config['styles'].items()
            for i, condition in enumerate(config['acceptance_conditions'])]
        args.num_envs = len(args.trials_data)
        points = [t['ball'] for t in args.trials_data]
        args.perturb_reset = True
    elif args.trials:
        if args.grid or args.reference_sweep:
            parser.error("Use a trial manifest without grid/reference sweep")
        args.trials_data = json.loads(Path(args.trials).read_text())
        args.num_envs = len(args.trials_data)
        if not 1 <= args.num_envs <= 24:
            parser.error("Trial manifest requires 1..24 trials")
        points = [tuple(t.get("ball", [args.ball_x, args.ball_y])) for t in args.trials_data]
    elif not args.baseline and not args.reference_sweep:
        controller_config = json.loads(Path(args.config).read_text())
        if hashlib.sha256(Path(args.model).read_bytes()).hexdigest() != controller_config["policy_sha256"]:
            parser.error("Style configuration was calibrated with a different policy checkpoint")
        args.trials_data = [controller_config["styles"][args.style] for _ in points]

    if not args.reference_sweep:
        from runup_sequence import resolve_motion_files
        args.motion_files = resolve_motion_files(args.trials_data, ROOT, args.motion_files[0])
    if args.running_diagnostic and (not args.sequence_evidence or args.steps > 300):
        parser.error('Running diagnostic requires sequence evidence and at most 300 policy steps')
    if any(t.get('coordination') or t.get('strike_velocity') for t in (args.trials_data or [])) and not args.sequence_evidence:
        parser.error('Velocity/coordination control requires --sequence-evidence for physical support feedback')
    if any(t.get('coordination') and t.get('strike_velocity') for t in (args.trials_data or [])):
        parser.error('Do not combine two velocity controllers in one trial')
    from runup_sequence import approach_offset
    for trial in args.trials_data or []:
        approach_offset(trial, 0.)
        if args.sequence_evidence and trial.get('mode', 'hip') not in ('hip', 'knee'):
            parser.error('Sequence mode must be hip or knee')
    if (args.gui or args.video_dir) and args.num_envs != 1:
        parser.error("Rendered review requires exactly one environment")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=not args.gui, enable_cameras=bool(args.video_dir)).app
    held_env = None
    try:
        # Isaac Sim chooses its compatible NumPy before any numerical imports.
        if args.reference_sweep:
            import numpy as np
            args.motion_files = [str(p) for p in sorted((ROOT / "third_party/HumanoidSoccer/motions").glob("*/*.npz"))]
            args.num_envs = len(args.motion_files)
            points = []
            for path in args.motion_files:
                with np.load(path) as clip:
                    points.append(tuple(float(x) for x in clip["body_pos_w"][-1, 9, :2]))
        args.source_provenance = motion_provenance(args.source_provenance, args.motion_files)
        held_env = run(args, points)
        while args.gui and args.hold and app.is_running():
            app.update()
            time.sleep(.02)
    finally:
        if held_env is not None:
            held_env.close()
        app.close()


def run(args, points):
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils.math import quat_apply_inverse
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from free_shooting_metrics import ShotTracker, contact_point_events
    from soccer_onnx_policy import OnnxSoccerPolicy
    from shooting_control import StrikeResidual
    from targeted_shooting import TargetShotTracker

    upstream = ROOT / "third_party/HumanoidSoccer/source/whole_body_tracking"
    sys.path.insert(0, str(upstream))
    shim = types.ModuleType("soccer.assets")
    shim.ASSET_DIR = str(upstream / "soccer/assets")
    sys.modules["soccer.assets"] = shim
    import soccer.tasks
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    def ball_observation(env):
        from runup_sequence import approach_offset
        robot, ball = env.scene["robot"], env.scene["soccer_ball"]
        times = env.episode_length_buf.detach().cpu().numpy() * env.step_dt
        offset = torch.tensor([approach_offset(t, float(now)) for t, now in zip(trials, times)], device=robot.device)
        return quat_apply_inverse(robot.data.root_quat_w, ball.data.root_pos_w + offset - robot.data.root_pos_w)

    def goal_observation(env):
        robot = env.scene["robot"]
        goal = torch.tensor([[*g, .11] for g in policy_goals], device=robot.device) + env.scene.env_origins
        return quat_apply_inverse(robot.data.root_quat_w, goal - robot.data.root_pos_w)

    trials = args.trials_data or [{} for _ in points]
    goals = [t.get("goal", [args.goal_x, args.goal_y]) for t in trials]
    policy_goals = [t.get('policy_goal', g) for t, g in zip(trials, goals)]

    urdf = ROOT / "configs/paid_original/paid_main_resolved.urdf"
    urdf_hash = hashlib.sha256(urdf.read_bytes()).hexdigest()
    if urdf_hash != "34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761":
        raise ValueError("Physical foot asset differs from the audited seven-cylinder PAiD geometry")
    cfg = G1FlatEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 6.
    cfg.sim.device = "cuda:0"
    cfg.seed = args.seed
    cfg.episode_length_s = max(12., args.steps * .02 + 1)
    if args.running_diagnostic:
        import math
        from isaaclab.envs import mdp as physical_mdp
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        cfg.terminations.ee_body_pos = None
        cfg.terminations.anchor_pos_z = None
        cfg.terminations.anchor_ori = None
        cfg.terminations.physical_height = DoneTerm(
            func=physical_mdp.root_height_below_minimum, params={'minimum_height': .45})
        cfg.terminations.physical_orientation = DoneTerm(
            func=physical_mdp.bad_orientation, params={'limit_angle': math.radians(60.)})
    cfg.commands.motion.motion_files = args.motion_files
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    cfg.scene.robot.spawn.asset_path = str(urdf)
    cfg.scene.robot.spawn.fix_base = False
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn = sim.DomeLightCfg(intensity=1100)
    cfg.viewer.eye = (4., .9, 2.4)
    cfg.viewer.lookat = (.15, -1.6, .45)
    if args.video_dir:
        from isaaclab.sensors import CameraCfg
        cfg.scene.review_camera = CameraCfg(prim_path="{ENV_REGEX_NS}/ReviewCamera", update_period=.04,
            height=720, width=1280, data_types=["rgb"],
            spawn=sim.PinholeCameraCfg(focal_length=24., horizontal_aperture=36., clipping_range=(.05, 100.)))
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.target_point_pos = ObsTerm(func=ball_observation)
    cfg.observations.policy.target_destination_pos_local = ObsTerm(func=goal_observation)
    for key in cfg.commands.motion.pose_range:
        cfg.commands.motion.pose_range[key] = (0., 0.)
    for key in cfg.commands.motion.velocity_range:
        cfg.commands.motion.velocity_range[key] = (0., 0.)
    cfg.commands.motion.joint_position_range = (0., 0.)
    if args.perturb_reset:
        cfg.commands.motion.pose_range.update(x=(-.01, .01), y=(-.01, .01), yaw=(-.01, .01))
        cfg.commands.motion.joint_position_range = (-.005, .005)
    for key in ("physics_material", "add_joint_default_pos", "base_com", "base_external_force_torque", "push_robot"):
        setattr(cfg.events, key, None)
    cfg.scene.soccer_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim.SphereCfg(radius=.11, activate_contact_sensors=args.sequence_evidence,
            collision_props=sim.CollisionPropertiesCfg(),
            rigid_props=sim.RigidBodyPropertiesCfg(), mass_props=sim.MassPropertiesCfg(mass=.43),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.94, .94, .90)),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=.7, dynamic_friction=.55, restitution=.3)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.8, 2.8, .113)))
    for foot in ("right", "left"):
        setattr(cfg.scene, foot + "_ball_contact", ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + foot + "_ankle_roll_link", update_period=0.,
            history_length=1, filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"], debug_vis=False))
    env = RslRlVecEnvWrapper(gym.make("Tracking-Flat-G1-v0", cfg=cfg), clip_actions=100.)
    raw = env.unwrapped
    robot, ball = raw.scene["robot"], raw.scene["soccer_ball"]
    motion = raw.command_manager.get_term("motion")
    assert not cfg.scene.robot.spawn.fix_base and len(raw.scene.rigid_objects) == 1
    policy = OnnxSoccerPolicy(args.model, args.num_envs)
    policy.validate_joints(robot.joint_names)
    action_term = raw.action_manager.get_term("joint_pos")
    def cpu_array(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    strike = StrikeResidual([t.get("strike", {}) for t in trials], robot.joint_names,
                            cpu_array(action_term._scale), cpu_array(action_term._offset),
                            cpu_array(robot.data.soft_joint_pos_limits))
    reference = StrikeResidual([t.get("reference", {}) for t in trials], robot.joint_names,
                               cpu_array(action_term._scale), cpu_array(action_term._offset),
                               cpu_array(robot.data.soft_joint_pos_limits))
    from runup_sequence import CoordinationTargets
    coordination = CoordinationTargets([t.get('coordination') for t in trials], robot.joint_names,
        cpu_array(action_term._scale), cpu_array(action_term._offset), cpu_array(robot.data.soft_joint_pos_limits))
    from strike_velocity import StrikeVelocityPulse
    velocity_pulse = StrikeVelocityPulse([t.get('strike_velocity') for t in trials], robot.joint_names)
    cartesian = None
    if any('cartesian' in t for t in trials):
        from cartesian_strike import CartesianStrike
        cartesian = CartesianStrike(trials, robot.joint_names, cpu_array(action_term._scale),
            cpu_array(action_term._offset), cpu_array(robot.data.soft_joint_pos_limits), urdf)
    if list(raw.observation_manager.active_terms["policy"]) != policy.observation_names:
        raise ValueError("Observation term ordering differs from checkpoint metadata")
    native_command_update = motion._update_command

    def exact_reference(command, env_ids):
        ids = torch.as_tensor(env_ids, device=robot.device, dtype=torch.long)
        command.motion_idx[ids] = ids % len(args.motion_files)
        command.motion_length[ids] = command.motion.file_lengths[command.motion_idx[ids]]
        command.time_steps[ids] = 0

    motion._uniform_sampling = types.MethodType(exact_reference, motion)

    def no_wrap(command):
        cap = command.motion_length - 1
        finished = command.time_steps >= cap
        command.time_steps[finished] = cap[finished] - 1
        return native_command_update()

    motion._update_command = types.MethodType(no_wrap, motion)
    env.reset()
    motion.time_steps[:] = -1
    motion._update_command()
    state = ball.data.default_root_state.clone()
    state[:, :3] = torch.tensor([[x, y, .113] for x, y in points], device=ball.device) + raw.scene.env_origins
    state[:, 7:] = 0
    ball.write_root_pose_to_sim(state[:, :7])
    ball.write_root_velocity_to_sim(state[:, 7:])
    ball.reset()
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    raw.scene.update(raw.physics_dt)
    # Reset returns observations before reference and ball initialization; refresh them.
    obs = raw.observation_manager.compute()["policy"]
    policy.reset()
    # ManagerBasedRLEnv normally resets immediately after a termination.
    # Evaluation records that termination and never writes another physical state.
    native_reset_idx = raw._reset_idx
    suppressed_resets = set()
    def suppress_automatic_reset(env_ids):
        suppressed_resets.update(int(i) for i in env_ids.cpu().tolist())
    raw._reset_idx = suppress_automatic_reset
    initial_root = (robot.data.root_pos_w - raw.scene.env_origins).detach().cpu().numpy()
    initial_robot_state = dict(root_position=initial_root.tolist(),
        root_quaternion=cpu_array(robot.data.root_quat_w).tolist(),
        joint_position=cpu_array(robot.data.joint_pos).tolist())
    panel = None
    if args.gui or args.video_dir:
        from shooting_visuals import create_shooting_visuals
        create_shooting_visuals(points[0], goals[0], forward=trials[0].get('goal_forward'),
                               target_x=trials[0].get('target', {}).get('target_x'))
    if args.video_dir:
        from PIL import Image
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        if any(video_dir.glob('frame_*.png')):
            raise FileExistsError('Choose an empty video directory to preserve previous evidence')
        camera = raw.scene['review_camera']
        camera.set_world_poses_from_view(eyes=torch.tensor([cfg.viewer.eye], device=robot.device),
                                        targets=torch.tensor([cfg.viewer.lookat], device=robot.device))
        for _ in range(5):
            raw.sim.render()
    if args.gui:
        import omni.ui as ui
        panel = ui.Window('KineticKick | Free shooting', width=430, height=250)
        with panel.frame:
            with ui.VStack(spacing=5):
                ui.Label(f"Style: {trials[0].get('style', args.style)} | Free root | Physical contact")
                if args.sequence_evidence:
                    ui.Label(f"Coordination: {trials[0].get('mode', 'hip')} | Full-sequence evidence")
                status_label = ui.Label('Approaching the ball')
                ui.Label('Goal: 1.6 m wide / 1.1 m high')
                if 'target' in trials[0]:
                    target = trials[0]['target']
                    ui.Label('Experimental aim / strength commands')
                    ui.Label(f"Target x: {target['target_x']:+.2f} m | Speed: {target['speed_mps']:.2f} m/s")
    trackers, target_trackers = [], []
    for i, point in enumerate(points):
        goal = np.asarray(goals[i])
        delta = goal - np.asarray(point)
        if np.linalg.norm(delta) < 1.5:
            raise ValueError("Goal must be at least 1.5 m beyond initial ball")
        trackers.append(ShotTracker(initial_root=initial_root[i], initial_ball=(*point, .113),
                                   goal=goal, forward=trials[i].get('goal_forward', delta),
                                   style=trials[i].get("style", args.style)))
        target_trackers.append(TargetShotTracker(**trials[i]['target'], goal_y=goal[1])
                               if 'target' in trials[i] else None)
    active = np.ones(args.num_envs, bool)
    foot_indices = {side: robot.body_names.index(side + "_ankle_roll_link") for side in ("left", "right")}
    all_contacts = raw.scene["contact_forces"]
    support_indices = [all_contacts.body_names.index(side + "_ankle_roll_link") for side in ("left", "right")]
    nonfoot_indices = [i for i in range(len(all_contacts.body_names)) if i not in support_indices]
    detail_views = {}
    for side in ("left", "right"):
        sensor = raw.scene[side + "_ball_contact"]
        detail_views[side] = sensor._physics_sim_view.create_rigid_contact_view(
            "/World/envs/env_*/Robot/" + side + "_ankle_roll_link",
            ["/World/envs/env_*/SoccerBall"], max_contact_data_count=64 * args.num_envs)
    traces, contacts, kinematics = ([[] for _ in points] for _ in range(3))
    if args.sequence_evidence:
        from cartesian_strike import LegKinematics
        from scipy.spatial.transform import Rotation
        leg = LegKinematics(urdf)
        leg_indices = [robot.joint_names.index(n) for n in leg.names]
        masses = robot.root_physx_view.get_masses().to(robot.device)
        mass_weights = masses / masses.sum(dim=1, keepdim=True)
        termination_reasons = [[] for _ in points]
        import omni.usd
        from pxr import UsdPhysics
        ground_paths = [str(prim.GetPath()) for prim in omni.usd.get_context().get_stage().Traverse()
                        if str(prim.GetPath()).startswith(cfg.scene.terrain.prim_path + '/')
                        and prim.HasAPI(UsdPhysics.CollisionAPI)]
        if not ground_paths:
            raise RuntimeError('No ground colliders found for sequence support evidence')
        ground_views = {s: all_contacts._physics_sim_view.create_rigid_contact_view(
                            '/World/envs/env_*/Robot/' + s + '_ankle_roll_link', ground_paths,
                            max_contact_data_count=128 * args.num_envs) for s in ('left', 'right')}
        ball_robot_view = all_contacts._physics_sim_view.create_rigid_contact_view(
            '/World/envs/env_*/SoccerBall', ['/World/envs/env_*/Robot/' + name for name in robot.body_names],
            max_contact_data_count=256 * args.num_envs)
        print('SEQUENCE_GROUND_COLLIDERS', ground_paths, flush=True)
    substep = 0
    native_scene_update = raw.scene.update

    def measure_substep(dt):
        nonlocal substep
        native_scene_update(dt)
        substep += 1
        rpos = (robot.data.root_pos_w - raw.scene.env_origins).detach().cpu().numpy()
        bpos = (ball.data.root_pos_w - raw.scene.env_origins).detach().cpu().numpy()
        bvel = ball.data.root_lin_vel_w.detach().cpu().numpy()
        rvel = robot.data.root_lin_vel_w.detach().cpu().numpy()
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1, 1)).detach().cpu().numpy() * 180 / np.pi
        net = all_contacts.data.net_forces_w.detach().cpu().numpy()
        support = np.maximum(net[:, support_indices, 2], 0).sum(axis=1)
        nonfoot = np.linalg.norm(net[:, nonfoot_indices], axis=-1).max(axis=1)
        if args.sequence_evidence:
            q, dq = cpu_array(robot.data.joint_pos), cpu_array(robot.data.joint_vel)
            root_pose = cpu_array(robot.data.root_state_w[:, :7])
            com = cpu_array((robot.data.body_com_pos_w * mass_weights[:, :, None]).sum(dim=1) - raw.scene.env_origins)
            com_velocity = cpu_array((robot.data.body_com_lin_vel_w * mass_weights[:, :, None]).sum(dim=1))
            ground = {s: cpu_array(view.get_contact_force_matrix(dt)).reshape(args.num_envs, -1, 3)[:, :, 2].sum(axis=1)
                      for s, view in ground_views.items()}
            body_ball_forces = np.linalg.norm(cpu_array(ball_robot_view.get_contact_force_matrix(dt)), axis=-1)
            torque = cpu_array(robot.data.applied_torque)
        events = [[] for _ in points]
        for side, view in detail_views.items():
            forces, positions, normals, separations, counts, starts = view.get_contact_data(dt)
            counts, starts = counts.detach().cpu().numpy(), starts.detach().cpu().numpy()
            for i in np.flatnonzero(active):
                count, start = int(counts[i, 0]), int(starts[i, 0])
                if not count:
                    continue
                magnitude = forces[start:start+count].flatten()
                normal = normals[start:start+count].clone()
                foot_pos = robot.data.body_pos_w[i, foot_indices[side]]
                inward = (normal * (ball.data.root_pos_w[i] - foot_pos)).sum(-1) < 0
                normal[inward] *= -1
                quat = robot.data.body_quat_w[i, foot_indices[side]]
                quats = quat[None].expand(count, -1)
                normal_local = quat_apply_inverse(quats, normal)
                point_local = quat_apply_inverse(quats, positions[start:start+count] - foot_pos)
                point_events = contact_point_events(cpu_array(magnitude), cpu_array(point_local),
                    cpu_array(normal_local), foot=side, time_s=substep * raw.physics_dt)
                for event in point_events:
                    event.update(foot_quat_w=quat.cpu().tolist(), joint_pos=robot.data.joint_pos[i].cpu().tolist())
                events[i].extend(point_events)
                contacts[i].extend(point_events)
        for i in np.flatnonzero(active):
            event = max(events[i], key=lambda e: e["force_N"]) if events[i] else None
            row = dict(time_s=substep * raw.physics_dt, root_xyz=rpos[i].tolist(),
                       root_speed_xy=float(np.linalg.norm(rvel[i, :2])), root_tilt_deg=float(tilt[i]),
                       ball_xyz=bpos[i].tolist(), ball_velocity=bvel[i].tolist(),
                       contact_force_N=sum(e['force_N'] for e in events[i]),
                       contact_normal_foot=event["normal_foot"] if event else [0., 0., 0.],
                       contact_point_foot=event["point_foot"] if event else None,
                       foot=event["foot"] if event else "right", feet_support_N=float(support[i]),
                       nonfoot_contact_N=float(nonfoot[i]), terminated=False,
                       contacts=events[i], physics_dt_s=dt)
            if args.sequence_evidence:
                root = np.eye(4)
                root[:3, :3] = Rotation.from_quat([*root_pose[i, 4:7], root_pose[i, 3]]).as_matrix()
                root[:3, 3] = root_pose[i, :3]
                foot_pose, jacobian = leg.forward(q[i, leg_indices], root)
                offset = foot_pose[:3, :3] @ np.array([.06, 0., -.025])
                point_jacobian = jacobian[:3] + np.cross(jacobian[3:].T, offset).T
                parts = point_jacobian * dq[i, leg_indices][None, :]
                foot_index = foot_indices['right']
                point_velocity = cpu_array(robot.data.body_link_lin_vel_w[i, foot_index]) + np.cross(
                    cpu_array(robot.data.body_ang_vel_w[i, foot_index]), offset)
                row.update(joint_pos=q[i].tolist(), joint_vel=dq[i].tolist(),
                    joint_torque_estimate=torque[i].tolist(),
                    joint_target=cpu_array(robot.data.joint_pos_target[i]).tolist(),
                    joint_velocity_target=cpu_array(robot.data.joint_vel_target[i]).tolist(),
                    whole_body_com_xyz=com[i].tolist(), whole_body_com_velocity=com_velocity[i].tolist(),
                    support_foot_force_N={s: float(max(0., ground[s][i])) for s in ground},
                    robot_ball_contact_force_N={name: float(body_ball_forces[i, j])
                        for j, name in enumerate(robot.body_names) if body_ball_forces[i, j] > 0.},
                    foot_velocity_contributions=dict(hip=parts[:, :3].sum(axis=1).tolist(),
                        knee=parts[:, 3].tolist(), ankle=parts[:, 4:].sum(axis=1).tolist(),
                        base=(point_velocity-parts.sum(axis=1)).tolist()),
                    right_foot_point_velocity=point_velocity.tolist(),
                    ball_angular_velocity=cpu_array(ball.data.root_ang_vel_w[i]).tolist())
                row['coordination_phase'] = coordination.phases[i]
            trackers[i].update(row)
            if target_trackers[i] is not None:
                target_trackers[i].update(row)
            traces[i].append(row)

    raw.scene.update = measure_substep
    print("FREE_SHOOTING_READY", json.dumps({"policy": str(args.model), "points": points, "goals": goals,
          "physics_dt": raw.physics_dt, "policy_dt": raw.step_dt, "asset_sha256": urdf_hash}), flush=True)
    try:
        for step in range(args.steps):
            step_start = time.monotonic()
            policy_obs = reference.apply_reference(cpu_array(obs), cpu_array(motion.time_steps), raw.step_dt)
            action = policy.step(policy_obs, motion.time_steps.cpu().numpy())
            action = strike.apply(action, motion.time_steps.cpu().numpy())
            if args.sequence_evidence:
                support = cpu_array(ground_views['left'].get_contact_force_matrix(raw.physics_dt)).reshape(args.num_envs, -1, 3)[:, :, 2].sum(axis=1)
                distance = cpu_array(torch.linalg.vector_norm(robot.data.body_pos_w[:, foot_indices['right']] - ball.data.root_pos_w, dim=-1))
                action = coordination.apply(action, substep * raw.physics_dt, cpu_array(motion.time_steps), support, distance)
                pulse = velocity_pulse.apply(cpu_array(motion.time_steps), support)
                velocity_targets = torch.tensor(coordination.velocity_targets + pulse, device=robot.device, dtype=torch.float32)
                velocity_targets = torch.clamp(velocity_targets, -robot.data.joint_vel_limits, robot.data.joint_vel_limits)
                robot.set_joint_velocity_target(velocity_targets)
            if cartesian is not None:
                root_poses = cpu_array(robot.data.root_state_w[:, :7]).copy()
                root_poses[:, :3] -= cpu_array(raw.scene.env_origins)
                action = cartesian.apply(action, cpu_array(motion.time_steps), root_poses,
                                         cpu_array(robot.data.joint_pos), raw.step_dt)
            if args.sequence_evidence or 75 <= step <= 170:
                for i in np.flatnonzero(active):
                    kinematics[i].append(dict(time_s=substep * raw.physics_dt,
                        root_pose=cpu_array(robot.data.root_state_w[i, :7]).tolist(),
                        joint_pos=cpu_array(robot.data.joint_pos[i]).tolist(),
                        action=action[i].tolist(),
                        policy_ball_input_local=policy_obs[i, -6:-3].tolist(),
                        policy_goal_input_local=policy_obs[i, -3:].tolist(),
                        cartesian_error=cartesian.last_errors[i] if cartesian is not None else None,
                        feet={s: dict(position=cpu_array(robot.data.body_pos_w[i, j] - raw.scene.env_origins[i]).tolist(),
                                      quaternion=cpu_array(robot.data.body_quat_w[i, j]).tolist())
                              for s, j in foot_indices.items()}))
            with torch.no_grad():
                obs, _, done, _ = env.step(torch.tensor(action, device=robot.device))
            for i in np.flatnonzero(active & done.detach().cpu().numpy().astype(bool)):
                if args.sequence_evidence:
                    termination_reasons[i] = [name for name in raw.termination_manager.active_terms
                                             if bool(raw.termination_manager.get_term(name)[i])]
                terminal = {**traces[i][-1], "terminated": True, "time_s": substep * raw.physics_dt + 1e-6}
                trackers[i].update(terminal)
                if target_trackers[i] is not None:
                    target_trackers[i].update(terminal)
                traces[i].append(terminal)
                active[i] = False
            if args.video_dir and step % 2 == 0:
                rgb = camera.data.output['rgb'][0].detach().cpu().numpy()
                Image.fromarray(rgb[:, :, :3]).save(video_dir / f'frame_{step//2:04d}.png')
            if panel is not None:
                summary = trackers[0].summary()
                measure = target_trackers[0].summary() if target_trackers[0] is not None else None
                complete = summary['success'] and (measure is None or measure['target_success'])
                status_label.text = (f"t={(step+1)*raw.step_dt:.2f}s | contact={summary['measured_surface']}\n"
                    f"goal={summary['goal']} | standing={summary['standing_seconds']:.2f}s\n"
                    f"shot_check={complete} | failed={summary['failed']}")
                if args.sequence_evidence:
                    from runup_sequence import phase_label
                    phase = coordination.phases[0]
                    if phase in ('idle', 'released'):
                        phase = phase_label((step+1)*raw.step_dt, summary['contact_time_s'], summary['recovered'], summary['failed'])
                    status_label.text += f'\nPhase: {phase} | full sequence checked at trial end'
                if measure is not None:
                    error = '--' if measure['aim_error_m'] is None else f"{measure['aim_error_m']:.3f} m"
                    speed = '--' if measure['outgoing_speed_mps'] is None else f"{measure['outgoing_speed_mps']:.2f} m/s"
                    status_label.text += f"\nAim error: {error} | Speed: {speed}"
                time.sleep(max(0., raw.step_dt-(time.monotonic()-step_start)))
            if (step + 1) % 50 == 0:
                summaries = [t.summary() for t in trackers]
                print("FREE_SHOOTING_STEP", step + 1, json.dumps({
                    "contacts": sum(x["contact_time_s"] is not None for x in summaries),
                    "goals": sum(x["goal"] for x in summaries),
                    "recovered": sum(x["recovered"] for x in summaries),
                    "failed": sum(x["failed"] for x in summaries)}), flush=True)
            if not active.any():
                break
        records = []
        for tracker, target_tracker in zip(trackers, target_trackers):
            record = tracker.summary()
            if target_tracker is not None:
                record.update(target_tracker.summary())
                record['targeted_success'] = record['success'] and record['target_success']
            records.append(record)
        result = dict(policy=str(args.model), policy_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                      asset_sha256=urdf_hash, seed=args.seed, requested_style=args.style, points=points,
                      goals=goals, policy_goals=policy_goals, trials=trials, goal_width_m=1.6, goal_height_m=1.1,
                      root_fixed=False, post_initialization_state_writes=False, motion_files=args.motion_files,
                      contact_sampling="every 0.005 s, individual PhysX contact points and normals",
                      contact_evidence_version=2,
                      records=records, contacts=contacts, traces=traces,
                      kinematics=kinematics, joint_names=robot.joint_names)
        result.update(running_diagnostic=args.running_diagnostic,
            termination_policy='absolute_height_0.45m_tilt_60deg' if args.running_diagnostic else 'upstream_reference_tracking')
        result.update(perturb_reset=args.perturb_reset, initial_robot_state=initial_robot_state)
        result['automatic_resets_suppressed'] = sorted(suppressed_resets)
        result.update(controller_config=args.config,
                      config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                      rendered=bool(args.video_dir or args.gui))
        result.update(targeted_config=args.target_config if args.targeted_config_sha256 else None,
                      targeted_config_sha256=args.targeted_config_sha256)
        result.update(args.source_provenance)
        if args.sequence_evidence:
            result.update(sequence_evidence_version=1, termination_reasons=termination_reasons,
                torque_evidence='Implicit actuator estimated clipped PD torque; not a hardware measurement',
                ground_collision_paths=ground_paths,
                foot_velocity_point_local=[.06, 0., -.025], contribution_foot='right',
                joint_effort_limits=cpu_array(robot.data.joint_effort_limits).tolist(),
                joint_velocity_limits=cpu_array(robot.data.joint_vel_limits).tolist(),
                sequence_sources_sha256={name: hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest()
                                        for name in ('runup_sequence.py', 'runup_metrics.py', 'cartesian_strike.py', 'strike_velocity.py')})
            from runup_metrics import summarize_trial
            result['sequence_records'] = [summarize_trial(result, i) for i in range(args.num_envs)]
        result.update(args.targeted_provenance or {})
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print("FREE_SHOOTING_RESULT", str(output), json.dumps(result["records"]), flush=True)
        if args.sequence_evidence:
            print('RUNUP_SEQUENCE_RESULT', json.dumps(result['sequence_records']), flush=True)
            if panel is not None:
                status_label.text += f"\nFull sequence passed: {result['sequence_records'][0]['sequence_success']}"
        if args.gui and args.hold:
            raw.sim.pause()
            status_label.text += '\nTrial finished (paused)'
            print('FREE_SHOOTING_GUI_HELD', 'physics_paused', not raw.sim.is_playing(), flush=True)
            return env
    finally:
        raw.scene.update = native_scene_update
        raw._reset_idx = native_reset_idx
        if not (args.gui and args.hold):
            env.close()


if __name__ == "__main__":
    main()
