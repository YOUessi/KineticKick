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
        robot, ball = env.scene["robot"], env.scene["soccer_ball"]
        return quat_apply_inverse(robot.data.root_quat_w, ball.data.root_pos_w - robot.data.root_pos_w)

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
        spawn=sim.SphereCfg(radius=.11, collision_props=sim.CollisionPropertiesCfg(),
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
                ui.Label(f'Style: {args.style} | Free root | Physical contact')
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
            if cartesian is not None:
                root_poses = cpu_array(robot.data.root_state_w[:, :7]).copy()
                root_poses[:, :3] -= cpu_array(raw.scene.env_origins)
                action = cartesian.apply(action, cpu_array(motion.time_steps), root_poses,
                                         cpu_array(robot.data.joint_pos), raw.step_dt)
            if 75 <= step <= 170:
                for i in np.flatnonzero(active):
                    kinematics[i].append(dict(time_s=substep * raw.physics_dt,
                        root_pose=cpu_array(robot.data.root_state_w[i, :7]).tolist(),
                        joint_pos=cpu_array(robot.data.joint_pos[i]).tolist(),
                        action=action[i].tolist(),
                        cartesian_error=cartesian.last_errors[i] if cartesian is not None else None,
                        feet={s: dict(position=cpu_array(robot.data.body_pos_w[i, j] - raw.scene.env_origins[i]).tolist(),
                                      quaternion=cpu_array(robot.data.body_quat_w[i, j]).tolist())
                              for s, j in foot_indices.items()}))
            with torch.no_grad():
                obs, _, done, _ = env.step(torch.tensor(action, device=robot.device))
            for i in np.flatnonzero(active & done.detach().cpu().numpy().astype(bool)):
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
                    f"complete={complete} | failed={summary['failed']}")
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
        result.update(perturb_reset=args.perturb_reset, initial_robot_state=initial_robot_state)
        result['automatic_resets_suppressed'] = sorted(suppressed_resets)
        result.update(controller_config=args.config,
                      config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                      rendered=bool(args.video_dir or args.gui))
        result.update(targeted_config=args.target_config if args.targeted_config_sha256 else None,
                      targeted_config_sha256=args.targeted_config_sha256)
        result.update(args.source_provenance)
        result.update(args.targeted_provenance or {})
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print("FREE_SHOOTING_RESULT", str(output), json.dumps(result["records"]), flush=True)
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
