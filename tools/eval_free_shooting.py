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
    parser.add_argument("--style", choices=("instep", "inside", "outside"), default="instep")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--reference-sweep", action="store_true",
                        help="Evaluate all local soccer references at their upstream nominal ball positions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(ROOT / ".pretrained_checkpoints/paid_official/policy_30000.onnx"))
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 24 or not 1 <= args.steps <= 1000:
        parser.error("Use 1..24 environments and 1..1000 policy steps")
    points = [(args.ball_x, args.ball_y)] * args.num_envs
    if args.grid:
        points = [(x, y) for y in (-.65, -.8, -.95, -1.1) for x in (.0, .14, .28)][:args.num_envs]
        if len(points) != args.num_envs:
            parser.error("The initial grid contains at most 12 positions")
    args.motion_files = [str(ROOT / "third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz")]

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app
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
        run(args, points)
    finally:
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
    from free_shooting_metrics import ShotTracker
    from soccer_onnx_policy import OnnxSoccerPolicy

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
        goal = torch.tensor([args.goal_x, args.goal_y, .11], device=robot.device) + env.scene.env_origins
        return quat_apply_inverse(robot.data.root_quat_w, goal - robot.data.root_pos_w)

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
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.target_point_pos = ObsTerm(func=ball_observation)
    cfg.observations.policy.target_destination_pos_local = ObsTerm(func=goal_observation)
    for key in cfg.commands.motion.pose_range:
        cfg.commands.motion.pose_range[key] = (0., 0.)
    for key in cfg.commands.motion.velocity_range:
        cfg.commands.motion.velocity_range[key] = (0., 0.)
    cfg.commands.motion.joint_position_range = (0., 0.)
    for key in ("physics_material", "add_joint_default_pos", "base_com", "base_external_force_torque", "push_robot"):
        setattr(cfg.events, key, None)
    cfg.scene.soccer_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim.SphereCfg(radius=.11, collision_props=sim.CollisionPropertiesCfg(),
            rigid_props=sim.RigidBodyPropertiesCfg(), mass_props=sim.MassPropertiesCfg(mass=.43),
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
    initial_root = (robot.data.root_pos_w - raw.scene.env_origins).detach().cpu().numpy()
    goal = np.array([args.goal_x, args.goal_y])
    trackers = []
    for i, point in enumerate(points):
        delta = goal - np.asarray(point)
        if np.linalg.norm(delta) < 1.5:
            raise ValueError("Goal must be at least 1.5 m beyond initial ball")
        trackers.append(ShotTracker(initial_root=initial_root[i], initial_ball=(*point, .113),
                                   goal=goal, forward=delta, style=args.style))
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
    traces, contacts = [[] for _ in points], [[] for _ in points]
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
                k = start + int(torch.argmax(magnitude))
                normal = normals[k].clone()
                foot_pos = robot.data.body_pos_w[i, foot_indices[side]]
                if torch.dot(normal, ball.data.root_pos_w[i] - foot_pos) < 0:
                    normal = -normal
                quat = robot.data.body_quat_w[i, foot_indices[side]]
                normal_local = quat_apply_inverse(quat[None], normal[None])[0].cpu().tolist()
                point_local = quat_apply_inverse(quat[None], (positions[k] - foot_pos)[None])[0].cpu().tolist()
                event = dict(foot=side, force_N=float(magnitude.sum()), normal_foot=normal_local,
                             point_foot=point_local, time_s=substep * raw.physics_dt)
                events[i].append(event)
                contacts[i].append(event)
        for i in np.flatnonzero(active):
            event = max(events[i], key=lambda e: e["force_N"]) if events[i] else None
            row = dict(time_s=substep * raw.physics_dt, root_xyz=rpos[i].tolist(),
                       root_speed_xy=float(np.linalg.norm(rvel[i, :2])), root_tilt_deg=float(tilt[i]),
                       ball_xyz=bpos[i].tolist(), ball_velocity=bvel[i].tolist(),
                       contact_force_N=event["force_N"] if event else 0.,
                       contact_normal_foot=event["normal_foot"] if event else [0., 0., 0.],
                       contact_point_foot=event["point_foot"] if event else None,
                       foot=event["foot"] if event else "right", feet_support_N=float(support[i]),
                       nonfoot_contact_N=float(nonfoot[i]), terminated=False)
            trackers[i].update(row)
            traces[i].append(row)

    raw.scene.update = measure_substep
    print("FREE_SHOOTING_READY", json.dumps({"policy": str(args.model), "points": points, "goal": goal.tolist(),
          "physics_dt": raw.physics_dt, "policy_dt": raw.step_dt, "asset_sha256": urdf_hash}), flush=True)
    try:
        for step in range(args.steps):
            action = policy.step(obs.detach().cpu().numpy(), motion.time_steps.cpu().numpy())
            with torch.no_grad():
                obs, _, done, _ = env.step(torch.tensor(action, device=robot.device))
            for i in np.flatnonzero(active & done.detach().cpu().numpy().astype(bool)):
                trackers[i].update({**traces[i][-1], "terminated": True, "time_s": substep * raw.physics_dt + 1e-6})
                active[i] = False
            if (step + 1) % 50 == 0:
                summaries = [t.summary() for t in trackers]
                print("FREE_SHOOTING_STEP", step + 1, json.dumps({
                    "contacts": sum(x["contact_time_s"] is not None for x in summaries),
                    "goals": sum(x["goal"] for x in summaries),
                    "recovered": sum(x["recovered"] for x in summaries),
                    "failed": sum(x["failed"] for x in summaries)}), flush=True)
            if not active.any():
                break
        result = dict(policy=str(args.model), policy_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                      asset_sha256=urdf_hash, seed=args.seed, requested_style=args.style, points=points,
                      goal=goal.tolist(), goal_width_m=1.6, goal_height_m=1.1,
                      root_fixed=False, post_initialization_state_writes=False, motion_files=args.motion_files,
                      contact_sampling="every 0.005 s, PhysX contact points and normals",
                      records=[t.summary() for t in trackers], contacts=contacts, traces=traces)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print("FREE_SHOOTING_RESULT", str(output), json.dumps(result["records"]), flush=True)
    finally:
        raw.scene.update = native_scene_update
        env.close()


if __name__ == "__main__":
    main()
