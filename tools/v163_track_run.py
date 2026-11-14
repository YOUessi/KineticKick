"""Execute original-asset G1 with source-conditioned joint drives at 200 Hz."""

import argparse, gzip, json, os, subprocess
from pathlib import Path
import numpy as np
import sapien
import v160_tracking_cpu as base
from v163_track_plan import plan
from free_shooting_metrics import goal_crossing
from scipy.spatial.transform import Rotation


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--duration", type=float, default=14)
    cli.add_argument("--ball-xy", type=float, nargs=2, default=[1.18723, -0.80613])
    cli.add_argument("--assets", type=Path, default=base.DEFAULT_ASSETS)
    args = cli.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    setup = json.loads(base.PROTOCOL.read_text())
    schema = json.loads((args.assets / base.SCHEMA).read_text())
    cfg = json.loads(args.config.read_text())
    ref = dict(np.load(args.reference))
    dt = setup["physics_dt_s"]
    forward = np.array([0.9978, 0.0663])
    forward /= np.linalg.norm(forward)
    setup["ball_initial_xyz_m"] = [*args.ball_xy, 0.113]
    setup["goal_forward_xy"] = forward.tolist()
    goal = np.r_[args.ball_xy + forward * 2.8, 0.11]
    setup["goal_xyz_m"] = goal.tolist()
    base.save_json(
        out / "trial.json",
        dict(
            config=cfg,
            reference=str(args.reference.resolve()),
            reference_sha256=base.digest(args.reference),
            code_sha256={
                f: base.digest(Path(__file__).with_name(f))
                for f in (
                    "v163_track_run.py",
                    "v163_track_wbc.py",
                    "v163_track_plan.py",
                )
            },
            scene=setup,
            duration_s=args.duration,
        ),
    )
    with base.cpu_policy(args.assets, setup) as (meta, _):
        scene, system, robot, ball, drives, order, limits, gains, links, ids = (
            base.make_plant(args.assets, setup, meta, schema)
        )
    robot.set_root_pose(sapien.Pose(ref["body_pos_w"][0, 0], ref["body_quat_w"][0, 0]))
    robot.set_root_linear_velocity(ref["body_lin_vel_w"][0, 0])
    robot.set_root_angular_velocity(ref["body_ang_vel_w"][0, 0])
    robot.set_qpos(ref["joint_pos"][0, np.argsort(order)])
    robot.set_qvel(ref["joint_vel"][0, np.argsort(order)])
    rows = {}
    commands = []
    counts = {}
    scored = False
    first = None
    blist = [links[n] for n in schema["body_names"]]
    last = ball.entity.pose.p.copy()
    p = plan(ref, cfg)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    with (
        subprocess.Popen(
            [
                cfg["dynamics_python"],
                "-s",
                "-u",
                str(Path(__file__).with_name("v163_track_wbc.py")),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        ) as proc,
        gzip.open(out / "native_contacts.jsonl.gz", "wt", compresslevel=1) as log,
    ):
        proc.stdin.write(
            json.dumps([cfg, schema["joint_names"], str(args.reference.resolve())])
            + "\n"
        )
        for i in range(round(args.duration / dt)):
            root = robot.root_pose
            state = np.r_[
                root.p,
                root.q,
                robot.root_linear_velocity,
                robot.root_angular_velocity,
                robot.qpos[order],
                robot.qvel[order],
            ]
            proc.stdin.write(json.dumps([i, state.tolist()]) + "\n")
            proc.stdin.flush()
            goalq = json.loads(proc.stdout.readline())
            for j, x in zip(drives, goalq):
                j.set_drive_target(x)
            commands.append(goalq)
            scene.step()
            root = robot.root_pose
            ct, loads, slip, nonfoot, ball_links = base.contacts_for_step(
                system, ids, dt
            )
            log.write(json.dumps([i + 1, ct], separators=(",", ":")) + "\n")
            if ball_links and first is None:
                first = i + 1
            for n in ball_links:
                counts[n] = counts.get(n, 0) + 1
            now = ball.entity.pose.p
            scored |= goal_crossing(
                last,
                now,
                goal=goal,
                forward=forward,
                width=setup["goal_width_height_m"][0],
                height=setup["goal_width_height_m"][1],
                radius=setup["ball_radius_m"],
            )
            last = now.copy()
            row = {
                "time_s": (i + 1) * dt,
                "source_time_s": p["source_time"][i + 1],
                "joint_q_rad": robot.qpos[order],
                "joint_dq_rad_s": robot.qvel[order],
                "root_pose_wxyz": np.r_[root.p, root.q],
                "root_velocity_w": np.r_[
                    robot.root_linear_velocity, robot.root_angular_velocity
                ],
                "body_position_w_m": [l.entity.pose.p for l in blist],
                "body_quat_wxyz": [l.entity.pose.q for l in blist],
                "ball_position_w_m": now,
                "ball_velocity_w_mps": ball.linear_velocity,
                "ball_angular_velocity_w_rad_s": ball.angular_velocity,
                "foot_ground_normal_load_N": loads,
                "foot_patch_slide_mps": slip,
                "nonfoot_net_contact_peak_N": nonfoot,
                "robot_ball_contact": bool(ball_links),
                "foot_link_origin_speed_xy_mps": [
                    np.linalg.norm(
                        links[n].linear_velocity[:2]
                        + np.cross(
                            links[n].angular_velocity,
                            links[n].entity.pose.p
                            - (links[n].entity.pose * links[n].cmass_local_pose).p,
                        )[:2]
                    )
                    for n in base.FEET
                ],
            }
            for k, val in row.items():
                rows.setdefault(k, []).append(np.asarray(val).copy())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("WBC worker failed")
    trace = {k: np.asarray(v) for k, v in rows.items()}
    trace["motor_position_commands"] = np.asarray(commands)
    np.savez_compressed(out / "rollout.npz", **trace)
    adjusted = setup.copy()
    adjusted["control_dt_s"] = 0.02 * cfg["slow"]
    # Physical 200Hz data are evaluated on the original source clock.
    true = base.metrics(trace, ref, adjusted, schema, limits, first, scored, counts)
    true.update(
        source_joint_names=schema["joint_names"],
        source_body_names=schema["body_names"],
        source_clip=str(args.reference),
        duration_s=args.duration,
        additional_state_writes_after_initialization=0,
    )
    base.save_json(out / "tracking_result.json", true)
    print(
        json.dumps(
            {
                k: true[k]
                for k in (
                    "first_robot_ball_s",
                    "goal_crossed_whole_ball",
                    "first_nonfoot_net_force_above_20N_s",
                    "first_height_below_0p45m_s",
                    "final_standing_seconds_cpu_definition",
                    "phase_metrics",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
