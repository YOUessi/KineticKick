"""Replay a full human reference through the released GMT policy on the original free G1.

GMT has 23 outputs. Six independent wrist axes follow their named reference.
Native GMT gains apply to the 23 mapped joints under the original G1 effort,
velocity and position limits. Six wrist drives keep their original gains.
GMT inputs follow Chen et al., humanoid-general-motion-tracking, revision 2a590de.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import v160_tracking_cpu as phys

REVISION = "2a590de25a1eb08e47491977a738549c22f16e1f"
CHECKPOINT = "bf965da77571a5bb0dbb291b349b387a258c1b34cb86173cc7502f9ec2015755"
LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"]
NAMES = (
    [s + "_" + j for s in ("left", "right") for j in LEG]
    + ["waist_yaw", "waist_roll", "waist_pitch"]
    + [s + "_" + j for s in ("left", "right") for j in ARM]
)
DEFAULT = np.array(
    [-0.2, 0, 0, 0.4, -0.2, 0] * 2 + [0] * 3 + [0, 0.4, 0, 1.2, 0, -0.4, 0, 1.2]
)
KP = np.array([100, 100, 100, 150, 40, 40] * 2 + [150] * 3 + [40] * 8)
KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [4] * 3 + [5] * 8)
FUTURE = np.array([1] + list(range(5, 100, 5)))


def load_adapter(path):
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def pulse(t, knots):
    return float(np.interp(t, knots, [0.0, 1.0, 1.0, 0.0], left=0.0, right=0.0))


def make_control(ref, schema, meta, policy, opts, recording=None):
    mode = opts.drive
    adapter = getattr(opts, "adapter", {})
    source_event = adapter.get("event_s", 2.94)
    legs = [schema["joint_names"].index("right_" + j + "_joint") for j in LEG]
    mid = np.array(adapter.get("swing", np.zeros(12))).reshape(2, 6)
    ids = [schema["joint_names"].index(n + "_joint") for n in NAMES]
    if len(set(ids)) != 23:
        raise ValueError("GMT joint map mismatch")
    k = np.fromstring(meta["joint_stiffness"], sep=",")[ids]
    d = np.fromstring(meta["joint_damping"], sep=",")[ids]
    rot = phys.rotation(ref["body_quat_w"][:, 0])
    features = np.column_stack(
        [
            ref["body_pos_w"][:, 0, 2],
            rot.as_euler("xyz")[:, :2],
            rot.inv().apply(ref["body_lin_vel_w"][:, 0]),
            rot.inv().apply(ref["body_ang_vel_w"][:, 0])[:, 2],
            ref["joint_pos"][:, ids],
        ]
    )
    if opts.settle:
        # The source cuts off mid-step. A finite continuation lets the lifted foot land,
        # decelerates the pelvis and levels the legs after the final motion sample.
        final = features[-1].copy()
        standing = np.r_[0.76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DEFAULT]
        u = np.linspace(0, 1, 61)[1:]
        w = (u * u * (3 - 2 * u))[:, None]
        features = np.concatenate((features, (1 - w) * final + w * standing))
    swing_plan = rot
    history = deque(np.zeros((20, 74)), maxlen=20)
    last = np.zeros(23)

    def control(ctx, unused):
        nonlocal last
        frame = ctx["frame"]
        quat = phys.rotation(ctx["root_quat"])
        q, dq = ctx["q"][ids], ctx["dq"][ids]
        odq = dq.copy()
        odq[[4, 5, 10, 11]] = 0.0
        prop = np.r_[
            0.25 * quat.inv().apply(ctx["root_velocity"][3:]),
            quat.as_euler("xyz")[:2],
            q - DEFAULT,
            0.05 * odq,
            last,
        ]
        clock = round(ctx["t"] / 0.02)
        plan = features[np.minimum(clock + FUTURE, len(features) - 1)].copy()
        if opts.feedback and clock < len(ref["joint_pos"]):
            error = ref["body_pos_w"][frame, 0, :2] - ctx["root_pos"][:2]
            cue = np.r_[np.clip(opts.feedback * error, -0.9, 0.9), 0.0]
            plan[:, 3:6] += (
                swing_plan[np.minimum(frame + FUTURE, len(ref["joint_pos"]) - 1)]
                .inv()
                .apply(cue)
            )
        if adapter:
            # Balance the horizontal approach before the source kick, then track the
            # measured pelvis heading through the change of support.
            look = np.minimum(clock + FUTURE, len(ref["joint_pos"]) - 1)
            goal_v = ref["body_lin_vel_w"][look, 0, :].copy()
            goal_v[:, 2] = 0.0
            amp = np.array([adapter.get("vx", 0.0), adapter.get("vy", 0.0), 0.0])
            gain = np.interp(
                (clock + FUTURE) * 0.02,
                [1.55, 1.92, 2.86, 3.15],
                [0.0, 1.0, 1.0, 0.0],
                left=0.0,
                right=0.0,
            )
            plan[:, 3:6] += swing_plan[look].inv().apply(goal_v * amp) * gain[:, None]
            if clock < len(ref["joint_pos"]):
                yaw = swing_plan[frame].as_euler("xyz")[2] - quat.as_euler("xyz")[2]
                plan[:, 6] += adapter.get("heading", 0.0) * np.arctan2(
                    np.sin(yaw), np.cos(yaw)
                )
        obs = np.r_[plan.ravel(), prop, np.asarray(history).ravel()]
        with torch.inference_mode():
            last = policy(torch.tensor(obs[None], dtype=torch.float32)).numpy().ravel()
        if recording is not None:
            recording["network_input"].append(obs.astype(np.float32))
            recording["network_action"].append(last.copy())
        history.append(prop)
        target = ref["joint_pos"][frame].astype(float).copy()
        desired = DEFAULT + 0.5 * np.clip(last, -10, 10)
        if mode == "matched":
            effort = np.clip(
                KP * (desired - q) - KD * dq,
                -ctx["limits"][ids, 3],
                ctx["limits"][ids, 3],
            )
            target[ids] = q + (effort + d * dq) / k
        else:
            target[ids] = desired
        if adapter:
            rel = ctx["t"] - source_event
            target[legs] += (
                pulse(rel, [-0.27, -0.15, -0.095, -0.015]) * mid[0]
                + pulse(rel, [-0.13, -0.055, 0.105, 0.29]) * mid[1]
            )
        plant = adapter.get("plant_transfer")
        if plant is not None:
            # Adjust the loaded left hip/ankle and trunk on either side of impact
            # so lateral momentum is corrected before the next support exchange.
            names = [
                "left_hip_roll",
                "left_ankle_roll",
                "waist_roll",
                "left_hip_pitch",
                "left_knee",
                "right_hip_roll",
            ]
            joints = [schema["joint_names"].index(n + "_joint") for n in names]
            age = ctx["t"] - source_event
            weights = np.array(
                [
                    pulse(age, [-0.245, -0.14, -0.065, 0.055]),
                    pulse(age, [-0.025, 0.07, 0.20, 0.36]),
                ]
            )
            target[joints] += weights @ np.asarray(plant).reshape(2, len(names))
        return target

    return control


class GuardedStep:
    """Run a joint torque correction at each 5 ms step while respecting physical drive effort caps."""

    def __init__(self, scene, robot, drives, order, limits, names, cfg, recording=None):
        self.recording = recording
        self.scene, self.robot, self.drives = scene, robot, drives
        self.order, self.limits = order, limits
        self.ix = np.array(
            [
                j
                for j, n in enumerate(names)
                if any(t in n for t in ("ankle", "hip", "knee", "waist"))
            ]
        )
        self.k = np.array([KP[NAMES.index(names[j][:-6])] for j in self.ix])
        self.margin = np.array(
            [
                cfg.get("ankle_margin", 0.115)
                if "ankle" in names[j]
                else cfg.get("joint_margin", 0.14)
                for j in self.ix
            ]
        )
        self.kp = cfg.get("guard_k", 210.0)
        self.kd = cfg.get("guard_d", 4.5)
        self.horizon = cfg.get("guard_preview", 0.030)

    def step(self):
        # Restoring the issued target after stepping preserves the 50 Hz input
        # exactly and prevents the 200 Hz correction from accumulating.
        j = self.ix
        q = self.robot.qpos[self.order[j]]
        dq = self.robot.qvel[self.order[j]]
        lo = self.limits[j, 0] + self.margin
        hi = self.limits[j, 1] - self.margin
        predicted = q + self.horizon * dq
        error = predicted - np.clip(predicted, lo, hi)
        changed = np.flatnonzero(abs(error) > 0)
        saved = []
        for v in changed:
            drive = self.drives[j[v]]
            original = float(np.asarray(drive.drive_target).ravel()[0])
            saved.append(original)
            torque = -self.kp * error[v] - self.kd * (
                max(0.0, dq[v]) if error[v] > 0 else min(0.0, dq[v])
            )
            drive.set_drive_target(original + torque / self.k[v])
        if self.recording is not None:
            self.recording["drive_targets_200hz_rad"].append(
                [float(drive.drive_target[0]) for drive in self.drives]
            )
        self.scene.step()
        for v, value in zip(changed, saved):
            self.drives[j[v]].set_drive_target(value)


def guarded(scene, robot, drives, order, limits, names, cfg, recording=None):
    return (
        GuardedStep(scene, robot, drives, order, limits, names, cfg, recording)
        if cfg.get("fast_guard")
        else scene
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=phys.DEFAULT_ASSETS)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=phys.REPO / "experiments/v163_gmt/assets/pretrained.pt",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--drive", choices=["matched", "original", "native"], default="matched"
    )
    p.add_argument("--adapter", type=Path)
    p.add_argument("--feedback", type=float, default=0.0)
    p.add_argument(
        "--paid-baseline",
        action="store_true",
        help="Run the released PAiD baseline in the same ball and goal world frame",
    )
    p.add_argument("--settle", action="store_true")
    p.add_argument("--ball", nargs=2, type=float)
    p.add_argument("--direction", nargs=2, type=float, default=[1, 0])
    p.add_argument("--goal-distance", type=float, default=2.8)
    p.add_argument("--duration", type=float, default=14.0)
    args = p.parse_args()
    args.adapter = load_adapter(args.adapter)
    torch.set_num_threads(1)
    if not args.paid_baseline and phys.digest(args.checkpoint) != CHECKPOINT:
        raise ValueError("Unknown GMT checkpoint")
    cfg = json.loads(phys.PROTOCOL.read_text())
    assets = args.assets.resolve()
    for name, h in cfg["asset_hashes"].items():
        if phys.digest(assets / name) != h:
            raise ValueError("Changed robot or deployment asset")
    schema = json.loads((assets / phys.SCHEMA).read_text())
    with np.load(args.reference, allow_pickle=False) as ref:
        if (
            ref["joint_names"].tolist() != schema["joint_names"]
            or ref["body_names"].tolist() != schema["body_names"]
            or ref["fps"].tolist() != [50]
        ):
            raise ValueError("Noncanonical human source")
    if not 0 < args.duration <= 14 or not np.isclose(
        args.duration * 50, round(args.duration * 50)
    ):
        p.error("Duration must consist of complete control ticks up to 14 s")
    if args.adapter.get("fast_guard") and args.drive != "native":
        p.error("Fast guard is calibrated for native GMT drive gains")
    if args.paid_baseline and (args.adapter or args.feedback or args.settle):
        p.error("PAiD baseline does not use GMT interventions")
    if args.ball and (
        not np.isfinite([*args.ball, *args.direction, args.goal_distance]).all()
        or np.linalg.norm(args.direction) == 0
        or args.goal_distance <= 0
    ):
        p.error(
            "Expected a finite football, a nonzero direction and a positive goal distance"
        )
    cfg["clip"] = str(args.reference.resolve())
    cfg["duration_s"] = args.duration
    if args.ball:
        xy = np.array(args.ball)
        direction = np.asarray(args.direction) / np.linalg.norm(args.direction)
        cfg["ball_initial_xyz_m"] = [*xy, cfg["ball_radius_m"]]
        cfg["goal_xyz_m"][:2] = (xy + args.goal_distance * direction).tolist()
        cfg["goal_forward_xy"] = direction.tolist()
    if args.paid_baseline:
        args.output.mkdir(parents=True, exist_ok=False)
        phys.save_json(
            args.output / "trial.json",
            {
                key: cfg[key]
                for key in [
                    "clip",
                    "ball_initial_xyz_m",
                    "goal_xyz_m",
                    "goal_forward_xy",
                ]
            },
        )
        with phys.cpu_policy(assets, cfg) as (meta, act):
            phys.run(assets, args.output, cfg, schema, meta, act)
        return
    policy = torch.jit.load(str(args.checkpoint), map_location="cpu").eval()
    recording = {
        key: []
        for key in ("network_input", "network_action", "drive_targets_200hz_rad")
    }
    with phys.cpu_policy(assets, cfg) as (meta, _):

        def factory(ref, schema):
            return ref, make_control(ref, schema, meta, policy, args, recording)

        args.output.mkdir(parents=True, exist_ok=False)
        phys.save_json(
            args.output / "trial.json",
            {
                "controller": "GMT 23 and 6 reference wrists",
                "checkpoint_sha256": CHECKPOINT,
                "upstream_revision": REVISION,
                "code_sha256": phys.digest(Path(__file__)),
                "reference_sha256": phys.digest(args.reference),
                "policy_inputs": "GMT only; PAiD metadata for the plant, PAiD actions unused",
                "physics_plant_source_sha256": phys.digest(Path(phys.__file__)),
                "drive": args.drive,
                "clip": cfg["clip"],
                "adapter": args.adapter,
                "duration_s": args.duration,
                "feedback": args.feedback,
                "settle": args.settle,
                "ball_initial_xyz_m": cfg["ball_initial_xyz_m"],
                "goal_xyz_m": cfg["goal_xyz_m"],
                "goal_forward_xy": cfg["goal_forward_xy"],
            },
        )

        def plant(assets, cfg, meta, schema):
            result = phys.make_plant(assets, cfg, meta, schema)
            if args.drive == "native":
                for j, n in enumerate(NAMES):
                    i = schema["joint_names"].index(n + "_joint")
                    result[4][i].set_drive_properties(
                        KP[j], KD[j], result[6][i, 3], mode="force"
                    )
            return (
                guarded(
                    result[0],
                    result[2],
                    result[4],
                    result[5],
                    result[6],
                    schema["joint_names"],
                    args.adapter,
                    recording,
                ),
                *result[1:],
            )

        phys.run(
            assets,
            args.output,
            cfg,
            schema,
            meta,
            lambda frame, obs: np.zeros(29),
            controller_factory=factory,
            plant_factory=plant,
        )
    signals = args.output / "gmt_commands.npz"
    np.savez_compressed(
        signals, **{k: np.asarray(v) for k, v in recording.items() if v}
    )
    phys.save_json(
        args.output / "gmt_recording.json",
        {
            "sha256": phys.digest(signals),
            "control_dt_s": 0.02,
            "physics_dt_s": 0.005,
            "network_output_axes": [n + "_joint" for n in NAMES],
            "target_axes": schema["joint_names"],
            "trace_note": "rollout.npz legacy PAiD columns are placeholders. Actual GMT tensors and guarded joint targets are in gmt_commands.npz.",
        },
    )


if __name__ == "__main__":
    main()
