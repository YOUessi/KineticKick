"""Execute the frozen TWIST2 whole-body tracker on the original free-root G1 in CPU PhysX.

Adapted from Amazon FAR TWIST2 (MIT, copyright 2025 Yanjie Ze), deploy_real
server_low_level_g1_sim.py and server_motion_lib.py at REVISION below. Keep the
original per-axis URDF effort bounds, armature and collision geometry. Only
joint drive targets change after initialization. No ball feedback to the actor.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import v160_tracking_cpu as base
from scipy.spatial.transform import Slerp

REVISION = "b06178f19a22f2138cbd31f60c6d494bc263f67d"
MODEL_HASH = "126ab9089b6ca0be8b6582fc9ef17b0b6c6d6577378f9324a2e10d580fe48f2e"
LEGS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARMS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
NAMES = [f"{side}_{part}_joint" for side in ("left", "right") for part in LEGS]
NAMES += [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
NAMES += [f"{side}_{part}_joint" for side in ("left", "right") for part in ARMS]
DEFAULT = np.array(
    [-0.2, 0, 0, 0.4, -0.2, 0] * 2
    + [0, 0, 0]
    + [0, 0.4, 0, 1.2, 0, 0, 0]
    + [0, -0.4, 0, 1.2, 0, 0, 0]
)
KP = np.array(
    [100, 100, 100, 150, 40, 40] * 2 + [150] * 3 + [40, 40, 40, 40, 4, 4, 4] * 2
)
KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [4] * 3 + [5, 5, 5, 5, 0.2, 0.2, 0.2] * 2)


def onnx_worker(model, assets):
    sys.path.insert(0, str(assets / "experiments/runtime/onnxruntime"))
    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), opt, providers=["CPUExecutionProvider"])
    (inp,) = session.get_inputs()
    if inp.shape[-1] != 1432 or session.get_outputs()[0].shape[-1] != 29:
        raise ValueError("Unexpected policy I/O")
    plant = ort.InferenceSession(
        str(assets / ".pretrained_checkpoints/paid_official/policy_30000.onnx"),
        opt,
        providers=["CPUExecutionProvider"],
    )
    print(json.dumps(plant.get_modelmeta().custom_metadata_map), flush=True)
    for line in sys.stdin:
        x = np.asarray(json.loads(line), dtype=np.float32).reshape(1, 1432)
        y = session.run(None, {inp.name: x})[0][0]
        print(json.dumps(y.tolist(), allow_nan=False), flush=True)


@contextmanager
def runtime(model, assets, exe):
    cmd = [
        exe,
        "-u",
        str(Path(__file__).resolve()),
        "--model",
        str(model),
        "--assets",
        str(assets),
        "--worker",
    ]
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    ) as p:
        metadata = json.loads(p.stdout.readline())

        def step(x):
            p.stdin.write(json.dumps(x.tolist(), allow_nan=False) + "\n")
            p.stdin.flush()
            return np.asarray(json.loads(p.stdout.readline()))

        try:
            yield metadata, step
        finally:
            p.stdin.close()
            if p.wait(timeout=15):
                raise RuntimeError("TWIST2 inference failed")


def resample(reference, hz):
    t = reference["time_s"]
    clock = np.arange(round(float(t[-1]) * hz) + 1) / hz
    if hz == 50:
        return {
            k: v
            for k, v in reference.items()
            if k
            in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
        }
    result = {
        k: base.interpolate(reference[k], clock * 50)
        for k in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
    }
    quat = np.stack(
        [
            Slerp(t, base.rotation(q))(clock).as_quat()[:, [3, 0, 1, 2]]
            for q in reference["body_quat_w"].transpose(1, 0, 2)
        ],
        axis=1,
    )
    result["body_quat_w"] = quat
    return result


def mimic_input(ref, order):
    rot = base.rotation(ref["body_quat_w"][:, 0])
    v = rot.inv().apply(ref["body_lin_vel_w"][:, 0])
    w = rot.inv().apply(ref["body_ang_vel_w"][:, 0])
    return np.column_stack(
        [
            v[:, :2],
            ref["body_pos_w"][:, 0, 2],
            rot.as_euler("xyz")[:, :2],
            w[:, 2],
            ref["joint_pos"][:, order],
        ]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--assets", type=Path, default=base.DEFAULT_ASSETS)
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--hz", type=int, choices=(50, 100), default=50)
    p.add_argument("--duration", type=float, default=14.0)
    p.add_argument("--ball", nargs=2, type=float, default=[1.18723, -0.80613])
    p.add_argument("--direction", nargs=2, type=float, default=[0.9978, 0.0663])
    p.add_argument(
        "--feedback",
        type=float,
        default=0.0,
        help="XY position error (1/s), converted to tracker local-velocity input",
    )
    p.add_argument(
        "--yaw-feedback", type=float, default=0.0, help="Heading error gain (1/s)"
    )
    p.add_argument(
        "--lead",
        type=float,
        default=0.0,
        help="Seconds of reference preview to offset tracking delay",
    )
    p.add_argument(
        "--root-lead",
        type=float,
        default=0.0,
        help="Additional root-only preview for gait latency",
    )
    p.add_argument(
        "--arm-gap",
        action="store_true",
        help="Move upper arms outward near medial shoulder angles",
    )
    p.add_argument(
        "--placement",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        help="World XY support approach compensation",
    )
    p.add_argument(
        "--guard",
        type=float,
        default=0.0,
        help="Predictive motor braking horizon, seconds",
    )
    p.add_argument(
        "--brake-margin",
        type=float,
        default=0.04,
        help="Torque-producing target margin for predictive braking",
    )
    p.add_argument(
        "--swing-ankle",
        type=float,
        default=0.0,
        help="Forward-swing foot pitch change in radians",
    )
    p.add_argument(
        "--rest",
        action="store_true",
        help="Blend to tracker neutral only after the final human frame",
    )
    args = p.parse_args()
    if args.worker:
        onnx_worker(args.model, args.assets)
        return
    if not args.reference or not args.output or base.digest(args.model) != MODEL_HASH:
        p.error("Need matching frozen checkpoint, source reference and output")
    if (
        args.duration <= 0
        or args.duration > 14
        or not np.isclose(args.duration * args.hz, round(args.duration * args.hz))
    ):
        p.error("Duration must contain whole control steps")
    params = np.asarray(
        [
            *args.direction,
            *args.ball,
            args.feedback,
            args.yaw_feedback,
            args.lead,
            args.root_lead,
            args.guard,
            args.swing_ankle,
            args.brake_margin,
            *args.placement,
        ]
    )
    if (
        not np.isfinite(params).all()
        or np.linalg.norm(args.direction) < 1e-9
        or min(args.lead, args.root_lead, args.guard) < 0
    ):
        p.error(
            "Need finite controls, a nonzero direction and nonnegative preview/guard"
        )
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(base.PROTOCOL.read_text())
    schema = json.loads((args.assets / base.SCHEMA).read_text())
    for name, expected in cfg["asset_hashes"].items():
        if base.digest(args.assets / name) != expected:
            raise ValueError("Changed original plant asset " + name)
    src = np.load(args.reference, allow_pickle=False)
    if (
        src["joint_names"].tolist() != schema["joint_names"]
        or src["body_names"].tolist() != schema["body_names"]
        or not np.allclose(src["time_s"], np.arange(len(src["joint_pos"])) / 50)
    ):
        raise ValueError("Reference name or time contract mismatch")
    ref = resample(src, args.hz)
    # Store the exact actor/metric reference if a control-rate change is requested.
    if args.hz != 50:
        np.savez_compressed(out / "reference_control_rate.npz", **ref)
    cfg["clip"] = str(
        (
            out / "reference_control_rate.npz" if args.hz != 50 else args.reference
        ).resolve()
    )
    dir_xy = np.asarray(args.direction, dtype=float)
    dir_xy /= np.linalg.norm(dir_xy)
    cfg.update(
        duration_s=args.duration,
        control_dt_s=1 / args.hz,
        ball_initial_xyz_m=[*args.ball, 0.11],
        goal_forward_xy=dir_xy.tolist(),
        goal_xyz_m=[*(np.asarray(args.ball) + 2.8 * dir_xy), 0.11],
    )
    canonical = schema["joint_names"]
    into = np.array([canonical.index(n) for n in NAMES])
    back = np.argsort(into)
    mimic = mimic_input(ref, into)
    actor_observations = []
    raw_actions = []
    robot_state = []

    def plant(assets, config, meta, schema):
        s = base.make_plant(assets, config, meta, schema)
        robot_state.append((s[2], s[5], s[6]))
        drives, limits, gains = s[4], s[6], s[7]
        for j, idx, limit in zip(drives, back, limits):
            j.set_drive_properties(
                float(KP[idx]), float(KD[idx]), float(limit[3]), mode="force"
            )
        gains["default_joint_pos"] = DEFAULT[back]
        gains["action_scale"] = np.full(29, 0.5)
        return s

    history = deque([np.zeros(127) for _ in range(10)], maxlen=10)
    previous = np.zeros(29)
    with runtime(args.model, args.assets, cfg["python_for_onnx"]) as (meta, infer):

        def actor(frame, obs):
            nonlocal previous
            qrel = obs[67:96][into]
            dq = obs[96:125][into].copy()
            dq[[4, 5, 10, 11]] = 0.0
            gravity = obs[58:61]
            roll = np.arctan2(-gravity[1], -gravity[2])
            pitch = np.arcsin(np.clip(gravity[0], -1, 1))
            ahead = min(frame + round(args.lead * args.hz), len(mimic) - 1)
            reference = mimic[ahead].copy()
            walk = min(ahead + round(args.root_lead * args.hz), len(mimic) - 1)
            if args.arm_gap:
                reference[22] = max(reference[22], 0.21)
                reference[29] = min(reference[29], -0.21)
            ((robot, jorder, limits),) = robot_state
            actual = base.rotation(robot.root_pose.q)
            desired = base.rotation(ref["body_quat_w"][walk, 0])
            if args.feedback:
                target_velocity = (
                    ref["body_lin_vel_w"][walk, 0]
                    if len(raw_actions) < len(mimic)
                    else np.zeros(3)
                )
                error = ref["body_pos_w"][walk, 0] - robot.root_pose.p
                # The movement interval is inferred from human root speed, including early approach.
                effort = np.linalg.norm(ref["body_lin_vel_w"][walk, 0, :2])
                error[:2] += np.array(args.placement) * np.clip(effort / 0.4, 0, 1)
                correction = np.clip(args.feedback * error[:2], -1.2, 1.2)
                reference[:2] = actual.inv().apply(
                    target_velocity + np.r_[correction, 0]
                )[:2]
            if args.yaw_feedback:
                yaw_difference = desired.as_euler("xyz")[2] - actual.as_euler("xyz")[2]
                reference[5] += np.clip(
                    args.yaw_feedback
                    * np.arctan2(np.sin(yaw_difference), np.cos(yaw_difference)),
                    -2,
                    2,
                )
            if len(raw_actions) >= len(mimic):
                if not args.feedback:
                    reference[:2] = 0.0
                if not args.yaw_feedback:
                    reference[5] = 0.0
                if args.rest:
                    u = np.clip(
                        (len(raw_actions) - len(mimic)) / (args.hz * 0.65), 0.0, 1.0
                    )
                    w = u * u * (3.0 - 2 * u)
                    neutral = np.r_[0.0, 0.0, 0.793, 0.0, 0.0, 0.0, DEFAULT]
                    reference = (1 - w) * reference + w * neutral
            proprio = np.r_[obs[64:67] * 0.25, roll, pitch, qrel, dq * 0.05, previous]
            single = np.r_[reference, proprio]
            x = np.concatenate([single, np.asarray(history).ravel(), reference]).astype(
                "f4"
            )
            if x.shape != (1432,):
                raise ValueError("Invalid reference/proprio layout")
            history.append(single)
            previous = infer(x)
            raw_actions.append(previous.copy())
            actor_observations.append(x)
            target = DEFAULT + 0.5 * np.clip(previous, -10, 10)
            if args.swing_ankle:
                # Forward ankle speed relative to the human pelvis selects the active swing.
                foot = schema["body_names"].index("right_ankle_roll_link")
                rel_v = (
                    ref["body_lin_vel_w"][ahead, foot] - ref["body_lin_vel_w"][ahead, 0]
                )
                strength = np.clip((rel_v[:2] @ dir_xy - 1.0) / 2.5, 0.0, 1.0)
                target[10] += args.swing_ankle * strength
            if args.guard > 0:
                position = robot.qpos[jorder][into]
                velocity = robot.qvel[jorder][into]
                bounds = limits[into, :2]
                for j in (5, 11, 14):
                    horizon = args.guard
                    if args.brake_margin > 0.04:
                        target[j] = np.clip(
                            target[j],
                            bounds[j, 0] + args.brake_margin,
                            bounds[j, 1] - args.brake_margin,
                        )
                    forecast = position[j] + horizon * velocity[j]
                    if forecast > bounds[j, 1] - 0.025:
                        target[j] = min(
                            target[j],
                            bounds[j, 1]
                            - args.brake_margin
                            - max(0.0, horizon * velocity[j]),
                        )
                    elif forecast < bounds[j, 0] + 0.025:
                        target[j] = max(
                            target[j],
                            bounds[j, 0]
                            + args.brake_margin
                            + max(0.0, -horizon * velocity[j]),
                        )
            return (2 * (target - DEFAULT))[back]

        base.run(args.assets, out, cfg, schema, meta, actor, plant_factory=plant)
    actor_trace = out / "twist_actor.npz"
    np.savez_compressed(
        actor_trace,
        observations=np.asarray(actor_observations),
        actions=np.asarray(raw_actions),
    )
    result = out / "tracking_result.json"
    r = json.loads(result.read_text())
    r["code_sha256"].pop("v160_tracking_worker.py", None)
    r["measurement_scope"] = (
        "SAPIEN CPU PhysX, original G1 URDF; actual original effort and velocity bounds; TWIST2 PD gains, unguided free-root. Native load threshold 30 N per foot for final bilateral stance. No hardware or original Isaac GPU result."
    )
    r.update(
        controller="TWIST2 25k CPU ONNX",
        runtime="SAPIEN PhysxCpuSystem with TWIST2 CPU ONNX; PAiD metadata used only to construct original plant",
        revision=REVISION,
        checkpoint_sha256=base.digest(args.model),
        reference_sha256=base.digest(args.reference),
        actor_code_sha256=base.digest(Path(__file__)),
        policy_hz=args.hz,
        pd_stiffness_twist_order=KP.tolist(),
        pd_damping_twist_order=KD.tolist(),
        effort_limits="original PAiD G1 URDF",
        action_source="TWIST2 followed by optional declared joint-target corrections",
        twist_actor_trace_sha256=base.digest(actor_trace),
        ball_initial_xyz_m=cfg["ball_initial_xyz_m"],
        goal_xyz_m=cfg["goal_xyz_m"],
        goal_forward_xy=cfg["goal_forward_xy"],
        feedback_xy_gain_per_s=args.feedback,
        feedback_yaw_gain_per_s=args.yaw_feedback,
        preview_s=args.lead,
        neutral_terminal_command=args.rest,
        additional_root_preview_s=args.root_lead,
        arm_clearance=args.arm_gap,
        approach_offset_xy=args.placement,
        predictive_brake_horizon_s=args.guard,
        swing_ankle_pitch=args.swing_ankle,
        brake_margin_rad=args.brake_margin,
    )
    result.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
