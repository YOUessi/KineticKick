"""Physical G1 tracking with a support-aware correction to the GMT reference velocity."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import v163_gmt as g
from scipy.ndimage import uniform_filter1d
from v164_clear_legs import LegClearance

b = g.phys


def smooth_velocity(x):
    v = np.concatenate([x, x[-1:]], axis=0) * 50.0
    return uniform_filter1d(v.astype(float), size=19, axis=0, mode="constant", cval=0.0)


class Support:
    def __init__(self, ref, policy, cfg, signals):
        self.signals = signals
        self.cfg = cfg
        self.ref = ref
        self.net = policy
        self.root = b.rotation(ref["body_quat_w"][:, 0])
        self.v = smooth_velocity(np.diff(ref["body_pos_w"][:, 0], axis=0))
        self.w = smooth_velocity((self.root[1:] * self.root[:-1].inv()).as_rotvec())
        self.correct_v = np.zeros(3)

    def __call__(self, observation):
        row = observation.numpy().copy()
        window = row[0, :600].reshape(20, 30)
        ctx = self.ctx
        cfg = self.cfg
        frame = ctx["frame"]
        clock = round(ctx["t"] / 0.02)
        n = len(self.v)
        physical = clock + g.FUTURE
        idx = np.minimum(n - 1, physical)
        # Preserve the terminal blend beyond the motion.
        ref_mask = physical < n
        rot = self.root[idx]
        if cfg.get("official_vel"):
            window[ref_mask, 3:6] = rot[ref_mask].inv().apply(self.v[idx[ref_mask]])
            window[ref_mask, 6] = rot[ref_mask].inv().apply(self.w[idx[ref_mask]])[:, 2]
        if cfg.get("inverse_stride", 0.0) and ref_mask.any():
            extra = self.v[idx[ref_mask]].copy()
            extra[:, 2] = 0
            window[ref_mask, 3:6] += (
                rot[ref_mask].inv().apply(cfg["inverse_stride"] * extra)
            )
        measured = b.rotation(ctx["root_quat"])
        target_pos = self.ref["body_pos_w"][frame, 0]
        error = target_pos - ctx["root_pos"]
        error[2] = 0.0
        dv = self.v[frame] - ctx["root_velocity"][:3]
        dv[2] = 0.0
        loaded = np.asarray(ctx["foot_load"]) > 35.0
        strength = (
            1.0
            if loaded.sum() == 2
            else cfg.get("single", 1.0)
            if loaded.sum() == 1
            else 0.0
        )
        cue = cfg.get("kp", 0.0) * error + cfg.get("kd", 0.0) * dv
        cap = cfg.get("cap", 0.65)
        cue *= min(1.0, cap / (np.linalg.norm(cue) + 1e-9)) * strength
        alpha = 1.0 - np.exp(-0.02 / cfg.get("tau", 0.1))
        self.correct_v += (cue - self.correct_v) * alpha
        yaw = self.root[frame].as_euler("xyz")[2] - measured.as_euler("xyz")[2]
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        if clock < n:
            window[ref_mask, 3:6] += measured.inv().apply(self.correct_v)
            window[ref_mask, 6] += np.clip(cfg.get("heading", 0.0) * yaw, -1.0, 1.0)
        action = self.net(torch.from_numpy(row))
        self.signals["network_input"].append(row[0].copy())
        self.signals["network_action"].append(action.numpy()[0].copy())
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=b.DEFAULT_ASSETS)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=b.REPO / "experiments/v163_gmt/assets/pretrained.pt",
    )
    parser.add_argument("--initial-state-config", type=Path)
    a = parser.parse_args()
    c = json.loads(a.config.read_text())
    cfg = json.loads(b.PROTOCOL.read_text())
    assets = a.assets.resolve()
    for file, digest in cfg["asset_hashes"].items():
        if b.digest(assets / file) != digest:
            raise ValueError("Changed plant " + file)
    if b.digest(a.checkpoint) != g.CHECKPOINT:
        raise ValueError("Unknown GMT model")
    schema = json.loads((assets / b.SCHEMA).read_text())
    with np.load(a.reference, allow_pickle=False) as r:
        if (
            r["joint_names"].tolist() != schema["joint_names"]
            or r["body_names"].tolist() != schema["body_names"]
            or r["fps"].tolist() != [50]
        ):
            raise ValueError("Invalid reference")
    cfg["clip"] = str(a.reference.resolve())
    cfg["duration_s"] = 14.0
    torch.set_num_threads(1)
    p = torch.jit.load(str(a.checkpoint), map_location="cpu").eval()
    a.output.mkdir(parents=True, exist_ok=False)
    clearance = (
        LegClearance(schema, c["leg_clearance"], assets / b.URDF)
        if c.get("leg_clearance")
        else None
    )
    signals = {
        k: [] for k in ("network_input", "network_action", "drive_targets_200hz_rad")
    }
    with b.cpu_policy(assets, cfg) as (meta, _):
        b.save_json(
            a.output / "trial.json",
            dict(
                controller="GMT support",
                checkpoint_sha256=g.CHECKPOINT,
                code_sha256=b.digest(Path(__file__)),
                leg_clearance_sha256=(
                    b.digest(Path(__file__).with_name("v164_clear_legs.py"))
                    if clearance
                    else None
                ),
                config=c,
                config_sha256=b.digest(a.config),
                reference_sha256=b.digest(a.reference),
                gmt_runner_sha256=b.digest(Path(g.__file__)),
                physical_runner_sha256=b.digest(Path(b.__file__)),
                **{
                    k: cfg[k]
                    for k in (
                        "clip",
                        "ball_initial_xyz_m",
                        "goal_xyz_m",
                        "goal_forward_xy",
                    )
                },
            ),
        )

        def factory(ref, sc):
            net = Support(ref, p, c, signals)
            actor = g.make_control(
                ref,
                sc,
                meta,
                net,
                SimpleNamespace(drive="native", adapter={}, settle=True, feedback=0.0),
            )

            def step(ctx, unused):
                net.ctx = ctx
                target = actor(ctx, unused)
                return clearance(ctx, target) if clearance else target

            return ref, step

        def plant(assets, cfg, meta, schema):
            scene, system, robot, ball, drives, order, limits, gains, links, ids = (
                b.make_plant(assets, cfg, meta, schema)
            )
            for j, n in enumerate(g.NAMES):
                i = schema["joint_names"].index(n + "_joint")
                drives[i].set_drive_properties(
                    g.KP[j], g.KD[j], limits[i, 3], mode="force"
                )
            return (
                g.guarded(
                    scene,
                    robot,
                    drives,
                    order,
                    limits,
                    schema["joint_names"],
                    c,
                    signals,
                ),
                system,
                robot,
                ball,
                drives,
                order,
                limits,
                gains,
                links,
                ids,
            )

        initializer = None
        if a.initial_state_config:
            from v164_initial_state import project_start

            settings = json.loads(a.initial_state_config.read_text())

            def initializer(q, dq, limits):
                safe, velocity, changed = project_start(
                    q, dq, limits, schema["joint_names"], settings, cfg["physics_dt_s"]
                )
                b.save_json(
                    a.output / "initial_state.json",
                    {
                        "settings": settings,
                        "config_sha256": b.digest(a.initial_state_config),
                        "initializer_sha256": b.digest(Path(__file__).with_name("v164_initial_state.py")),
                        "changed": changed,
                        "scope": "Joint initialization only. Source observations and all root states retain the original initialization.",
                    },
                )
                return safe, velocity

        b.run(
            assets,
            a.output,
            cfg,
            schema,
            meta,
            lambda f, o: np.zeros(29),
            controller_factory=factory,
            plant_factory=plant,
            joint_initializer=initializer,
        )
    file = a.output / "gmt_commands.npz"
    np.savez_compressed(file, **signals)
    b.save_json(
        a.output / "gmt_recording.json",
        {
            "sha256": b.digest(file),
            "joint_names": schema["joint_names"],
            "network_axes": g.NAMES,
            "leg_clearance_corrected_ticks": clearance.corrections if clearance else 0,
            "input_observations": "GMT velocity and heading plan with source time preserved; legacy PAiD trace columns are placeholders.",
        },
    )


if __name__ == "__main__":
    main()
