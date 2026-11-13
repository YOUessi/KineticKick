#!/usr/bin/env python3
"""v146 read-only instrumentation of the original v080 G1 runner.

The caller alone steps physics and selects actions. Nothing here applies a USD
schema, writes a robot/ball tensor, changes collision filters or advances time.
Callbacks are copied during the native scene.update() bracket, but this is NOT
a calibrated exact PhysX backend tick. CPU fakes have zero physical provenance.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import v134_native_contact_vector_gate as native
from v142_native_pair_ledger import safe_write_once, verify_original_state, packet_review
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA = "7847215203ef936332dd98b5a597a72d2bf1fb72"
CONFIG = "configs/v146_frozen_native_original_runner_bridge.json"
PHYSICS_FIELDS = (
    "step", "q", "dq", "base_pose", "base_twist", "com_world",
    "left_foot_net_force", "right_foot_net_force", "nonfoot_net_force",
    "ball_pose", "ball_twist", "motor_torque"
)


def frozen():
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", FROZEN_SHA + ":" + CONFIG])
    if raw != (ROOT / CONFIG).read_bytes():
        raise ValueError("V146_FROZEN_PROTOCOL_DRIFT")
    cfg = json.loads(raw)
    if (cfg["version"] != "v0.1.146" or cfg["physics_dt_s"] != .005
            or cfg["original_dof"] != 29 or cfg["full_horizon_steps"] != 2800
            or not cfg["must_not_apply_contact_report_schema"]
            or not cfg["must_not_change_original_control_or_termination"]
            or cfg["counterfactual_cpu_fixture_cannot_certify_runtime"] is not True):
        raise ValueError("V146_FROZEN_ASSUMPTIONS_DRIFT")
    return cfg


def gpu_preflight(*, gpu_table=None, disk_free_bytes=None):
    """Enforce user safety requirements before importing/starting IsaacLab.

    A successful check is not a lock on foreign processes; real live runs must
    additionally acquire and release the original shared GPU mutex.
    """
    cfg = frozen()
    if disk_free_bytes is None:
        disk_free_bytes = shutil.disk_usage(ROOT).free
    if disk_free_bytes < cfg["min_available_disk_gib"] * 1024 ** 3:
        raise RuntimeError("V146_DISK_BELOW_20GIB_NO_PHYSICS")
    if gpu_table is None:
        p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        if p.returncode:
            raise RuntimeError("V146_GPU_STATUS_UNKNOWN_NO_PHYSICS")
        gpu_table = p.stdout
    if gpu_table.strip():
        raise RuntimeError("V146_FOREIGN_GPU_CONTEXT_NO_PHYSICS")
    return {"gpu_is_idle_at_preflight": True, "disk_gib": disk_free_bytes / 1024**3,
            "gpu_preflight_is_not_exclusive_mutex": True}


def json_sha(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf8")).hexdigest()


def _number_vector(value, size, label):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError("G1_VECTOR_LENGTH_" + label)
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(float(x)) for x in value):
        raise ValueError("G1_VECTOR_NONFINITE_" + label)
    return [float(x) for x in value]


class OriginalG1StateSampler:
    """Read existing scene buffers. Torques are simulator estimates, NOT hardware."""
    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError("ORIGINAL_G1_REQUIRES_SINGLE_ENV")
        self.robot = env.scene["robot"]
        self.ball = env.scene["soccer_ball"]
        self.sensor = env.scene["contact_forces"]
        if tuple(self.robot.joint_names) != G1_JOINT_ORDER:
            raise ValueError("ORIGINAL_G1_JOINT_ORDER_MISMATCH")
        if self.robot.data.joint_pos.shape[-1] != 29:
            raise ValueError("ORIGINAL_G1_NOT_29_DOF")
        self.left = self.sensor.body_names.index("left_ankle_roll_link")
        self.right = self.sensor.body_names.index("right_ankle_roll_link")
        self.nonfoot = [(i, n) for i, n in enumerate(self.sensor.body_names)
                        if i not in (self.left, self.right)]
        if not self.nonfoot:
            raise ValueError("MISSING_ORIGINAL_NONFOOT_CONTACT_BODIES")
        masses = self.robot.root_physx_view.get_masses()[0]
        if (len(masses) != self.robot.data.body_com_pos_w.shape[1]
                or float(masses.sum()) <= 0):
            raise ValueError("G1_MASS_TO_BODY_MISMATCH")
        self.weights = masses / masses.sum()
        self.nonfoot_peak_names = []

    def sample(self, step):
        r = self.robot.data
        f = self.sensor.data.net_forces_w[0]
        idx, name = max(self.nonfoot, key=lambda pair: float(f[pair[0]].norm().item()))
        self.nonfoot_peak_names.append(name)
        com = (r.body_com_pos_w[0] * self.weights[:, None]).sum(dim=0)
        v = _number_vector
        row = {
            "step": step,
            "q": v(r.joint_pos[0], 29, "Q"),
            "dq": v(r.joint_vel[0], 29, "DQ"),
            "motor_torque": v(r.applied_torque[0], 29, "SIM_TORQUE"),
            "base_pose": {"position": v(r.root_pos_w[0], 3, "ROOT"),
                          "quaternion": v(r.root_quat_w[0], 4, "ROOT_QUAT")},
            "base_twist": {"linear": v(r.root_lin_vel_w[0], 3, "ROOT_LIN"),
                           "angular": v(r.root_ang_vel_w[0], 3, "ROOT_ANG")},
            "com_world": v(com, 3, "COM"),
            "left_foot_net_force": v(f[self.left], 3, "LEFT_NET"),
            "right_foot_net_force": v(f[self.right], 3, "RIGHT_NET"),
            "nonfoot_net_force": v(f[idx], 3, "NONFOOT_NET"),
            "ball_pose": v(self.ball.data.root_pos_w[0], 3, "BALL"),
            "ball_twist": v(self.ball.data.root_lin_vel_w[0], 3, "BALL_SPEED"),
        }
        if set(row) != set(PHYSICS_FIELDS):
            raise ValueError("INCOMPLETE_ORIGINAL_G1_FIELDS")
        verify_original_state(row)
        return row


def read_only_live_stage():
    """No USD mutations. Returns current stage ID plus existing collider owner map."""
    import omni.usd
    from v133_runtime_stage_contract import inspect_actual_usd_stage
    from v134_native_contact_vector_gate import native_path_decoder
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    stage_id = ctx.get_stage_id()
    if (stage is None or isinstance(stage_id, bool)
            or not isinstance(stage_id, int) or stage_id <= 0):
        raise ValueError("REAL_USD_STAGE_UNAVAILABLE")
    review = inspect_actual_usd_stage(stage)
    owner_map = review["collider_to_nearest_rigidbody_owner"]
    if not owner_map:
        raise ValueError("REAL_USD_STAGE_COLLIDER_MAP_EMPTY")
    return stage_id, owner_map, native_path_decoder, review


class OriginalG1ReadOnlyCapture:
    def __init__(self, *, mode, sampler, stage_reader=None,
                 native_subscribe=None, max_packets=None):
        cfg = frozen()
        if mode not in cfg["modes"]:
            raise ValueError("ORIGINAL_G1_OBSERVER_MODE_INVALID")
        self.mode = mode
        self.sampler = sampler
        self.stage_reader = stage_reader or read_only_live_stage
        self.native_subscribe = native_subscribe
        self.max_packets = max_packets or cfg["per_episode_max_native_callback_packets"]
        self.rows, self.packets, self.pending, self.errors = [], [], [], []
        self.pending_step = None
        self.stage_id, self.owner_map, self.stage_review = None, {}, {}
        self.callback_count = 0
        self.subscription = None
        self.failed = False

    def attach(self):
        if self.mode == "off":
            return None
        if self.subscription is not None:
            raise ValueError("V146_DUPLICATE_NATIVE_SUBSCRIPTION")
        try:
            self.stage_id, self.owner_map, decode, self.stage_review = self.stage_reader()
            subscribe = self.native_subscribe
            if subscribe is None:
                from omni.physx import get_physx_simulation_interface
                subscribe = get_physx_simulation_interface().subscribe_contact_report_events
            self._decode = decode
            self.subscription = subscribe(self.on_callback)
            if self.subscription is None:
                raise ValueError("NO_EXISTING_NATIVE_CONTACT_SUBSCRIPTION")
        except Exception as exc:
            self.errors.append("READONLY_STAGE_OR_SUBSCRIBE_UNAVAILABLE:" +
                               type(exc).__name__ + ":" + str(exc)[:180])
            self.failed = True
        return self.subscription

    def before_native_update(self, dt):
        if (self.pending_step is not None or self.failed
                or isinstance(dt, bool) or not isinstance(dt, (int, float))
                or not math.isfinite(float(dt)) or abs(dt - .005) > 1e-7):
            self.errors.append("PHYSICS_STEP_BRACKET_INVALID")
            self.failed = True
            return
        self.pending_step = len(self.rows) + 1

    def on_callback(self, headers, points):
        self.callback_count += 1
        if self.mode != "on" or self.pending_step is None or self.failed:
            self.errors.append("CALLBACK_OUTSIDE_UNCALIBRATED_PHYSICS_INTERVAL")
            return
        if len(self.packets) + len(self.pending) >= self.max_packets:
            self.errors.append("NATIVE_CALLBACK_MEMORY_CAP_REACHED")
            self.failed = True
            return
        try:
            packet = native.copy_contact_vector_frame(
                headers, points, decode_path=self._decode,
                step_hint=self.pending_step, expected_stage_id=self.stage_id)
            packet["clock_basis"] = "SCENE_UPDATE_BRACKET_NOT_NATIVE_STEP_ID"
            self.pending.append(packet)
        except Exception as exc:
            self.errors.append("UNTRUSTED_NATIVE_CALLBACK:" +
                               type(exc).__name__ + ":" + str(exc)[:180])
            self.failed = True

    def after_native_update(self, dt):
        if self.failed:
            self.pending_step = None
            return
        try:
            if self.pending_step != len(self.rows) + 1 or abs(dt - .005) > 1e-7:
                raise ValueError("SCENE_UPDATE_BRACKET_NOT_COMPLETE")
            row = self.sampler.sample(self.pending_step)
            verify_original_state(row)
            self.rows.append(copy.deepcopy(row))
            self.packets.extend(self.pending)
            self.pending.clear()
            self.pending_step = None
        except Exception as exc:
            self.errors.append("ORIGINAL_G1_POST_UPDATE_READ_FAILED:" +
                               type(exc).__name__ + ":" + str(exc)[:180])
            self.failed = True
            self.pending_step = None

    def review(self):
        candidates = []
        rejected = []
        for pkt in self.packets:
            step = pkt["step_hint"]
            if step <= 0 or step > len(self.rows):
                rejected.append("ORPHAN_NATIVE_CALLBACK")
                continue
            try:
                result = packet_review(pkt, self.rows[step - 1], self.owner_map,
                                       expected_stage_id=self.stage_id)
                candidates.extend(result["left_wrist_hip_colliding_pair_candidates"])
            except Exception as exc:
                rejected.append(type(exc).__name__ + ":" + str(exc)[:130])
        return {
            "schema": "v146_g1_unattested_bracket_and_physical_rows",
            "mode": self.mode,
            "original_29dof_completed_5ms_rows": len(self.rows),
            "copied_native_callback_packets": len(self.packets),
            "callback_invocations_not_step_count": self.callback_count,
            "copied_manifold_points": sum(p["total_points"] for p in self.packets),
            "wrist_hip_interval_candidates_not_truth": candidates,
            "rejected_packet_reasons": rejected,
            "errors": self.errors,
            "readout_incomplete": bool(self.errors or rejected or self.failed or self.pending or self.pending_step),
            "original_full_state_checksum": json_sha(self.rows),
            "collision_stage_readback": self.stage_review,
            "left_wrist_hip_native_physx_truth": "UNKNOWN",
            "same_native_physics_tick_calibrated": False,
            "off_on_physical_equality_certified": False,
            "cpu_fixtures_cannot_certify_physx": True,
            "root_tilt_is_not_fall_proof": True,
            "net_foot_force_is_not_isolated_ground_reaction_force": True,
            "motor_torque_is_simulator_estimate_not_hardware_measurement": True,
            "strict_14s_recovery_if_early_terminated": "UNKNOWN",
            "new_physx_episode_success": None,
            "motor_execution_authorized": False,
        }

    def persist_once(self, output):
        return safe_write_once(Path(output), {
            "summary": self.review(), "collider_to_rigid_owner": self.owner_map,
            "raw_callback_copies": self.packets,
            "original29_5ms_state_rows": self.rows,
            "named_max_nonfoot_net_body": getattr(self.sampler, "nonfoot_peak_names", []),
        })
