#!/usr/bin/env python3
"""Build a CPU-only, run-only PAiD training command from local source motion.

Preserves playback speed and all joints except a bounded smooth left hip pitch
floor that clears the pelvis/thigh capsule pair. Applies constant world yaw
and minimum constant height offset for runtime-capsule foot clearance.
No simulator, runtime robot-state edits, arm replacement, or braking tail.
This is a geometrically screened command, not evidence of physical running.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from scipy.special import expit
from scipy.spatial.transform import Rotation, RotationSpline

from running_reference_geometry import ReferenceGeometry, segment_distance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clearance-m', type=float, default=.001)
    parser.add_argument('--hip-capsule-clearance-m', type=float, default=.001)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not np.isfinite(args.clearance_m) or args.clearance_m < 0.:
        parser.error('--clearance-m must be finite and nonnegative')
    if not np.isfinite(args.hip_capsule_clearance_m) or args.hip_capsule_clearance_m < 0.:
        parser.error('--hip-capsule-clearance-m must be finite and nonnegative')
    if output.suffix != '.npz':
        parser.error('--output must have .npz suffix')
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Do not overwrite reference artifacts: ' + str(output))
    sys.path.insert(0, str(root / 'tools'))
    from g1_retarget_kinematics import JOINTS

    geometry = ReferenceGeometry(root)
    joint_names, body_names = geometry.joint_names, geometry.body_names
    if set(joint_names) != set(JOINTS):
        raise ValueError('Source and runtime joint names mismatch')
    source = root / 'data/human_motion/straight_reference29/run_straight.npz'
    with np.load(source) as src:
        frames = src['frames'].astype(float)
        source_fps = float(np.asarray(src['fps']).item())
    fps, duration = 50, 2.2
    if (frames.ndim != 2 or frames.shape[1] != 36 or len(frames) < 3
            or not np.isfinite(frames).all() or not np.isfinite(source_fps) or source_fps <= 0.):
        raise ValueError('Invalid 29DoF source motion')
    times = np.arange(round(duration * fps) + 1) / fps
    source_times = np.arange(len(frames)) / source_fps
    if source_times[-1] < duration:
        raise ValueError('Source must cover first 2.2 seconds')
    raw_q = frames[:, 7:][:, [JOINTS.index(name) for name in joint_names]]
    q_interp = CubicHermiteSpline(source_times, raw_q, np.gradient(raw_q, 1 / source_fps, axis=0))
    p_interp = CubicHermiteSpline(source_times, frames[:, :3], np.gradient(frames[:, :3], 1 / source_fps, axis=0))
    r_interp = RotationSpline(source_times, Rotation.from_quat(frames[:, 3:7]))
    yaw_radians = -np.pi / 2
    yaw = Rotation.from_euler('z', yaw_radians)
    q, dq = q_interp(times), q_interp(times, 1)
    positions = yaw.apply(p_interp(times))
    quaternions = (yaw * r_interp(times)).as_quat()[:, [3, 0, 1, 2]]
    original_q = q.copy()
    hip_index = joint_names.index('left_hip_pitch_joint')
    def hip_gap(pitch):
        pose = original_q[0].copy()
        pose[hip_index] = pitch
        primitives = geometry.primitives(pose, positions[0], quaternions[0])
        a = next(g for g in primitives if g['name'] == 'pelvis')
        b = next(g for g in primitives if g['name'] == 'left_hip_roll_link')
        return segment_distance(a['start'], a['end'], b['start'], b['end']) - a['radius'] - b['radius']
    # The capsule's proximal endpoint lies on the hip-roll axis. Roll/yaw cannot
    # separate it from the pelvis sphere; a bounded pitch-floor correction can.
    bracket = (-1.8, -.5)
    residuals = [hip_gap(v) - args.hip_capsule_clearance_m for v in bracket]
    if residuals[0] > 0. or residuals[1] < 0.:
        raise ValueError(f'Hip clearance {args.hip_capsule_clearance_m} m cannot be solved within pitch bracket {bracket}; residuals {residuals}')
    pitch_floor = brentq(lambda v: hip_gap(v) - args.hip_capsule_clearance_m, *bracket)
    smoothing_rad = .01
    phase = (q[:, hip_index] - pitch_floor) / smoothing_rad
    q[:, hip_index] = pitch_floor + smoothing_rad * np.logaddexp(0., phase)
    dq[:, hip_index] *= expit(phase)
    pose_correction = q[:, hip_index] - original_q[:, hip_index]
    max_hip_correction_rad = .25
    if pose_correction.max() > max_hip_correction_rad:
        raise ValueError('Hip correction exceeds explicit .25 rad bound')
    before, foot_heights = geometry.check(q, positions, quaternions)
    height_offset = max(0., args.clearance_m - float(foot_heights.min()))
    positions[:, 2] += height_offset

    body_positions, body_quaternions = [], []
    for frame in range(len(times)):
        poses = geometry.body_poses(q[frame], positions[frame], quaternions[frame])
        if set(poses) != set(body_names):
            raise ValueError('FK body names mismatch')
        body_positions.append([poses[name][0] for name in body_names])
        body_quaternions.append([poses[name][1].as_quat()[[3, 0, 1, 2]] for name in body_names])
    body_positions = np.asarray(body_positions)
    body_quaternions = np.asarray(body_quaternions)
    for frame in range(1, len(times)):
        flip = np.sum(body_quaternions[frame - 1] * body_quaternions[frame], axis=-1) < 0.
        body_quaternions[frame, flip] *= -1
    linear_velocities = np.gradient(body_positions, 1 / fps, axis=0, edge_order=2)
    angular_velocities = np.zeros_like(linear_velocities)
    for body in range(len(body_names)):
        rotations = Rotation.from_quat(body_quaternions[:, body, [1, 2, 3, 0]])
        angular_velocities[1:-1, body] = (rotations[2:] * rotations[:-2].inv()).as_rotvec() * (fps / 2)
        angular_velocities[0, body] = (rotations[1] * rotations[0].inv()).as_rotvec() * fps
        angular_velocities[-1, body] = (rotations[-1] * rotations[-2].inv()).as_rotvec() * fps
    limits = {j.attrib['name']: j.find('limit') for j in geometry.xml.findall('joint')
              if j.find('limit') is not None}
    lower = np.array([float(limits[name].attrib['lower']) for name in joint_names])
    upper = np.array([float(limits[name].attrib['upper']) for name in joint_names])
    violations = np.maximum(lower - q, q - upper).max(axis=0).clip(0.)
    if violations.max() > 1e-7:
        raise ValueError('Source interpolation exceeds hard joint limits')
    arrays = dict(fps=np.array([fps], np.int64), joint_pos=q.astype(np.float32),
                  joint_vel=dq.astype(np.float32), body_pos_w=body_positions.astype(np.float32),
                  body_quat_w=body_quaternions.astype(np.float32),
                  body_lin_vel_w=linear_velocities.astype(np.float32),
                  body_ang_vel_w=angular_velocities.astype(np.float32), kick_leg=np.array('right'))
    if not all(np.isfinite(value).all() for name, value in arrays.items() if name != 'kick_leg'):
        raise ValueError('Nonfinite reference arrays')
    after, _ = geometry.check(arrays['joint_pos'], arrays['body_pos_w'][:, 0], arrays['body_quat_w'][:, 0])
    if after['min_nonadjacent_capsule_gap_m'] < 0.:
        raise ValueError('Serialized reference has nonadjacent capsule overlap: ' + str(after['closest_nonadjacent_pair']))
    if after['min_foot_z'] < args.clearance_m - 1e-6:
        raise ValueError('Serialized reference loses requested foot clearance')
    up_z = Rotation.from_quat(body_quaternions.reshape(-1, 4)[:, [1, 2, 3, 0]]).apply([0., 0., 1.])[:, 2]
    tilt = np.rad2deg(np.arccos(np.clip(up_z, -1., 1.))).reshape(len(times), len(body_names))
    displacement = positions[-1] - positions[0]
    initial_speed = np.linalg.norm(linear_velocities[0, 0, :2])
    goal_speed = -displacement[1] / duration
    source_paths = [source, geometry.urdf_path, geometry.schema_path, Path(__file__),
                    Path(__file__).with_name('running_reference_geometry.py'),
                    root / 'tools/audit_paid_urdf_reference.py', root / 'tools/g1_retarget_kinematics.py']
    metadata = dict(
        purpose='Geometrically screened 29DoF tracking command; not a trained physical running result',
        frames=len(times), duration_s=duration, fps=fps, source_fps=source_fps,
        hip_pitch_correction=dict(joint='left_hip_pitch_joint', floor_rad=float(pitch_floor), smoothing_rad=smoothing_rad, target_capsule_clearance_m=args.hip_capsule_clearance_m, max_allowed_correction_rad=max_hip_correction_rad, initial_correction_rad=float(pose_correction[0]), max_correction_rad=float(pose_correction.max()), frames_above_1mrad=int((pose_correction>.001).sum())),
        modifications=[
            'Retain only source first 2.2 seconds, resampled from 30 Hz to 50 Hz; no braking tail',
            'Constant world yaw -pi/2 makes travel approximately world -Y',
            f'Constant world Z offset {height_offset:.12f} m to meet {args.clearance_m} m foot clearance',
            'Only left hip pitch receives a smooth lower-floor correction to separate pelvis sphere and proximal thigh capsule; other joints and playback timing unchanged'],
        initialization_caveat=(
            f'Reference starts at {initial_speed:.3f} m/s horizontal speed in a running pose '
            f'(mean forward speed {goal_speed:.3f} m/s). Phase-sampled resets do not establish standing-to-running. '
            'Endpoint remains moving: terminate/restart; do not freeze it or loop positions without continuous phase/translation handling.'),
        contact_feasibility_caveat=(
            'Exact imported capsule/sphere ground support and nonadjacent separation are kinematic checks only. '
            'Directly adjacent primitives are excluded and require the original simulator collision filtering. '
            'Capsule segment lengths are the original URDF cylinder lengths, matching verified converted USD. Contact timing, forces, and dynamic balance are unvalidated.'),
        joint_names=joint_names, body_names=body_names, root_quaternion_order='wxyz', angular_velocity_frame='world',
        world_yaw_rad=yaw_radians, constant_height_offset_m=height_offset,
        geometry_before=before, geometry_after_float32=after,
        root_displacement_m=displacement.tolist(), mean_root_velocity_mps=(displacement / duration).tolist(),
        nominal_goal_forward=[0, -1], nominal_goal_speed_mps=float(goal_speed),
        initial_root_velocity_mps=linear_velocities[0, 0].tolist(),
        initial_pelvis_tilt_deg=float(tilt[0, body_names.index('pelvis')]),
        initial_torso_tilt_deg=float(tilt[0, body_names.index('torso_link')]),
        torso_tilt_min_max_deg=[float(tilt[:, body_names.index('torso_link')].min()),
                                float(tilt[:, body_names.index('torso_link')].max())],
        max_joint_speed_rad_s=float(abs(dq).max()), hard_joint_limit_violations_rad={},
        sources={str(path): sha(path) for path in source_paths})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    metadata['output'], metadata['output_sha256'] = str(output), sha(output)
    with output.with_suffix('.json').open('x') as stream:
        stream.write(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
