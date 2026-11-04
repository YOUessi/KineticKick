#!/usr/bin/env python3
"""CPU-only LAFAN running reference conversion; no simulator/state-write calls.

The generated clip is a motion-tracking command, never a physical running claim.
It starts from an already moving reference pose. Initializing a physical trial
from this pose therefore does not establish standing-to-running capability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, RotationSpline


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-duration', type=float, default=2.2)
    p.add_argument('--tail-duration', type=float, default=1.5)
    p.add_argument('--yaw-rad', type=float, default=0.)
    a = p.parse_args()
    root = a.root.resolve()
    sys.path.insert(0, str(root / 'tools'))
    from g1_retarget_kinematics import JOINTS
    from audit_paid_urdf_reference import load_urdf, fk_at_frame
    source = root / 'data/human_motion/straight_reference29/run_straight.npz'
    standing = root / 'third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-009_right.npz'
    urdf = root / 'configs/paid_original/main.urdf'
    schema = root / 'reports/paid_physics_parity_clean_paid.json'
    s = json.loads(schema.read_text())
    joint_names, body_names = s['joint_names'], s['body_names']
    if len(joint_names) != 29 or len(body_names) != 30 or set(joint_names) != set(JOINTS):
        raise ValueError('Verified PAiD runtime schema mismatch')
    tree, urdf_joints = load_urdf(urdf)
    if set(urdf_joints) != set(JOINTS):
        raise ValueError('Reference and PAiD joint names mismatch')
    src, stand = np.load(source), np.load(standing)
    f = src['frames'].astype(np.float64)
    source_fps = int(src['fps'])
    t_src = np.arange(len(f)) / source_fps
    fps = 50
    if (a.run_duration <= 0 or a.run_duration > t_src[-1] or a.tail_duration <= 0
            or not np.allclose(np.array([a.run_duration, a.tail_duration])*fps,
                               np.rint(np.array([a.run_duration, a.tail_duration])*fps))):
        raise ValueError('Durations must be positive 50 Hz frame multiples within source')
    times = np.arange(round((a.run_duration+a.tail_duration)*fps)+1)/fps
    run_mask = times <= a.run_duration + 1e-10
    run_t, tail_t = times[run_mask], times[~run_mask]-a.run_duration
    order = [JOINTS.index(n) for n in joint_names]
    source_q = f[:, 7:][:, order]
    q_interp = CubicHermiteSpline(t_src, source_q, np.gradient(source_q, 1/source_fps, axis=0))
    p_interp = CubicHermiteSpline(t_src, f[:, :3], np.gradient(f[:, :3], 1/source_fps, axis=0))
    r_interp = RotationSpline(t_src, Rotation.from_quat(f[:, 3:7]))
    end_q, end_dq = q_interp(a.run_duration), q_interp(a.run_duration, 1)
    end_p, end_v = p_interp(a.run_duration), p_interp(a.run_duration, 1)
    end_r = r_interp(a.run_duration)
    target_q = stand['joint_pos'][0].astype(float)
    # Keep final heading; stand reference supplies the unchanged PAiD resting pose.
    stand_r = Rotation.from_quat(stand['body_quat_w'][0, 0, [1, 2, 3, 0]])
    final_yaw = end_r.as_euler('xyz')[2]
    stand_yaw = stand_r.as_euler('xyz')[2]
    target_r = Rotation.from_euler('z', final_yaw-stand_yaw)*stand_r
    target_p = end_p.copy()
    target_p[:2] += .5*a.tail_duration*end_v[:2]
    target_p[2] = float(stand['body_pos_w'][0, 0, 2])
    # Brake the running joint tangent over 120 ms before moving to standing.
    # A single long Hermite span overshoots knee limits when dq points away
    # from the final standing pose. This C1 two-stage tail has no clipping.
    brake_duration = min(.12, a.tail_duration/3)
    brake_q = end_q + .5*brake_duration*end_dq
    q_tail = CubicHermiteSpline([0., brake_duration, a.tail_duration],
                              np.stack([end_q, brake_q, target_q]),
                              np.stack([end_dq, np.zeros(29), np.zeros(29)]))
    p_tail = CubicHermiteSpline([0., a.tail_duration], np.stack([end_p, target_p]),
                              np.stack([end_v, np.zeros(3)]))
    eps = 1e-5
    # World angular velocity at the seam. Exp(v(t))*R0 preserves this tangent.
    omega = (r_interp(a.run_duration+eps)*r_interp(a.run_duration-eps).inv()).as_rotvec()/(2*eps)
    final_delta = (target_r*end_r.inv()).as_rotvec()
    rv_tail = CubicHermiteSpline([0., a.tail_duration], np.stack([np.zeros(3), final_delta]),
                               np.stack([omega, np.zeros(3)]))
    q = np.concatenate([q_interp(run_t), q_tail(tail_t)])
    dq = np.concatenate([q_interp(run_t, 1), q_tail(tail_t, 1)])
    pos = np.concatenate([p_interp(run_t), p_tail(tail_t)])
    rotations = Rotation.concatenate([r_interp(run_t), Rotation.from_rotvec(rv_tail(tail_t))*end_r])
    # Optional constant world yaw only; never alter joint angles or playback time.
    yaw = Rotation.from_euler('z', a.yaw_rad)
    pos = yaw.apply(pos)
    rotations = yaw*rotations
    quat_xyzw = rotations.as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    all_pos, all_quat = [], []
    for i in range(len(times)):
        fk = {name: (xyz, rot) for name, xyz, rot in
              fk_at_frame(tree, dict(zip(joint_names, q[i])), pos[i], quat_wxyz[i])}
        if set(fk) != set(body_names):
            raise ValueError('FK body schema mismatch')
        all_pos.append([fk[n][0] for n in body_names])
        all_quat.append([fk[n][1].as_quat()[[3, 0, 1, 2]] for n in body_names])
    body_pos = np.asarray(all_pos)
    body_quat = np.asarray(all_quat)
    # Quaternion signs are continuous, although q/-q represent the same rotation.
    for i in range(1, len(times)):
        flip = np.sum(body_quat[i-1]*body_quat[i], axis=-1) < 0
        body_quat[i, flip] *= -1
    lin_vel = np.gradient(body_pos, 1/fps, axis=0, edge_order=2)
    ang_vel = np.zeros_like(lin_vel)
    for b in range(len(body_names)):
        br = Rotation.from_quat(body_quat[:, b, [1, 2, 3, 0]])
        ang_vel[1:-1, b] = (br[2:]*br[:-2].inv()).as_rotvec()*(fps/2)
        ang_vel[0, b] = (br[1]*br[0].inv()).as_rotvec()*fps
    # Analytic tail endpoint is static; finite difference endpoint is only approximate.
    lin_vel[-1] = 0.
    ang_vel[-1] = 0.
    arrays = {'fps': np.array([fps], np.int64), 'joint_pos': q.astype(np.float32),
              'joint_vel': dq.astype(np.float32), 'body_pos_w': body_pos.astype(np.float32),
              'body_quat_w': body_quat.astype(np.float32), 'body_lin_vel_w': lin_vel.astype(np.float32),
              'body_ang_vel_w': ang_vel.astype(np.float32), 'kick_leg': np.array('right')}
    if not all(np.isfinite(v).all() for k, v in arrays.items() if k != 'kick_leg'):
        raise ValueError('Nonfinite reference')
    limits = {j.attrib['name']: j.find('limit') for j in ET.parse(urdf).getroot().findall('joint')}
    lower = np.array([float(limits[n].attrib['lower']) for n in joint_names])
    upper = np.array([float(limits[n].attrib['upper']) for n in joint_names])
    violation = np.maximum(lower-q, q-upper).max(axis=0).clip(0)
    seam = round(a.run_duration*fps)
    metadata = {'purpose': 'kinematic command for bounded 29DoF physical actor probe; NOT a running result',
        'contact_feasibility_caveat': 'Raw running/tail FK can penetrate the ground; this is an unvalidated reference, never a physical capability.',
        'initialization_caveat': 'Reference begins in motion; does not establish standing-to-running transition.',
        'sources': {str(x): sha(x) for x in [source, standing, urdf, schema, Path(__file__)]},
        'source_fps': source_fps, 'output_fps': fps, 'frames': len(times),
        'run_duration_s': a.run_duration, 'tail_duration_s': a.tail_duration,
        'joint_names': joint_names, 'body_names': body_names,
        'root_quaternion_order': 'wxyz', 'angular_velocity_frame': 'world',
        'joint_interpolation': 'cubic Hermite, central finite-difference source tangents',
        'root_orientation_interpolation': 'RotationSpline; world-rotvec Hermite tail to PAiD standing orientation',
        'tail': 'C1 120 ms joint deceleration then standing blend to 009 frame0; not a dynamics guarantee',
        'tail_joint_brake_duration_s': brake_duration,
        'world_yaw_rad': a.yaw_rad,
        'seam': {'frame': seam, 'max_q_jump_rad': float(np.max(abs(q_tail(0.)-end_q))),
                 'max_dq_jump_rad_s': float(np.max(abs(q_tail(0., 1)-end_dq))),
                 'root_velocity_jump_mps': float(np.linalg.norm(p_tail(0., 1)-end_v)),
                 'max_sampled_q_step_rad': float(np.max(abs(np.diff(q[seam-1:seam+2], axis=0)))),
                 'max_sampled_dq_step_rad_s': float(np.max(abs(np.diff(dq[seam-1:seam+2], axis=0)))),
                 'initial_tail_joint_speed_max_rad_s': float(np.max(abs(end_dq)))},
        'max_joint_speed_rad_s': float(abs(dq).max()),
        'hard_joint_limit_violations_rad': {n: float(v) for n, v in zip(joint_names, violation) if v > 1e-7},
        'final_max_joint_speed_rad_s': float(abs(dq[-1]).max()),
        'shapes': {k: list(v.shape) for k, v in arrays.items()}}
    if metadata['hard_joint_limit_violations_rad']:
        raise ValueError('Conversion requires a different tail: '+str(metadata['hard_joint_limit_violations_rad']))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists() or a.output.with_suffix('.json').exists():
        raise FileExistsError('Do not overwrite prior probe artifacts: '+str(a.output))
    np.savez_compressed(a.output, **arrays)
    metadata['output'] = str(a.output)
    metadata['output_sha256'] = sha(a.output)
    a.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: metadata[k] for k in ['output', 'output_sha256', 'frames', 'run_duration_s',
          'tail_duration_s', 'seam', 'max_joint_speed_rad_s', 'hard_joint_limit_violations_rad',
          'final_max_joint_speed_rad_s', 'shapes']}, indent=2))


if __name__ == '__main__':
    main()
