#!/usr/bin/env python3
"""Rebuild the stationary receive reference from a saved, explicit ready pose.

Run in a separate CPU process before the Isaac process imports numerical libs.
The reference initializes the free robot once; it does not animate the robot.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from running_reference_geometry import ReferenceGeometry

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    output = a.output or ROOT / plan['reference']
    if output.exists():
        raise FileExistsError(output)
    geo = ReferenceGeometry(ROOT)
    if geo.joint_names != plan['joint_names']:
        raise ValueError('Ready pose and runtime joint schema differ')
    q = np.array(plan['ready_joint_pos'], dtype=float)
    root = np.array(plan['root_pose'], dtype=float)
    poses = geo.body_poses(q, root[:3], root[3:])
    positions = np.array([poses[n][0] for n in geo.body_names])
    rotations = np.array([poses[n][1].as_quat()[[3, 0, 1, 2]] for n in geo.body_names])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, fps=np.array([50]),
             joint_pos=np.repeat(q[None], 250, axis=0).astype('f'),
             joint_vel=np.zeros((250, 29), 'f'),
             body_pos_w=np.repeat(positions[None], 250, axis=0).astype('f'),
             body_quat_w=np.repeat(rotations[None], 250, axis=0).astype('f'),
             body_lin_vel_w=np.zeros((250, 30, 3), 'f'),
             body_ang_vel_w=np.zeros((250, 30, 3), 'f'), kick_leg=np.array('right'))
    print(output)


if __name__ == '__main__':
    main()
