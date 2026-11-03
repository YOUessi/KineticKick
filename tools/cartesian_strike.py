"""Foot-pose feedback through bounded joint targets, with a free robot root."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


class LegKinematics:
    def __init__(self, urdf):
        self.names = ['right_' + name + '_joint' for name in
                      ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
        joints = ET.parse(urdf).getroot().findall('joint')
        chain = [next(j for j in joints if j.attrib['name'] == n) for n in self.names]
        self.origins, self.axes, self.limits = [], [], []
        for joint in chain:
            origin = joint.find('origin')
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            self.origins.append(transform)
            self.axes.append(np.fromstring(joint.find('axis').attrib['xyz'], sep=' '))
            self.limits.append([float(joint.find('limit').attrib[key]) for key in ('lower', 'upper')])
        self.limits = np.array(self.limits)

    def forward(self, q, root):
        transform = np.array(root, copy=True)
        axes, centers = [], []
        for origin, axis, angle in zip(self.origins, self.axes, q):
            transform = transform @ origin
            axes.append(transform[:3, :3] @ axis)
            centers.append(transform[:3, 3].copy())
            transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * angle).as_matrix()
        jacobian = np.vstack((np.cross(axes, transform[:3, 3] - centers).T, np.array(axes).T))
        return transform, jacobian

    def solve(self, target, root, seed):
        q = np.clip(seed, self.limits[:, 0], self.limits[:, 1]).copy()
        for _ in range(24):
            pose, jacobian = self.forward(q, root)
            position_error = target[:3, 3] - pose[:3, 3]
            rotation_error = Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec()
            if np.linalg.norm(position_error) < .0005 and np.linalg.norm(rotation_error) < .005:
                break
            error = np.r_[position_error, rotation_error * .15]
            jacobian[3:] *= .15
            delta = np.linalg.solve(jacobian.T @ jacobian + np.eye(6) * 1e-5, jacobian.T @ error)
            q = np.clip(q + np.clip(delta, -.20, .20), self.limits[:, 0], self.limits[:, 1])
        pose, _ = self.forward(q, root)
        return q, [float(np.linalg.norm(target[:3, 3]-pose[:3, 3])),
                   float(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec()))]


class CartesianStrike:
    def __init__(self, trials, joint_names, scale, offset, limits, urdf):
        from shooting_control import StrikeResidual
        self.entries = [t.get('cartesian') for t in trials]
        self.leg = LegKinematics(urdf)
        self.indices = [joint_names.index(n) for n in self.leg.names]
        self.envelope = StrikeResidual([e or {} for e in self.entries], joint_names, scale, offset, limits)
        self.seeds = [None for _ in trials]
        self.last_errors = [None for _ in trials]
        self.targets = []
        for trial, entry in zip(trials, self.entries):
            if entry is None:
                self.targets.append(None)
                continue
            normal = np.array(entry['normal_local'], float)
            normal /= np.linalg.norm(normal)
            ball = np.array([*trial.get('ball', [.14, -.93]), .11])
            direction = np.array([entry.get('aim_x', trial['target']['target_x']) - ball[0], trial['goal'][1] - ball[1], 0.])
            direction /= np.linalg.norm(direction)
            elevation = float(entry['elevation_rad'])
            desired_normal = direction*np.cos(elevation) + np.array([0, 0, np.sin(elevation)])
            base = Rotation.from_quat(entry['orientation_xyzw']).as_matrix()
            align, _ = Rotation.align_vectors([desired_normal], [base @ normal])
            rotation = align.as_matrix() @ base
            position = ball - rotation @ np.array(entry['point_local']) - .11 * desired_normal
            self.targets.append((rotation, position, direction))

    def apply(self, actions, frames, root_poses, joint_positions, dt):
        result = np.array(actions, copy=True)
        weights, _ = self.envelope.envelope(frames)
        for i, entry in enumerate(self.entries):
            if entry is None or weights[i, 0] <= 0:
                continue
            pose = root_poses[i]
            root = np.eye(4)
            root[:3, :3] = Rotation.from_quat([*pose[4:7], pose[3]]).as_matrix()
            root[:3, 3] = pose[:3]
            rotation, contact_position, direction = self.targets[i]
            target = np.eye(4)
            target[:3, :3] = rotation
            target[:3, 3] = contact_position + direction * entry['foot_speed_mps'] * (frames[i]-entry['contact_frame']) * dt
            seed = self.seeds[i] if self.seeds[i] is not None else np.array(entry.get('seed', joint_positions[i, self.indices]))
            q, self.last_errors[i] = self.leg.solve(target, root, seed)
            original = result[i, self.indices]*self.envelope.scale[i, self.indices] + self.envelope.offset[i, self.indices]
            if self.last_errors[i][0] > .015 or self.last_errors[i][1] > .15:
                self.seeds[i] = original.copy()
                continue
            self.seeds[i] = q
            limit = entry.get('max_joint_correction_rad', .3)
            modified = original + np.clip(q-original, -limit, limit) * weights[i, 0]
            bounds = self.envelope.limits[i, self.indices]
            modified = np.clip(modified, bounds[:, 0], bounds[:, 1])
            result[i, self.indices] = (modified-self.envelope.offset[i, self.indices])/self.envelope.scale[i, self.indices]
        return result
