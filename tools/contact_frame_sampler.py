"""Sample capsule/sphere contact geometry at collision-generation time.

PhysX sphere/capsule contact generation returns the sphere-side point.
The capsule-side point is sphere_point - separation * outward_normal.
Both belong to the pre-integration pose, not the subsequent articulation pose.
Raw samples are preserved so this conversion can be independently audited.
"""
import numpy as np


def capsule_side_point(sphere_point, outward_normal, separation):
    # Accept NumPy arrays and torch tensors without changing device or dtype.
    point, normal = sphere_point, outward_normal
    gap = separation.reshape(-1, 1)
    if point.shape != normal.shape or point.shape != (len(gap), 3):
        raise ValueError('Matching point, normal and separation arrays required')
    return point - gap * normal


class ContactFrameSampler:
    def __init__(self, env):
        from running_balance import capsule_foot_local
        from pathlib import Path
        self.env, self.robot = env, env.scene['robot']
        self.ids, self.views, self.capsules = {}, {}, {}
        for side in ('left', 'right'):
            self.ids[side] = self.robot.body_names.index(side + '_ankle_roll_link')
            self.views[side] = env.scene[side + '_ball_contact']._physics_sim_view.create_rigid_contact_view(
                '/World/envs/env_*/Robot/' + side + '_ankle_roll_link',
                ['/World/envs/env_*/SoccerBall'], max_contact_data_count=64)
            points, radii = capsule_foot_local(Path(__file__).resolve().parents[1] / 'configs/paid_original/main.urdf', side)
            self.capsules[side] = (points[::2], points[1::2], radii[::2])
        self.cached = {}

    def before_update(self):
        self.previous_ball = self.env.scene['soccer_ball'].data.root_pos_w.clone()
        self.previous = {side: (self.robot.data.body_pos_w[:, idx].clone(),
                               self.robot.data.body_quat_w[:, idx].clone())
                         for side, idx in self.ids.items()}

    def after_update(self, dt):
        from isaaclab.utils.math import quat_apply_inverse
        for side, view in self.views.items():
            force, point, normal, separation, count, start = view.get_contact_data(dt)
            count, start = int(count[0, 0]), int(start[0, 0])
            sl = slice(start, start + count)
            f, p, n, gap = force[sl].clone(), point[sl].clone(), normal[sl].clone(), separation[sl].clone()
            pos, quat = self.previous[side]
            n[(n * (self.previous_ball - pos)).sum(-1) < 0] *= -1
            previous_quats = quat.expand(count, -1)
            corrected_world = capsule_side_point(p, n, gap)
            corrected_local = quat_apply_inverse(previous_quats, corrected_world - pos)
            previous_local = quat_apply_inverse(previous_quats, p - pos)
            corrected_normals = quat_apply_inverse(previous_quats, n)
            idx = self.ids[side]
            current_local = quat_apply_inverse(self.robot.data.body_quat_w[:, idx].expand(count, -1),
                                               p - self.robot.data.body_pos_w[:, idx])
            cpu = lambda x: x.detach().cpu().numpy()
            local = cpu(corrected_local)
            a, b, radii = self.capsules[side]
            segment = b - a
            delta = local[:, None, :] - a[None]
            u = np.clip((delta * segment).sum(-1) / np.maximum((segment * segment).sum(-1), 1e-12), 0., 1.)
            distance = np.linalg.norm(delta - u[:, :, None] * segment, axis=-1)
            residual = np.abs(distance - radii[None]).min(axis=1)
            self.cached[side] = dict(forces=cpu(f).reshape(-1), current_local=cpu(current_local),
                                    points=local, normals=cpu(corrected_normals), raw_world=cpu(p),
                                    previous_local=cpu(previous_local), separation=cpu(gap).reshape(-1),
                                    raw_normal_world=cpu(n), foot_position_world=cpu(pos)[0],
                                    foot_quaternion_world=cpu(quat)[0],
                                    capsule_surface_residual_m=residual)

    def events(self, original, forces, points, normals, *, foot, time_s):
        data = self.cached[foot]
        if (len(forces) != len(data['forces']) or not np.allclose(forces, data['forces'], atol=.002)
                or not np.allclose(points, data['current_local'], atol=2e-5)):
            raise ValueError('Contact views disagree; refusing to reframe unmatched samples')
        events = original(forces, data['points'], data['normals'], foot=foot, time_s=time_s)
        for i, event in enumerate(events):
            event.update(raw_sphere_point_world=data['raw_world'][i].tolist(),
                         raw_outward_normal_world=data['raw_normal_world'][i].tolist(),
                         contact_foot_position_w=data['foot_position_world'].tolist(),
                         contact_foot_quat_w=data['foot_quaternion_world'].tolist(),
                         raw_point_in_post_step_foot=data['current_local'][i].tolist(),
                         raw_point_in_pre_step_foot=data['previous_local'][i].tolist(),
                         raw_separation_m=float(data['separation'][i]),
                         foot_pose_time_s=time_s-self.env.physics_dt,
                         point_frame='pre_step_capsule_surface',
                         capsule_surface_residual_m=float(data['capsule_surface_residual_m'][i]))
        return events
