"""Read-only ball-side contact evidence; no force-sign fitting or state writes.

Normal contacts and friction anchors have independent pair ranges. PhysX reuses
their count/start buffers, so each getter is copied before the next getter.
"""
import hashlib
from pathlib import Path

import numpy as np

from ground_contact_observer import indexed_entries
from strike_dynamics_telemetry import StrikeDynamicsObserver

NORMAL_TOLERANCE_N = .001
REACTION_TOLERANCE_N = .001
BALL_PATTERN = '/World/envs/env_*/SoccerBall'


def cpu_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def json_raw(value):
    """Preserve invalid numbers explicitly; never replace an unknown force by zero."""
    if isinstance(value, np.ndarray): value = value.tolist()
    if isinstance(value, (list, tuple)): return [json_raw(v) for v in value]
    if isinstance(value, dict): return {k: json_raw(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return 'NaN' if np.isnan(value) else '+Infinity' if value > 0 else '-Infinity'
    return value


def contact_evidence(normal_data, friction_data, matrix, friction_sign):
    """Validate and reconstruct every sensor/filter pair, without changing signs."""
    if friction_sign not in (-1, 1): raise ValueError('Certified friction sign must be +1 or -1')
    f, p, n, d, counts, starts = [np.asarray(v) for v in normal_data]
    ff, fp, fc, fs = [np.asarray(v) for v in friction_data]
    matrix = np.asarray(matrix)
    if (counts.ndim != 2 or counts.shape[0] != 1 or starts.shape != counts.shape or
            fc.shape != counts.shape or fs.shape != counts.shape or matrix.shape != counts.shape+(3,)):
        raise ValueError('One source actor and matching pair/matrix shapes required')
    if (f.ndim != 2 or f.shape[1] != 1 or d.shape != f.shape or p.shape != (len(f), 3) or n.shape != p.shape or
            ff.ndim != 2 or ff.shape[1] != 3 or fp.shape != ff.shape):
        raise ValueError('Unexpected normal or independent friction buffer shapes')
    if not np.isfinite(matrix).all(): raise ValueError('Nonfinite contact force matrix')
    normal = indexed_entries(dict(force_scalar_N=f[:, 0], point_w=p, normal_w=n,
                                   separation_m=d[:, 0], force_w_N=f*n), counts, starts)
    friction = indexed_entries(dict(point_w=fp, force_raw_w_N=ff, force_w_N=friction_sign*ff), fc, fs)
    for entries in (normal, friction):
        if len({r['buffer_index'] for r in entries}) != len(entries):
            raise ValueError('Overlapping pair ranges in one contact buffer')
    if any(r['force_scalar_N'] < 0 for r in normal): raise ValueError('Negative normal-force magnitude')
    reconstructed = np.zeros(matrix.shape); raw_friction = np.zeros(matrix.shape)
    for row in normal: reconstructed[tuple(row['pair_index'])] += row['force_w_N']
    for row in friction: raw_friction[tuple(row['pair_index'])] += row['force_raw_w_N']
    errors = np.linalg.norm(reconstructed-matrix, axis=-1)
    normal_ok = bool(np.all(errors <= NORMAL_TOLERANCE_N))
    reaches_end = lambda rows, size: bool(rows and max(r['buffer_index'] for r in rows)+1 >= size)
    exhausted = bool(len(normal) >= len(f) or len(friction) >= len(ff)
                     or reaches_end(normal, len(f)) or reaches_end(friction, len(ff)))
    normal_sum, friction_sum = matrix.sum(axis=(0, 1)), raw_friction.sum(axis=(0, 1))
    valid = normal_ok and not exhausted
    return dict(status='available' if valid else 'invalid', force_on='ball', frame='world',
        force_on_ball_valid=valid, force_on_ball_w_N=(normal_sum+friction_sign*friction_sum).tolist() if valid else None,
        normal_force_matrix_w_N=matrix.tolist(), raw_normal_reconstruction_matrix_w_N=reconstructed.tolist(),
        normal_force_on_ball_w_N=normal_sum.tolist(), normal_reconstruction_verified=normal_ok,
        normal_reconstruction_errors_N=errors.tolist(), normal_reconstruction_max_error_N=float(errors.max()),
        normal_reconstruction_tolerance_N=NORMAL_TOLERANCE_N, friction_sign=friction_sign,
        friction_raw_matrix_w_N=raw_friction.tolist(), friction_raw_w_N=friction_sum.tolist(),
        friction_force_on_ball_w_N=(friction_sign*friction_sum).tolist(),
        normal_contacts=normal, friction_anchors=friction, normal_counts=counts.tolist(), normal_starts=starts.tolist(),
        friction_counts=fc.tolist(), friction_starts=fs.tolist(), normal_buffer_capacity=len(f), friction_buffer_capacity=len(ff),
        normal_contact_count=len(normal), friction_anchor_count=len(friction), capacity_exhausted=exhausted)


def raw_ball_contacts(view, dt, friction_sign):
    """Copy raw SDK buffers immediately; retain malformed evidence as invalid."""
    if not np.isfinite(dt) or abs(dt-.005) > 1e-8: raise ValueError('Original finite 5 ms step required')
    raw = {}
    try:
        raw['normal'] = tuple(cpu_array(v) for v in view.get_contact_data(dt))
        raw['friction'] = tuple(cpu_array(v) for v in view.get_friction_data(dt))
        raw['normal_force_matrix_w_N'] = cpu_array(view.get_contact_force_matrix(dt))
        return contact_evidence(raw['normal'], raw['friction'], raw['normal_force_matrix_w_N'], friction_sign)
    except (ValueError, TypeError, IndexError, RuntimeError) as error:
        return dict(status='invalid', error=str(error), force_on='ball', frame='world',
                    force_on_ball_valid=False, force_on_ball_w_N=None, friction_sign=friction_sign,
                    raw_buffers=json_raw(raw))


def view_paths(view):
    paths = lambda x: [paths(v) for v in x] if isinstance(x, (list, tuple)) else str(x)
    return dict(sensor_paths=paths(view.sensor_paths), filter_paths=paths(view.filter_paths),
                sensor_count=int(view.sensor_count), filter_count=int(view.filter_count),
                max_contact_data_count=int(view.max_contact_data_count))


class BallContactObserver(StrikeDynamicsObserver):
    """Before/after scene-update hooks; measurements add only ball_contact_detail."""
    def __init__(self, env, probe_path, window=(1.45, 1.72)):
        super().__init__(env, probe_path)
        self.window = tuple(map(float, window))
        if len(self.window) != 2 or not np.isfinite(self.window).all() or not 0 <= self.window[0] < self.window[1] <= 14.:
            raise ValueError('Finite bounded ball-contact window required')
        self.ball = env.scene['soccer_ball']
        self.feet_indices = {s: self.robot.body_names.index(s+'_ankle_roll_link') for s in ('left', 'right')}
        simulation_view = env.scene['contact_forces']._physics_sim_view
        robot_paths = {name: '/World/envs/env_*/Robot/'+name for name in self.robot.body_names}
        feet = {s: robot_paths[s+'_ankle_roll_link'] for s in self.feet_indices}
        self.filters = dict(right_foot=[feet['right']], left_foot=[feet['left']],
            other_robot=[p for p in robot_paths.values() if p not in feet.values()], ground=list(self.ground))
        flat = [p for paths in self.filters.values() for p in paths]
        if len(flat) != len(set(flat)): raise ValueError('Contact groups overlap')
        self.ball_views = {group: simulation_view.create_rigid_contact_view(BALL_PATTERN, filters, max_contact_data_count=512)
                           for group, filters in self.filters.items()}
        self.pair_groups = {group: dict(source_actor_pattern=BALL_PATTERN, filter_patterns=self.filters[group], **view_paths(view))
                            for group, view in self.ball_views.items()}
        if any(v['sensor_count'] != 1 or v['filter_count'] != len(v['filter_patterns']) for v in self.pair_groups.values()):
            raise ValueError('Unexpected resolved ball/contact-filter coverage')
        self.reaction_views = {s: env.scene[s+'_ball_contact'].contact_physx_view for s in self.feet_indices}
        self.parameters = self.ball_parameters(simulation_view)
        self.previous_state = self.snapshot()
        self.pending_before = None
        self.detail_samples = 0

    def ball_parameters(self, simulation_view):
        import omni.usd
        from pxr import PhysxSchema
        view = self.ball.root_physx_view
        masses, disabled = cpu_array(view.get_masses()), cpu_array(view.get_disable_gravities())
        gravity_raw = simulation_view.get_gravity()
        gravity = np.array([gravity_raw[i] for i in range(3)], float)
        if masses.size != 1 or not np.isfinite(masses).all() or masses.flat[0] <= 0 or not np.isfinite(gravity).all():
            raise ValueError('Actual finite ball mass and scene gravity required')
        if disabled.size != 1 or disabled.flat[0] not in (0, 1): raise ValueError('Actual ball gravity flag required')
        actor_paths = list(view.prim_paths)
        if len(actor_paths) != 1: raise ValueError('One actual ball actor required')
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(actor_paths[0])
        api = PhysxSchema.PhysxRigidBodyAPI(prim)  # Read existing schema; never Apply/Create/Set.
        damping = {}
        for key, getter in [('linear_per_s', 'GetLinearDampingAttr'), ('angular_per_s', 'GetAngularDampingAttr')]:
            attr = getattr(api, getter)(); value = attr.Get()
            damping[key] = float(value) if value is not None and np.isfinite(value) else None
            damping[key+'_authored'] = bool(attr.HasAuthoredValueOpinion())
        props = self.ball.cfg.spawn.rigid_props
        damping.update(source='Read-only USD PhysxRigidBodyAPI resolved attributes, including schema defaults; not a measured damping impulse',
            config={key: getattr(props, key, None) for key in ('linear_damping', 'angular_damping', 'disable_gravity')})
        return dict(ball_mass_kg=float(masses.flat[0]), ball_mass_source='soccer_ball.root_physx_view.get_masses()',
            ball_actor_paths=actor_paths, gravity_w_mps2=gravity.tolist(), gravity_source='SimulationView.get_gravity()',
            configured_gravity_w_mps2=list(self.env.cfg.sim.gravity), ball_disable_gravity=bool(disabled.flat[0]),
            disable_gravity_source='soccer_ball.root_physx_view.get_disable_gravities()',
            effective_ball_gravity_w_mps2=(np.zeros(3) if disabled.flat[0] else gravity).tolist(), damping=damping,
            ball_com_pose_local_xyzw=json_raw(cpu_array(view.get_coms())),
            ball_inertia_com_kg_m2=json_raw(cpu_array(view.get_inertias())))

    def command_state(self):
        data = self.robot.data
        fields = dict(joint_position_target_rad='joint_pos_target', joint_velocity_target_rad_s='joint_vel_target',
            joint_effort_target_Nm='joint_effort_target', applied_torque_clipped_pd_estimate_Nm='applied_torque',
            computed_torque_pd_estimate_Nm='computed_torque')
        return {key: json_raw(cpu_array(getattr(data, name)[0])) for key, name in fields.items()}

    def snapshot(self):
        r, b = self.robot.data, self.ball.data
        value = lambda x: json_raw(cpu_array(x))
        ball = dict(link_position_w_m=value(b.root_link_pos_w[0]), link_quaternion_wxyz=value(b.root_link_quat_w[0]),
            link_origin_linear_velocity_w_mps=value(b.root_link_lin_vel_w[0]), com_position_w_m=value(b.root_com_pos_w[0]),
            com_quaternion_wxyz=value(b.root_com_quat_w[0]), com_linear_velocity_w_mps=value(b.root_com_lin_vel_w[0]),
            angular_velocity_w_rad_s=value(b.root_com_ang_vel_w[0]))
        feet = {s: dict(body_name=s+'_ankle_roll_link', position_w_m=value(r.body_pos_w[0, i]),
            quaternion_wxyz=value(r.body_quat_w[0, i]), link_origin_linear_velocity_w_mps=value(r.body_link_lin_vel_w[0, i]),
            angular_velocity_w_rad_s=value(r.body_ang_vel_w[0, i])) for s, i in self.feet_indices.items()}
        robot = dict(joint_position_rad=value(r.joint_pos[0]), joint_velocity_rad_s=value(r.joint_vel[0]), **self.command_state())
        return dict(ball=ball, feet=feet, robot=robot)

    def before_update(self):
        # Scene.update follows integration. Reuse frozen previous post-update
        # kinematics instead of accidentally querying the new PhysX state here.
        self.pending_before = dict(self.previous_state, robot=dict(self.previous_state['robot'], **self.command_state()))

    def after_update(self, dt):
        after = self.snapshot()
        if not self.active or dt <= 0:
            self.previous_state = after; self.pending_before = None
            return
        before = self.pending_before
        if before is None: raise ValueError('Ball observer requires before_update before native scene update')
        super().after_update(dt)
        now = self.rows[-1]['time_s']
        if self.window[0]-1e-9 <= now <= self.window[1]+1e-9:
            pairs = {group: dict(raw_ball_contacts(view, dt, self.friction_sign),
                source_actor_pattern=BALL_PATTERN, filter_patterns=self.filters[group]) for group, view in self.ball_views.items()}
            reactions = {}
            for side, view in self.reaction_views.items():
                matrix = cpu_array(view.get_contact_force_matrix(dt)); force = matrix.reshape(-1, 3).sum(0)
                ball_force = pairs[side+'_foot'].get('normal_force_on_ball_w_N')
                error = float(np.linalg.norm(force+ball_force)) if ball_force is not None and np.isfinite(matrix).all() else None
                verified = error is not None and error <= REACTION_TOLERANCE_N
                reactions[side] = dict(force_on=side+'_foot', normal_force_matrix_w_N=json_raw(matrix),
                    normal_force_on_foot_w_N=json_raw(force), opposed_ball_normal_error_N=error,
                    verified=verified, tolerance_N=REACTION_TOLERANCE_N, **view_paths(view))
                if not verified:
                    pairs[side+'_foot'].update(status='invalid', force_on_ball_valid=False, force_on_ball_w_N=None)
            self.rows[-1]['ball_contact_detail'] = dict(step_start_s=now-dt, step_end_s=now,
                ball_before=before['ball'], ball_after=after['ball'], feet_before=before['feet'], feet_after=after['feet'],
                robot_before=before['robot'], robot_after=after['robot'], pairs=pairs, foot_reaction_normal=reactions,
                ball_net_contact_force_w_N_raw=json_raw(cpu_array(self.ball_views['right_foot'].get_net_contact_forces(dt))))
            self.detail_samples += 1
        self.previous_state = after; self.pending_before = None

    def attach(self, result):
        super().attach(result)
        details = [r['ball_contact_detail'] for r in self.rows if 'ball_contact_detail' in r]
        expected = sum(self.window[0]-1e-9 <= i*.005 <= self.window[1]+1e-9 for i in range(1, 2801))
        result['dynamics_telemetry']['ball_contact_detail'] = dict(schema_version=1, window_s=list(self.window),
            samples=self.detail_samples, expected_samples=expected, window_complete=self.detail_samples == expected,
            physical_writes=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            indexed_entries_source_sha256=hashlib.sha256(Path(__file__).with_name('ground_contact_observer.py').read_bytes()).hexdigest(),
            probe_path=str(self.probe_path), probe_sha256=hashlib.sha256(self.probe_path.read_bytes()).hexdigest(),
            friction_sign=self.friction_sign, friction_sign_source='Recomputed original isolated v029 source-body sign certificate; never fitted to this ball replay',
            normal_reconstruction_tolerance_N=NORMAL_TOLERANCE_N, reaction_tolerance_N=REACTION_TOLERANCE_N,
            pair_groups=self.pair_groups, robot_body_names=list(self.robot.body_names), joint_names=list(self.robot.joint_names),
            ground_collision_paths=list(self.ground), **self.parameters,
            force_coverage_complete=bool(details) and self.detail_samples == expected and all(
                all(p.get('force_on_ball_valid') is True for p in d['pairs'].values()) for d in details),
            point_stage='Raw SDK solver contact/anchor points; exact material point and solver staging not independently certified',
            force_units='Step-average N from dt=0.005; matrix and normal*f independently retained, no normal flipping',
            state_frame='World coordinates; ball COM velocity is explicit. Foot velocity is at the ankle-roll link actor origin. Quaternions are wxyz unless a raw getter field says xyzw.',
            before_state_stage='Kinematics copied from preceding post-scene-update sample; commands and PD estimate buffers read in before_update, after targets for the current step were set',
            after_state_stage='Kinematics after current scene update; command/estimate buffers retained as read, not assumed to be post-integration measured actuator torque',
            torque_semantics='applied_torque is the existing clipped implicit-PD estimate; computed_torque is the corresponding PD estimate, neither is a pure measured actuator torque',
            coverage_scope='Mutually exclusive right/left ankle-roll links, all other named robot links, and original ground colliders; no assertion that unregistered external objects cannot exist',
            reaction_semantics='Existing foot-side sensor normal matrices oppose ball-side foot-filter matrices. Its max_contact_data_count=0 disables point detail, not aggregate matrix reporting.',
            net_force_semantics='Raw get_net_contact_forces output retained only as auxiliary evidence; its normal/friction composition is not independently certified and it is not added to the pair force sum',
            invalid_number_encoding='Nonfinite raw numbers are explicit strings NaN/+Infinity/-Infinity; invalid contact evidence never becomes a zero force')
        return result
