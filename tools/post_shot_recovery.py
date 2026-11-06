"""Bounded post-shot joint references over the unchanged v30 placement expert.

The protected prefix returns the exact original action. The residual's amplitude
and slew bounds constrain controller references, not physical joint limits.
There is deliberately no new soft-position projection: the original PD target
can already lie outside that interval. No method writes robot/ball state.

Features use the previous call's original v30 final action, not a nominal motion
sample. Evaluation caches features before advancing the original expert, matching
the observation supplied to the training actor. Import after AppLauncher in a
simulator process; the tensor/state contracts can be tested on CPU without Isaac.
"""
import hashlib
import math
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic

from placement_strike import PlacementStrikeTask

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOVERY_CONFIG = dict(maximum_rad=.12, maximum_step_rad=.02, onset_s=.12,
    recovery_fade_s=.20, checkpoint=None, checkpoint_sha256=None)
FEATURE_LAYOUT = (
    ('joint_position_relative_default_rad', 29), ('joint_velocity_over_limit', 29),
    ('projected_gravity_b', 3), ('root_velocity_w_over_3', 3),
    ('root_angular_velocity_w_over_5', 3), ('feet_position_pelvis_frame_m', 6),
    ('feet_link_origin_velocity_w_over_5', 6), ('feet_angular_velocity_w_over_10', 6),
    ('ground_force_over_force_plus_100', 2), ('capsule_height_over_point1', 2),
    ('capsule_descent_over_2', 2),
    ('phase_over_14_contact_activated_activation_age_over_14_standing_age_over_point2', 5),
    ('actual_delta_over_point12', 29), ('previous_baseline_action', 29))
FEATURE_WIDTH = sum(width for _, width in FEATURE_LAYOUT)
FEATURE_SLICES = {}
_offset = 0
for _name, _width in FEATURE_LAYOUT:
    FEATURE_SLICES[_name] = slice(_offset, _offset + _width)
    _offset += _width


def _quintic(value):
    u = value.clamp(0., 1.)
    return (u * u * u * (10. + u * (-15. + 6. * u))).clamp(0., 1.)


def _rotation(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


class PostShotRecoveryActor(ActorCritic):
    """One feed-forward 29-channel residual; no phase action or recurrent state."""
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, init_noise_std=.15, **kwargs):
        if num_actor_obs != FEATURE_WIDTH or num_actions != 29:
            raise ValueError(f'Expected {FEATURE_WIDTH} recovery observations and 29 actions')
        super().__init__(num_actor_obs, num_critic_obs, num_actions,
            actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
            activation='elu', init_noise_std=init_noise_std, **kwargs)
        torch.nn.init.zeros_(self.actor[-1].weight)
        torch.nn.init.zeros_(self.actor[-1].bias)


class PostShotRecoveryTask(PlacementStrikeTask):
    def __init__(self, env, config):
        if config.get('support_reference_frame'):
            raise ValueError('Post-shot recovery uses the v30 reference, without support_reference_frame')
        cfg = {**DEFAULT_RECOVERY_CONFIG, **config.get('post_shot_recovery', {})}
        for key in ('maximum_rad', 'maximum_step_rad', 'onset_s', 'recovery_fade_s'):
            if not math.isfinite(cfg[key]) or abs(cfg[key] - DEFAULT_RECOVERY_CONFIG[key]) > 1e-10:
                raise ValueError('Recovery bounds/timing must match the declared v32 protocol: ' + key)
        if abs(float(env.step_dt) - .02) > 1e-8 or abs(float(env.physics_dt) - .005) > 1e-8:
            raise ValueError('Original 20 ms control / 5 ms physics intervals required')
        super().__init__(env, config)
        if self.landing is None:
            raise ValueError('Original measured ball separation is required')
        self.recovery_cfg = cfg
        self.activation_time = torch.full((self.n,), -1., device=self.device)
        self.fade_start_time = self.activation_time.clone()
        self.activation_root_velocity = torch.zeros(self.n, 3, device=self.device)
        self.actual_delta = torch.zeros(self.n, 29, device=self.device)
        self.previous_delta = self.actual_delta.clone()
        self.fade_initial_delta = self.actual_delta.clone()
        self.recent_baseline_action = self.actual_delta.clone()
        self.last_output = self.actual_delta.clone()
        self._physics_steps = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self._recovery_time_s = torch.zeros(self.n, device=self.device)
        self._support_time_s = torch.zeros(self.n, 2, device=self.device)
        self._quality_time_s = torch.zeros(self.n, device=self.device)
        self._quality_integral = torch.zeros(self.n, device=self.device)
        self._quality_components = {}
        self._qd_ratio_peak = torch.zeros(self.n, device=self.device)
        self._nonfoot_force_peak = self._qd_ratio_peak.clone()
        self.last_recovery_reward_components = {}
        self.recovery_actor = None
        from running_balance import capsule_foot_local
        urdf = ROOT / 'configs/paid_original/paid_main_resolved.urdf'
        self._foot_capsules = [tuple(torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for value in capsule_foot_local(urdf, side)) for side in ('left', 'right')]
        from shank_clearance import ShankClearance
        self._recovery_shanks = self.shank_geometry or ShankClearance(urdf, self.robot.joint_names, self.device)
        sensor = self.env.scene['contact_forces']
        self._nonfoot_ids = [i for i, name in enumerate(sensor.body_names)
            if name not in ('left_ankle_roll_link', 'right_ankle_roll_link')]
        checkpoint, digest = cfg.get('checkpoint'), cfg.get('checkpoint_sha256')
        if bool(checkpoint) != bool(digest):
            raise ValueError('Recovery checkpoint and SHA256 must be provided together')
        if checkpoint:
            path = ROOT / checkpoint
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Recovery checkpoint SHA256 mismatch')
            saved = torch.load(path, map_location=self.device, weights_only=False)
            self.recovery_actor = PostShotRecoveryActor(FEATURE_WIDTH, FEATURE_WIDTH, 29).to(self.device)
            self.recovery_actor.load_state_dict(saved['model_state_dict'], strict=True)
            self.recovery_actor.eval()
            for parameter in self.recovery_actor.parameters(): parameter.requires_grad_(False)

    def _feet_state(self):
        r = self.robot.data
        position = r.body_pos_w[:, self.feet]
        rotation = _rotation(r.body_quat_w[:, self.feet])
        linear = r.body_link_lin_vel_w[:, self.feet]
        angular = r.body_ang_vel_w[:, self.feet]
        heights, velocities = [], []
        for side, (points, radii) in enumerate(self._foot_capsules):
            lever = torch.einsum('nij,pj->npi', rotation[:, side], points)
            # A capsule's ground-facing surface point is endpoint - radius*Z.
            # Include this radius lever in tangential material-point velocity.
            surface_lever = lever.clone(); surface_lever[:, :, 2] -= radii[None]
            h = position[:, side, 2, None] + surface_lever[:, :, 2] - self.env.scene.env_origins[:, 2, None]
            v = linear[:, side, None] + torch.linalg.cross(
                angular[:, side, None].expand_as(surface_lever), surface_lever, dim=-1)
            lowest, index = h.min(-1)
            heights.append(lowest)
            velocities.append(v[torch.arange(self.n, device=self.device), index])
        return position, linear, angular, torch.stack(heights, 1), torch.stack(velocities, 1)

    def recovery_state(self):
        from running_balance import ground_force
        r = self.robot.data
        _, linear, angular, height, point_velocity = self._feet_state()
        active = self.activation_time >= 0.
        finished = (self.fade_start_time >= 0.) & (self._recovery_time_s + 1e-7 >=
            self.fade_start_time + self.recovery_cfg['recovery_fade_s'])
        # Initial observation shape discovery precedes TerminationManager.
        manager = getattr(self.env, 'termination_manager', None)
        terminated = torch.zeros_like(active) if manager is None else manager.terminated
        # Quality supervision continues after the residual itself fades away.
        return dict(post_active=active, residual_active=active & ~finished,
            time_since_activation=torch.where(active, (self._recovery_time_s-self.activation_time).clamp_min(0.), 0.),
            root_height_m=r.root_pos_w[:, 2]-self.env.scene.env_origins[:, 2],
            gravity=r.projected_gravity_b, upright_cos=-r.projected_gravity_b[:, 2],
            root_velocity_w=r.root_lin_vel_w, root_angular_velocity_w=r.root_ang_vel_w,
            activation_root_velocity_w=self.activation_root_velocity,
            ground_force_N=ground_force(self.env), foot_link_velocity_w=linear,
            foot_angular_velocity_w=angular, capsule_height_m=height,
            capsule_lowest_point_velocity_w=point_velocity,
            capsule_descent_mps=(-point_velocity[:, :, 2]).clamp_min(0.),
            support_time_s=self._support_time_s,
            shank_gap_m=self._recovery_shanks.gap(r.joint_pos),
            substep_qd_ratio_peak=self._qd_ratio_peak,
            substep_nonfoot_force_peak=self._nonfoot_force_peak,
            actual_delta=self.actual_delta, previous_delta=self.previous_delta,
            terminated=terminated)

    def recovery_features(self):
        r = self.robot.data; state = self.recovery_state()
        feet_relative = torch.einsum('nji,nkj->nki', _rotation(r.root_quat_w),
            r.body_pos_w[:, self.feet]-r.root_pos_w[:, None])
        active = self.activation_time >= 0.
        standing_age = torch.where(self.expert.start >= 0.,
            (self._recovery_time_s-self.expert.start*.02).clamp_min(0.), 0.)
        phase = torch.stack((self.frame*.02/14., self.contact.float(), active.float(),
            state['time_since_activation']/14., standing_age/.2), -1)
        values = (r.joint_pos-r.default_joint_pos, r.joint_vel/r.joint_vel_limits,
            r.projected_gravity_b, r.root_lin_vel_w/3., r.root_ang_vel_w/5.,
            feet_relative.reshape(self.n, 6), state['foot_link_velocity_w'].reshape(self.n, 6)/5.,
            state['foot_angular_velocity_w'].reshape(self.n, 6)/10.,
            state['ground_force_N']/(state['ground_force_N']+100.),
            state['capsule_height_m']/.1, state['capsule_descent_mps']/2., phase,
            self.actual_delta/.12, self.recent_baseline_action)
        result = torch.cat(values, -1)
        if result.shape != (self.n, FEATURE_WIDTH) or not bool(torch.isfinite(result).all()):
            raise ValueError('Finite recovery feature contract required')
        return result

    def commands(self, actions=None):
        if actions is not None and (actions.shape != (self.n, 29) or not bool(torch.isfinite(actions).all())):
            raise ValueError('Finite (N,29) raw recovery actions required')
        # This is the observation available to the external training actor.
        features = self.recovery_features() if actions is None and self.recovery_actor is not None else None
        now = self.elapsed.clone()*.02
        base = super().commands()  # Exactly one unchanged expert invocation.
        if actions is None:
            with torch.no_grad():
                actions = self.recovery_actor.act_inference(features) if self.recovery_actor is not None else torch.zeros_like(base)
        if actions.shape != (self.n, 29) or not bool(torch.isfinite(actions).all()):
            raise ValueError('Finite (N,29) recovery actor output required')
        actions = actions.detach()
        plan = self.expert.plan
        ready = self.contact & (self.landing.separated_s+1e-8 >= .015) & (self.expert.start < 0.)
        ready &= (self.frame+1. >= plan['strike']['window'][-1])
        ready &= (self.frame+1. >= plan['strike_velocity']['window'][-1])
        ready &= (self.frame*.02 >= plan['stride']['window_s'][-1])
        ready &= self.step_velocity.abs().amax(-1) <= 1e-8
        ready &= torch.isfinite(now) & torch.isfinite(self.frame) & torch.isfinite(self.landing.separated_s)
        ready &= torch.isfinite(self.expert.start)
        ready &= torch.isfinite(self.robot.data.root_lin_vel_w).all(-1)
        trigger = ready & (self.activation_time < 0.)
        self.activation_time = torch.where(trigger, now, self.activation_time)
        self.activation_root_velocity = torch.where(trigger[:, None], self.robot.data.root_lin_vel_w,
            self.activation_root_velocity)
        active = self.activation_time >= 0.
        entering = active & (self.expert.start >= 0.) & (self.fade_start_time < 0.)
        self.fade_start_time = torch.where(entering, now, self.fade_start_time)
        self.fade_initial_delta = torch.where(entering[:, None], self.actual_delta, self.fade_initial_delta)
        onset = _quintic((now-self.activation_time)/self.recovery_cfg['onset_s'])
        requested = self.recovery_cfg['maximum_rad'] * torch.tanh(actions) * onset[:, None]
        fading = self.fade_start_time >= 0.
        fade = 1.-_quintic((now-self.fade_start_time)/self.recovery_cfg['recovery_fade_s'])
        requested = torch.where(fading[:, None], self.fade_initial_delta*fade[:, None], requested)
        requested = torch.where(active[:, None], requested, torch.zeros_like(requested))
        self.previous_delta = self.actual_delta.clone()
        maximum_step = self.recovery_cfg['maximum_step_rad']
        delta = self.actual_delta + (requested-self.actual_delta).clamp(-maximum_step, maximum_step)
        finished = fading & (now+1e-7 >= self.fade_start_time+self.recovery_cfg['recovery_fade_s'])
        delta = torch.where((~active | finished)[:, None], torch.zeros_like(delta), delta)
        # Frozen fade endpoint, the 200 ms duration, and the 20 ms step allow
        # the declared slew cap to settle exactly to zero without a target jump.
        scale = torch.as_tensor(self.env.action_manager.get_term('joint_pos')._scale,
            device=base.device, dtype=base.dtype)
        if not bool(torch.isfinite(scale).all()) or bool((scale == 0.).any()):
            raise ValueError('Finite nonzero original actuator action scales required')
        changed = delta != 0.
        out = torch.where(changed, base+delta/scale, base) if bool(changed.any()) else base
        # Track the correction actually representable in the original action dtype.
        self.actual_delta = (out-base)*scale
        self.recent_baseline_action = base.detach().clone()
        self.last_output = out.detach().clone()
        if self.n == 1 and self.debug:
            self.debug[-1]['post_shot_recovery'] = dict(
                activation_time_s=float(self.activation_time[0]), fade_start_time_s=float(self.fade_start_time[0]),
                active=bool(active[0] and not finished[0]), eligible=bool(ready[0]),
                onset=float(onset[0]) if bool(active[0]) else 0., fade=float(fade[0]) if bool(fading[0]) else 1.,
                actual_delta_rad=self.actual_delta[0].tolist(), previous_delta_rad=self.previous_delta[0].tolist(),
                baseline_action=base[0].tolist(), final_action=out[0].tolist(),
                original_phase_s=float(self.frame[0]*.02), ball_state_write=False)
        return out

    def _objective(self, state):
        from post_shot_objective import recovery_objective, DEFAULT_OBJECTIVE_CONFIG
        return recovery_objective(state, self.recovery_cfg.get('objective', DEFAULT_OBJECTIVE_CONFIG))

    def after_physics(self, dt):
        if dt == 0.:
            # Original initialization/reset forwarding is not a physics sample.
            super().after_physics(dt)
            return
        if not math.isfinite(dt) or abs(dt-.005) > 1e-8:
            raise ValueError('Original 5 ms quality sampling required')
        super().after_physics(dt)
        self._physics_steps += 1
        self._recovery_time_s = self._physics_steps.to(self.actual_delta.dtype)*.005
        state = self.recovery_state()
        force = state['ground_force_N']
        self._support_time_s = torch.where(torch.isfinite(force) & (force >= 80.), self._support_time_s+dt, 0.)
        qd = (self.robot.data.joint_vel/self.robot.data.joint_vel_limits).abs().amax(-1)
        nonfoot = self.env.scene['contact_forces'].data.net_forces_w[:, self._nonfoot_ids].norm(dim=-1).amax(-1)
        if not bool(torch.isfinite(qd).all() and torch.isfinite(nonfoot).all()):
            raise ValueError('Nonfinite 5 ms recovery quality measurement')
        self._qd_ratio_peak = torch.maximum(self._qd_ratio_peak, qd)
        self._nonfoot_force_peak = torch.maximum(self._nonfoot_force_peak, nonfoot)
        # Integrate this sample, not the running maximum from an earlier sample.
        state.update(support_time_s=self._support_time_s, substep_qd_ratio_peak=qd,
            substep_nonfoot_force_peak=nonfoot, terminated=torch.zeros_like(state['terminated']))
        rate, components = self._objective(state)
        self._quality_integral += rate.detach()*dt
        self._quality_time_s += dt
        for key, value in components.items():
            if key not in self._quality_components: self._quality_components[key] = torch.zeros_like(rate)
            self._quality_components[key] += value.detach()*dt

    def reward(self):
        available = self._quality_time_s > 0.
        if not bool(available.any()): return torch.zeros(self.n, device=self.device)
        state = self.recovery_state()
        _, terminal_components = self._objective(state)
        terminal = terminal_components['terminal_cost']
        duration = self._quality_time_s.clamp_min(.005)
        result = self._quality_integral/duration + terminal/self.env.step_dt
        result = torch.where(available, result, torch.zeros_like(result))
        self.last_recovery_reward_components = {key: value.clone()/duration for key, value in self._quality_components.items()}
        self.last_recovery_reward_components.update(terminal_cost=terminal.clone(),
            terminal_rate=terminal.clone()/self.env.step_dt,
            sampled_duration_s=self._quality_time_s.clone(), qd_ratio_peak=self._qd_ratio_peak.clone(),
            nonfoot_force_peak_N=self._nonfoot_force_peak.clone())
        self._quality_time_s.zero_(); self._quality_integral.zero_()
        self._qd_ratio_peak.zero_(); self._nonfoot_force_peak.zero_()
        for value in self._quality_components.values(): value.zero_()
        return result

    def reset_fields(self, ids):
        super().reset_fields(ids)
        self.activation_time[ids] = -1.; self.fade_start_time[ids] = -1.
        for name in ('activation_root_velocity', 'actual_delta', 'previous_delta', 'fade_initial_delta',
                'recent_baseline_action', 'last_output', '_recovery_time_s', '_support_time_s',
                '_physics_steps', '_quality_time_s', '_quality_integral', '_qd_ratio_peak', '_nonfoot_force_peak'):
            getattr(self, name)[ids] = 0.
        for value in self._quality_components.values(): value[ids] = 0.
        for value in self.last_recovery_reward_components.values(): value[ids] = 0.


def get_task(env):
    if not hasattr(env, '_post_shot_recovery_task'):
        env._post_shot_recovery_task = PostShotRecoveryTask(env, env.cfg.joint_strike)
    return env._post_shot_recovery_task


def task_features(env): return get_task(env).recovery_features()
def task_reward(env): return get_task(env).reward()
