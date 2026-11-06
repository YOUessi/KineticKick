"""Measured post-shot recovery objective, independent of simulator or controller.

All state tensors are batched; world Z is upward and the ground is horizontal.
``capsule_lowest_point_velocity_w`` is the rigid-body velocity at the actual
lowest capsule surface point, not link-origin velocity or a finite difference
of the moving minimum. ``shank_gap_m`` is signed original-URDF capsule clearance.
``ground_force_N`` contains nonnegative ground-filtered vertical normal loads;
it is a support gate, never a horizontal-force or energy-transfer estimate.

Evaluate each measured 5 ms sample, integrate reward rate with dt, and expose
the 20 ms mean rate to RewardManager (which integrates its own step_dt). The
two ``substep_*_peak`` inputs at a 5 ms call mean maxima across joints/bodies
in THAT sample. A public debug state may separately retain the 20 ms peak.

Terminal cost is an event, not a repeated sample cost. Physics-sample callers
set terminated=False. Once termination is known at the policy boundary, add
components['terminal_cost']/policy_dt_s to the accumulated mean rate exactly
once. A direct policy-boundary call returns that converted terminal rate too.

These fixed scales are engineering shaping choices. They do not certify a
feasible trajectory, original-task success, actuator torque, or human motion.
"""
import math

import torch


DEFAULT_OBJECTIVE_CONFIG = {
    'policy_dt_s': .02,
    'height_target_m': .78,
    'height_scale_m': .10,
    'upright_scale': .10,
    'deceleration_s': 2.,
    'velocity_scale_mps': .6,
    'vertical_velocity_scale_mps': .5,
    'root_angular_scale_rad_s': 2.,
    'near_ground_height_m': .10,
    'descent_gate_mps': .10,
    'approach_normal_scale_mps': .75,
    'approach_tangent_scale_mps': 1.5,
    'approach_angular_scale_rad_s': 6.,
    'loaded_slip_scale_mps': .5,
    'unloaded_force_max_N': 5.,
    'support_minimum_force_N': 80.,
    'support_confirmation_s': .015,
    'shank_margin_m': .02,
    'shank_scale_m': .02,
    'joint_speed_excess_scale': .05,
    'nonfoot_force_scale_N': 20.,
    'residual_scale_rad': .12,
    'residual_change_scale_rad': .02,
    'normalized_squared_cost_cap': 25.,
    'weights': {
        'height': 1.5,
        'upright': 2.,
        'planar_velocity': 2.,
        'vertical_velocity': .4,
        'root_angular': .3,
        'approach_normal': 1.,
        'approach_tangent': 1.,
        'approach_angular': .25,
        'loaded_slip': .8,
        'handover': 1.5,
        'shank_clearance': 3.,
        'joint_speed': 4.,
        'nonfoot': 4.,
        'residual_magnitude': .15,
        'residual_change': .10,
        'terminal': 20.,
    },
}


def _settings(config):
    if not isinstance(config, dict) or set(config)-set(DEFAULT_OBJECTIVE_CONFIG):
        raise ValueError('Unknown or invalid objective configuration')
    cfg = {**DEFAULT_OBJECTIVE_CONFIG, **config}
    supplied = config.get('weights', {})
    if not isinstance(supplied, dict) or set(supplied)-set(DEFAULT_OBJECTIVE_CONFIG['weights']):
        raise ValueError('Unknown objective weight')
    cfg['weights'] = {**DEFAULT_OBJECTIVE_CONFIG['weights'], **supplied}
    for key, value in cfg.items():
        if key == 'weights':
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.:
            raise ValueError('Finite positive objective scale required: '+key)
    for key, value in cfg['weights'].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.:
            raise ValueError('Finite nonnegative objective weight required: '+key)
    if cfg['unloaded_force_max_N'] >= cfg['support_minimum_force_N']:
        raise ValueError('Unloaded and confirmed-support force ranges must remain separate')
    return cfg


def _validate_state(state):
    active = state.get('post_active')
    if not isinstance(active, torch.Tensor) or active.ndim != 1 or active.dtype != torch.bool or not len(active):
        raise ValueError('post_active must be a nonempty batched boolean tensor')
    n = len(active)
    shape = {key:(n,) for key in ('time_since_activation','root_height_m','upright_cos','shank_gap_m',
                                  'substep_qd_ratio_peak','substep_nonfoot_force_peak')}
    shape.update({key:(n,3) for key in ('root_velocity_w','root_angular_velocity_w','activation_root_velocity_w')})
    shape.update({key:(n,2) for key in ('ground_force_N','capsule_height_m','support_time_s')})
    shape.update({key:(n,2,3) for key in ('foot_angular_velocity_w','capsule_lowest_point_velocity_w')})
    shape.update({key:(n,29) for key in ('actual_delta','previous_delta')})
    floats = []
    dtype = None
    for key, expected in shape.items():
        value = state.get(key)
        if (not isinstance(value, torch.Tensor) or value.shape != expected or not value.is_floating_point()
                or value.device != active.device or (dtype is not None and value.dtype != dtype)):
            raise ValueError('Wrong shape/device/floating dtype for '+key)
        dtype = value.dtype
        floats.append(value.reshape(n,-1))
    terminated = state.get('terminated')
    if (not isinstance(terminated, torch.Tensor) or terminated.shape != (n,) or terminated.dtype != torch.bool
            or terminated.device != active.device):
        raise ValueError('terminated must match the batched boolean mask')
    if not bool(torch.isfinite(torch.cat(floats, -1)).all()):
        raise ValueError('Nonfinite measured objective state')
    nonnegative = torch.cat([state[k].reshape(n,-1) for k in
        ('ground_force_N','support_time_s','substep_qd_ratio_peak','substep_nonfoot_force_peak')], -1)
    if bool((nonnegative < 0.).any()) or bool((state['upright_cos'].abs() > 1.+1e-5).any()):
        raise ValueError('Invalid force, duration, speed ratio, or upright cosine')
    if bool((active & (state['time_since_activation'] < 0.)).any()):
        raise ValueError('Active elapsed time must be nonnegative')


def recovery_objective(state, config):
    """Return [N] reward rate and named [N] signed contributions.

    Every component is a rate except ``terminal_cost`` (one event cost).
    ``terminal_rate`` is that same event divided by policy_dt_s; do not add both
    to a scalar total. All components and rewards are exactly zero when inactive.
    No input tensor is modified. Missing/invalid measurements raise ValueError.
    """
    cfg = _settings(config)
    _validate_state(state)
    w, cap = cfg['weights'], cfg['normalized_squared_cost_cap']
    bound = math.sqrt(cap)

    def square(value, scale):
        # Clip before squaring, so extreme finite samples cannot overflow a cost.
        return (value/scale).clamp(-bound,bound).square()

    def norm_square(value, scale):
        return square(value, scale).sum(-1).clamp_max(cap)

    t = state['time_since_activation'].clamp_min(0.)
    u = (t/cfg['deceleration_s']).clamp(0.,1.)
    stop = (u*u*u*(10.+u*(-15.+6.*u))).clamp(0.,1.)
    desired_velocity = state['activation_root_velocity_w'][:,:2]*(1.-stop[:,None])
    root_velocity = state['root_velocity_w']
    c = {
        'height_reward': w['height']*torch.exp(-square(state['root_height_m']-cfg['height_target_m'],cfg['height_scale_m'])),
        'upright_reward': w['upright']*torch.exp(-(1.-state['upright_cos'].clamp(-1.,1.))/cfg['upright_scale']),
        'planar_velocity_reward': w['planar_velocity']*torch.exp(-norm_square(root_velocity[:,:2]-desired_velocity,cfg['velocity_scale_mps'])),
        'vertical_velocity_cost': -w['vertical_velocity']*square(root_velocity[:,2],cfg['vertical_velocity_scale_mps']),
        'root_angular_cost': -w['root_angular']*norm_square(state['root_angular_velocity_w'],cfg['root_angular_scale_rad_s']),
    }

    force = state['ground_force_N']
    foot_velocity = state['capsule_lowest_point_velocity_w']
    descent = (-foot_velocity[:,:,2]).clamp_min(0.)
    proximity = ((cfg['near_ground_height_m']-state['capsule_height_m'])/cfg['near_ground_height_m']).clamp(0.,1.)
    approaching = proximity*(descent/cfg['descent_gate_mps']).clamp(0.,1.)*(force <= cfg['unloaded_force_max_N'])
    loaded = force >= cfg['support_minimum_force_N']
    confirmed = loaded & (state['support_time_s']+1e-8 >= cfg['support_confirmation_s'])
    c.update({
        'approach_normal_cost': -w['approach_normal']*(approaching*square(descent,cfg['approach_normal_scale_mps'])).sum(-1),
        'approach_tangent_cost': -w['approach_tangent']*(approaching*norm_square(foot_velocity[:,:,:2],cfg['approach_tangent_scale_mps'])).sum(-1),
        'approach_angular_cost': -w['approach_angular']*(approaching*norm_square(state['foot_angular_velocity_w'],cfg['approach_angular_scale_rad_s'])).sum(-1),
        'loaded_slip_cost': -w['loaded_slip']*(loaded*norm_square(foot_velocity[:,:,:2],cfg['loaded_slip_scale_mps'])).sum(-1),
        # This is a close descending-foot handover risk, not a blanket flight cost.
        'handover_cost': -w['handover']*(approaching*(~confirmed.flip(-1))).amax(-1),
        'shank_clearance_cost': -w['shank_clearance']*square((cfg['shank_margin_m']-state['shank_gap_m']).clamp_min(0.),cfg['shank_scale_m']),
        'joint_speed_cost': -w['joint_speed']*square((state['substep_qd_ratio_peak']-1.).clamp_min(0.),cfg['joint_speed_excess_scale']),
        'nonfoot_cost': -w['nonfoot']*square(state['substep_nonfoot_force_peak'],cfg['nonfoot_force_scale_N']),
        'residual_magnitude_cost': -w['residual_magnitude']*square(state['actual_delta'],cfg['residual_scale_rad']).mean(-1),
        'residual_change_cost': -w['residual_change']*square(state['actual_delta']-state['previous_delta'],cfg['residual_change_scale_rad']).mean(-1),
        'terminal_cost': -w['terminal']*state['terminated'].to(root_velocity.dtype),
    })
    c['terminal_rate'] = c['terminal_cost']/cfg['policy_dt_s']
    c = {key:torch.where(state['post_active'],value,torch.zeros_like(value)) for key,value in c.items()}
    reward = torch.stack([value for key,value in c.items() if key != 'terminal_cost']).sum(0)
    return reward, c
