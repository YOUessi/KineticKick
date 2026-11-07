"""Fixed pure-Torch training rewards for the bounded v35 pre-strike residual.

``prestrike_rate`` returns reward per second. Sample it at 5 ms, integrate
with dt, and supply the 20 ms mean rate to RewardManager. ``shot_event``
returns one event bonus, NOT a rate; runtime samples at first touch +80 ms,
consumes once and performs the event-to-rate conversion. Neither helper owns
time, latches, goal/limit/termination events, state writes or physical limits.

The +80 ms ball velocity is a measured training sample, not certified contact
separation. ``legal_right_proxy`` is a training proxy, not the original exact
collision-surface acceptance. The common 4--5.5 m/s soft band and these reward
weights do not replace the original task, instep, limit or 14 s recovery checks.
"""
import torch


def _mask(value, n, device, name):
    if (not isinstance(value, torch.Tensor) or value.shape != (n,)
            or value.dtype != torch.bool or value.device != device):
        raise ValueError(name+' must be a matching batched boolean tensor')


def _floating(value, shape, dtype, device, name):
    if (not isinstance(value, torch.Tensor) or value.shape != shape or not value.is_floating_point()
            or value.dtype != dtype or value.device != device):
        raise ValueError('Wrong shape/device/floating dtype for '+name)
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite '+name)


def _finite_outputs(values):
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise ValueError('Objective arithmetic overflowed; invalid reward is not masked or replaced')


def prestrike_rate(state):
    """Return [N] rate and additive [N] components, without modifying inputs.

    All terms require permanent ``eligible``; tracking and applied-residual
    terms additionally require ``pre_control``. The speed-margin cost remains
    active after control fades. Velocity tracks the supplied running reference:
    this helper never introduces the frozen post-shot objective's deceleration.
    ``qd_ratio`` is this actual 5 ms sample's maximum absolute joint-speed ratio.
    Delta fields are actually applied position-reference increments in radians.
    """
    eligible = state.get('eligible')
    if (not isinstance(eligible, torch.Tensor) or eligible.ndim != 1
            or eligible.dtype != torch.bool or len(eligible) == 0):
        raise ValueError('eligible must be a nonempty batched boolean tensor')
    n, device = len(eligible), eligible.device
    _mask(state.get('pre_control'), n, device, 'pre_control')
    height = state.get('root_height_m')
    if not isinstance(height, torch.Tensor) or not height.is_floating_point():
        raise ValueError('Floating measured height required')
    dtype = height.dtype
    shapes = {k:(n,) for k in ('root_height_m', 'reference_height_m', 'upright_cos', 'qd_ratio')}
    shapes.update({k:(n,3) for k in ('root_velocity_w', 'reference_velocity_w')})
    shapes.update({k:(n,29) for k in ('joint_position_error_rad', 'actual_delta', 'previous_delta')})
    for key, shape in shapes.items():
        _floating(state.get(key), shape, dtype, device, key)
    if bool((state['qd_ratio'] < 0.).any()) or bool((state['upright_cos'].abs() > 1.+1e-5).any()):
        raise ValueError('Nonnegative actual speed ratio and valid upright cosine required')

    pre = eligible & state['pre_control']
    components = dict(
        height_reward=1.5*torch.exp(-((height-state['reference_height_m'])/.1).square()),
        upright_reward=1.5*torch.exp(-(1.-state['upright_cos'].clamp(-1.,1.))/.1),
        velocity_reward=torch.exp(-((state['root_velocity_w']-state['reference_velocity_w'])/.8).square().sum(-1)),
        joint_tracking_reward=.5*torch.exp(-state['joint_position_error_rad'].square().mean(-1)/.25**2),
        residual_magnitude_cost=-.15*(state['actual_delta']/.04).square().mean(-1),
        residual_change_cost=-.1*((state['actual_delta']-state['previous_delta'])/.008).square().mean(-1))
    _finite_outputs(components.values())
    components = {key:torch.where(pre,value,torch.zeros_like(value)) for key,value in components.items()}
    margin = -.5*((state['qd_ratio']-.98).clamp_min(0.)/.03).clamp(0.,5.).square()
    components['joint_speed_margin_cost'] = torch.where(eligible,margin,torch.zeros_like(margin))
    rate = torch.stack(list(components.values())).sum(0)
    _finite_outputs([rate])
    return rate, components


def shot_event(velocity_w, direction_xy, eligible, legal_right_proxy):
    """One [N] bonus from measured world ball velocity and ball-to-goal direction.

    Direction is normalized internally; a zero direction is invalid even for
    an ineligible sample. Vertical velocity is not part of this XY objective.
    Components include unmasked speed diagnostics and the gated ``shot_bonus``;
    they are not all additive reward terms. Runtime owns the once-only event.
    """
    if (not isinstance(velocity_w, torch.Tensor) or velocity_w.ndim != 2
            or velocity_w.shape[1] != 3 or not velocity_w.is_floating_point() or len(velocity_w) == 0):
        raise ValueError('Nonempty floating N x 3 ball velocity required')
    n, dtype, device = len(velocity_w), velocity_w.dtype, velocity_w.device
    _floating(velocity_w, (n,3), dtype, device, 'velocity_w')
    _floating(direction_xy, (n,2), dtype, device, 'direction_xy')
    _mask(eligible, n, device, 'eligible')
    _mask(legal_right_proxy, n, device, 'legal_right_proxy')
    norm = torch.linalg.vector_norm(direction_xy, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norm).all()) or bool((norm == 0.).any()):
        raise ValueError('Finite nonzero horizontal goal direction required')
    direction = direction_xy/norm
    lateral = torch.stack((-direction[:,1],direction[:,0]),-1)
    forward_speed = (velocity_w[:,:2]*direction).sum(-1)
    lateral_speed = (velocity_w[:,:2]*lateral).sum(-1)
    speed_error = (4.-forward_speed).clamp_min(0.)+(forward_speed-5.5).clamp_min(0.)
    _finite_outputs([forward_speed,lateral_speed,speed_error])
    raw = 10.*torch.exp(-(lateral_speed/.5).square()-(speed_error/.5).square())
    bonus = torch.where(eligible & legal_right_proxy,raw,torch.zeros_like(raw))
    _finite_outputs([bonus])
    return bonus, dict(forward_speed_mps=forward_speed,lateral_speed_mps=lateral_speed,
                      speed_band_error_mps=speed_error,shot_bonus=bonus)
