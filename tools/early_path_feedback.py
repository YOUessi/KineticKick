"""Bounded world-horizontal feedback toward the instantaneous ball-goal line."""
import math
from numbers import Real

import torch


def lateral_correction(point_w, point_velocity_w, ball_w, ball_velocity_w, goal_w, *,
                       position_gain=.25, velocity_gain_s=.04, maximum_m=.04):
    """Return a lateral XY target offset and measured signed diagnostics.

    Inputs are finite floating Torch tensors of identical shape (..., 3), dtype
    and device. Positive error/velocity points left of the ball-to-goal heading.
    Velocity is the instantaneous projection of point-minus-ball velocity; it
    does not include rotation of that heading as the ball moves. Gains and the
    translation limit are finite nonnegative scalars. No input is modified.
    """
    values = (point_w, point_velocity_w, ball_w, ball_velocity_w, goal_w)
    if not isinstance(point_w, torch.Tensor) or point_w.ndim < 1 or point_w.shape[-1] != 3 or point_w.numel() == 0:
        raise ValueError('Nonempty world-vector tensors of shape (..., 3) required')
    for value in values:
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or value.shape != point_w.shape or value.dtype != point_w.dtype or value.device != point_w.device):
            raise ValueError('World vectors must share a floating dtype, device and shape (..., 3)')
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Finite world positions and velocities required')
    for value in (position_gain, velocity_gain_s, maximum_m):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0.:
            raise ValueError('Feedback gains and maximum translation must be finite nonnegative scalars')

    target = goal_w[..., :2] - ball_w[..., :2]
    distance = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
    if not bool(torch.isfinite(distance).all()) or bool((distance == 0.).any()):
        raise ValueError('A finite nonzero horizontal ball-to-goal distance is required')
    direction = target / distance
    normal = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
    error = ((point_w[..., :2]-ball_w[..., :2])*normal).sum(-1)
    velocity = ((point_velocity_w[..., :2]-ball_velocity_w[..., :2])*normal).sum(-1)
    requested = -position_gain*error - velocity_gain_s*velocity
    if not all(bool(torch.isfinite(value).all()) for value in (error, velocity, requested)):
        raise ValueError('Lateral error, relative velocity or requested correction overflowed')
    applied = requested.clamp(-maximum_m, maximum_m)
    return applied[..., None]*normal, dict(cross_error_m=error, cross_velocity_mps=velocity,
        requested_scalar_m=requested, applied_scalar_m=applied, target_direction_xy=direction)
