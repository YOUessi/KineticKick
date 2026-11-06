"""One-shot planar support frame for reference observations, using Torch only.

The caller supplies the strike/standing readiness gate. This module never calls
a policy, changes its phase, or writes physical state. All transforms preserve
world Z and use one frozen support pivot and one shortest yaw displacement.
"""
import math

import torch


class SupportReferenceFrame:
    def __init__(self, n, device, config):
        self.n = n
        self.cfg = dict(config)
        keys = ('minimum_force_N', 'support_confirmation_s', 'duration_s')
        if not isinstance(n, int) or n < 1:
            raise ValueError('A positive environment count is required')
        if not all(k in config and math.isfinite(config[k]) and config[k] > 0. for k in keys):
            raise ValueError('Finite positive support-frame settings are required')
        self.support_time = torch.zeros(n, 2, device=device)
        self.trigger_time = torch.full((n,), -1., device=device)
        self.selected_side = torch.full((n,), -1, dtype=torch.long, device=device)
        self.reference_pivot = torch.zeros(n, 3, device=device)
        self.actual_pivot = torch.zeros(n, 3, device=device)
        self.yaw_delta = torch.zeros(n, device=device)
        self.alpha = torch.zeros(n, device=device)
        self.yaw_rate = torch.zeros(n, device=device)
        self.rotation = torch.eye(3, device=device).expand(n, 3, 3).clone()

    @staticmethod
    def _shape(name, tensor, shape):
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape:
            raise ValueError(f'{name} must be a tensor with shape {shape}')

    def reset(self, ids):
        self.support_time[ids] = 0.
        self.trigger_time[ids] = -1.
        self.selected_side[ids] = -1
        self.reference_pivot[ids] = 0.
        self.actual_pivot[ids] = 0.
        self.yaw_delta[ids] = 0.
        self.alpha[ids] = 0.
        self.yaw_rate[ids] = 0.
        self.rotation[ids] = torch.eye(3, device=self.rotation.device, dtype=self.rotation.dtype)

    def after_physics(self, support_N, dt=.005):
        self._shape('support_N', support_N, (self.n, 2))
        if not math.isfinite(dt) or abs(dt - .005) > 1e-8:
            raise ValueError('The original 5 ms physics interval is required')
        supported = torch.isfinite(support_N) & (support_N >= self.cfg['minimum_force_N'])
        self.support_time = torch.where(supported, self.support_time + dt, torch.zeros_like(self.support_time))

    def update(self, now, ready_mask, support_N, reference_feet, actual_feet, reference_yaw, actual_yaw):
        for name, tensor, shape in [('now', now, (self.n,)), ('ready_mask', ready_mask, (self.n,)),
                ('support_N', support_N, (self.n, 2)), ('reference_feet', reference_feet, (self.n, 2, 3)),
                ('actual_feet', actual_feet, (self.n, 2, 3)), ('reference_yaw', reference_yaw, (self.n,)),
                ('actual_yaw', actual_yaw, (self.n,))]:
            self._shape(name, tensor, shape)
        if ready_mask.dtype != torch.bool:
            raise ValueError('ready_mask must be boolean')

        valid_time = torch.isfinite(now) & (now >= 0.)
        finite = valid_time & torch.isfinite(support_N).all(-1)
        finite &= torch.isfinite(reference_feet).all(dim=(-2, -1)) & torch.isfinite(actual_feet).all(dim=(-2, -1))
        finite &= torch.isfinite(reference_yaw) & torch.isfinite(actual_yaw)
        difference = actual_yaw - reference_yaw
        angle = torch.atan2(torch.sin(difference), torch.cos(difference))
        finite &= torch.isfinite(angle)
        eligible = ((self.support_time + 1e-8 >= self.cfg['support_confirmation_s']) &
                    (support_N >= self.cfg['minimum_force_N']) & torch.isfinite(support_N))
        # argmax breaks an equal-force tie toward left (index 0).
        side = torch.where(eligible, support_N, torch.full_like(support_N, -float('inf'))).argmax(-1)
        trigger = (self.trigger_time < 0.) & ready_mask & finite & eligible.any(-1)
        index = side[:, None, None].expand(-1, 1, 3)
        reference = reference_feet.gather(1, index).squeeze(1)
        actual = actual_feet.gather(1, index).squeeze(1)
        self.reference_pivot = torch.where(trigger[:, None], reference, self.reference_pivot)
        self.actual_pivot = torch.where(trigger[:, None], actual, self.actual_pivot)
        self.yaw_delta = torch.where(trigger, angle, self.yaw_delta)
        self.trigger_time = torch.where(trigger, now, self.trigger_time)
        self.selected_side = torch.where(trigger, side, self.selected_side)

        u = ((now - self.trigger_time) / self.cfg['duration_s']).clamp(0., 1.)
        smooth = (u * u * u * (10. + u * (-15. + 6. * u))).clamp(0., 1.)
        slope = 30. * u.square() * (1. - u).square() / self.cfg['duration_s']
        active = self.trigger_time >= 0.
        advance = valid_time & active & (now >= self.trigger_time)
        # Invalid time cannot contaminate a previously locked frame. Normal
        # callers provide monotonically increasing policy times.
        self.alpha = torch.where(advance, smooth, self.alpha)
        self.yaw_rate = torch.where(advance, slope * self.yaw_delta, self.yaw_rate)
        theta = self.alpha * self.yaw_delta
        c, s = torch.cos(theta), torch.sin(theta)
        zero, one = torch.zeros_like(theta), torch.ones_like(theta)
        self.rotation = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(self.n, 3, 3)
        return {key: getattr(self, key) for key in ('rotation', 'alpha', 'yaw_rate', 'trigger_time',
            'selected_side', 'reference_pivot', 'actual_pivot', 'yaw_delta', 'support_time')}

    def _transform(self, values, positions):
        if (not isinstance(values, torch.Tensor) or values.ndim < 2 or values.shape[0] != self.n
                or values.shape[-1] != 3 or not values.is_floating_point()):
            raise ValueError('Reference values must be floating tensors shaped [N, ..., 3]')
        flat = values.reshape(self.n, -1, 3)
        rotation = self.rotation.to(dtype=values.dtype)
        if positions:
            pivot = self.reference_pivot.to(dtype=values.dtype)[:, None, :]
            translation = self.alpha.to(dtype=values.dtype)[:, None, None] * (
                self.actual_pivot - self.reference_pivot).to(dtype=values.dtype)[:, None, :]
            transformed = pivot + torch.bmm(flat - pivot, rotation.transpose(1, 2)) + translation
        else:
            transformed = torch.bmm(flat, rotation.transpose(1, 2))
        # Planar means exact original Z, including during the moving blend.
        transformed = torch.cat((transformed[..., :2], flat[..., 2:3]), -1)
        active = ((self.trigger_time >= 0.) & (self.alpha > 0.))[:, None, None]
        return torch.where(active, transformed, flat).reshape(values.shape)

    def transform_positions(self, values):
        """Rz(alpha*yaw) about the reference pivot, plus alpha times XY shift."""
        return self._transform(values, positions=True)

    def transform_vectors(self, values):
        """Apply the same yaw rotation, without translation; preserve Z."""
        return self._transform(values, positions=False)
