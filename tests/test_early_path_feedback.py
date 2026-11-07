"""CPU geometry contracts for bounded horizontal path feedback."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch


def feedback(*args, **kwargs):
    path = Path(__file__).resolve().parents[1] / 'tools/early_path_feedback.py'
    assert path.is_file(), 'The lateral path feedback helper is not implemented'
    spec = importlib.util.spec_from_file_location('early_path_feedback', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.lateral_correction(*args, **kwargs)


def inputs():
    point = torch.tensor([[.3, .08, .2], [.7, -.04, -.4], [.2, 0., 2.]], dtype=torch.float64)
    velocity = torch.tensor([[2., 0., 3.], [-1., 0., -4.], [8., .8, -2.]], dtype=torch.float64)
    ball = torch.zeros_like(point)
    ball_velocity = torch.zeros_like(point); ball_velocity[2, 1] = .3
    goal = torch.tensor([[2., 0., .11]], dtype=torch.float64).expand_as(point).clone()
    return [point, velocity, ball, ball_velocity, goal]


def test_measured_error_and_relative_velocity_have_restoring_sign_without_mutation():
    values = inputs(); originals = [x.clone() for x in values]
    delta, info = feedback(*values)
    torch.testing.assert_close(info['cross_error_m'], torch.tensor([.08, -.04, 0.], dtype=torch.float64))
    torch.testing.assert_close(info['cross_velocity_mps'], torch.tensor([0., 0., .5], dtype=torch.float64))
    torch.testing.assert_close(delta, torch.tensor([[0., -.02], [0., .01], [0., -.02]], dtype=torch.float64))
    assert all(torch.equal(x, before) for x, before in zip(values, originals))
    assert delta.shape == (3, 2) and delta.dtype == values[0].dtype
    assert torch.equal((delta*info['target_direction_xy']).sum(-1), torch.zeros(3, dtype=delta.dtype))


def test_common_translation_rotation_and_velocity_boost_preserve_geometry():
    values = inputs(); base, before = feedback(*values)
    angle = .731
    turn = torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
                         [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]], dtype=torch.float64)
    shift = torch.tensor([7., -3., 4.], dtype=torch.float64)
    boost = torch.tensor([.7, -.3, 1.1], dtype=torch.float64)
    moved = [x@turn.T + (boost if i in (1, 3) else shift) for i, x in enumerate(values)]
    actual, after = feedback(*moved)
    torch.testing.assert_close(actual, base@turn[:2, :2].T)
    torch.testing.assert_close(after['target_direction_xy'], before['target_direction_xy']@turn[:2, :2].T)
    for key in ('cross_error_m', 'cross_velocity_mps', 'requested_scalar_m', 'applied_scalar_m'):
        torch.testing.assert_close(after[key], before[key])


def test_clamp_zero_limit_and_forward_vertical_changes_do_not_add_other_motion():
    values = inputs(); values[0][:, 1] = torch.tensor([1., -1., 0.], dtype=torch.float64)
    delta, info = feedback(*values)
    torch.testing.assert_close(info['requested_scalar_m'], torch.tensor([-.25, .25, -.02], dtype=torch.float64))
    torch.testing.assert_close(info['applied_scalar_m'], torch.tensor([-.04, .04, -.02], dtype=torch.float64))
    assert bool((delta.norm(dim=-1) <= .04).all())
    _, configured = feedback(*values, maximum_m=.03)
    torch.testing.assert_close(configured['applied_scalar_m'], torch.tensor([-.03, .03, -.02], dtype=torch.float64))
    moved = [x.clone() for x in values]
    moved[0][:, 0] += 20.; moved[0][:, 2] -= 5.
    moved[1][:, 0] -= 30.; moved[1][:, 2] += 15.
    torch.testing.assert_close(feedback(*moved)[0], delta)
    assert torch.equal(feedback(*values, maximum_m=0.)[0], torch.zeros_like(delta))
    assert torch.equal(feedback(*values, position_gain=0., velocity_gain_s=0.)[0], torch.zeros_like(delta))
    single, _ = feedback(*(x[0] for x in values))
    torch.testing.assert_close(single, delta[0])


@pytest.mark.parametrize('bad', ['zero_line', 'nan_position', 'inf_velocity', 'wrong_shape',
                                'integer_dtype', 'mixed_dtype', 'negative_gain', 'nan_gain', 'negative_bound'])
def test_undefined_geometry_or_invalid_values_are_rejected(bad):
    values = inputs(); kwargs = {}
    if bad == 'zero_line': values[4][1, :2] = values[2][1, :2]
    elif bad == 'nan_position': values[0][0, 2] = float('nan')
    elif bad == 'inf_velocity': values[3][0, 0] = float('inf')
    elif bad == 'wrong_shape': values[4] = values[4][:, :2]
    elif bad == 'integer_dtype': values = [x.to(torch.int64) for x in values]
    elif bad == 'mixed_dtype': values[0] = values[0].float()
    elif bad == 'negative_gain': kwargs['position_gain'] = -.25
    elif bad == 'nan_gain': kwargs['velocity_gain_s'] = float('nan')
    elif bad == 'negative_bound': kwargs['maximum_m'] = -.04
    with pytest.raises(ValueError): feedback(*values, **kwargs)
