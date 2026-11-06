"""CPU reward contracts: physical signs, protected stage, and bounded costs."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch


def module():
    path = Path(__file__).resolve().parents[1] / 'tools/post_shot_objective.py'
    assert path.is_file(), 'The declared post-shot objective is not implemented'
    spec = importlib.util.spec_from_file_location('post_shot_objective', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def state(n=1):
    return dict(post_active=torch.ones(n, dtype=torch.bool), time_since_activation=torch.zeros(n),
        root_height_m=torch.full((n,), .78), upright_cos=torch.ones(n),
        root_velocity_w=torch.zeros(n, 3), root_angular_velocity_w=torch.zeros(n, 3),
        activation_root_velocity_w=torch.zeros(n, 3), ground_force_N=torch.full((n, 2), 160.),
        foot_angular_velocity_w=torch.zeros(n, 2, 3), capsule_height_m=torch.zeros(n, 2),
        capsule_lowest_point_velocity_w=torch.zeros(n, 2, 3), support_time_s=torch.full((n, 2), .1),
        shank_gap_m=torch.full((n,), .05), substep_qd_ratio_peak=torch.zeros(n),
        substep_nonfoot_force_peak=torch.zeros(n), actual_delta=torch.zeros(n, 29),
        previous_delta=torch.zeros(n, 29), terminated=torch.zeros(n, dtype=torch.bool))


def evaluate(s, cfg=None):
    m = module()
    return m.recovery_objective(s, m.DEFAULT_OBJECTIVE_CONFIG if cfg is None else cfg)


def test_inactive_prefix_has_exact_zero_reward_and_terminal_cost():
    s = state(2); s['post_active'][0] = False; s['terminated'][0] = True
    s['root_height_m'][0] = .4; s['substep_nonfoot_force_peak'][0] = 100.
    reward, components = evaluate(s)
    assert reward[0] == 0. and reward[1] > 0.
    assert all(x.shape == (2,) and x[0] == 0. for x in components.values())


def test_deceleration_follows_initial_world_direction_and_reaches_stop():
    s = state(3)
    s['activation_root_velocity_w'][:, 1] = -2.
    s['root_velocity_w'][0, 1] = -2.
    s['root_velocity_w'][1, 1] = 2.
    s['time_since_activation'][2] = 2.
    _, c = evaluate(s)
    assert c['planar_velocity_reward'][0] == c['planar_velocity_reward'][2]
    assert c['planar_velocity_reward'][0] > c['planar_velocity_reward'][1]


def test_stability_reward_favors_upright_height_and_controlled_vertical_motion():
    s = state(3)
    s['root_height_m'][1] = .48; s['upright_cos'][1] = .5
    s['root_velocity_w'][2, 2] = 1.; s['root_angular_velocity_w'][2, 0] = 4.
    reward, c = evaluate(s)
    assert reward[0] > reward[1] and reward[0] > reward[2]
    assert c['height_reward'][0] > c['height_reward'][1]
    assert c['upright_reward'][0] > c['upright_reward'][1]
    assert c['vertical_velocity_cost'][2] < 0. and c['root_angular_cost'][2] < 0.


def test_only_low_descending_unloaded_foot_receives_approach_risk():
    s = state(4); s['ground_force_N'][:, 1] = 0.; s['support_time_s'][:, 1] = 0.
    s['capsule_height_m'][:, 1] = .02
    s['capsule_lowest_point_velocity_w'][:, 1] = torch.tensor([2., 0., -1.])
    s['capsule_lowest_point_velocity_w'][1, 1, 2] = 1.  # same slip, moving away
    s['capsule_height_m'][2, 1] = .3  # high flight is not low-clearance approach
    s['ground_force_N'][3, 1] = 160.  # loaded state has a separate slip term
    _, c = evaluate(s)
    for key in ('approach_normal_cost', 'approach_tangent_cost'):
        assert c[key][0] < 0.
        assert torch.equal(c[key][1:], torch.zeros(3))
    assert c['loaded_slip_cost'][3] < 0.


def test_handover_needs_other_confirmed_support_but_does_not_penalize_all_flight():
    s = state(4); s['ground_force_N'][:, 1] = 0.; s['support_time_s'][:, 1] = 0.
    s['capsule_height_m'][:, 1] = .02
    s['capsule_lowest_point_velocity_w'][:, 1, 2] = -1.
    s['support_time_s'][1, 0] = .005
    s['ground_force_N'][2:] = 0.; s['support_time_s'][2:] = 0.
    s['capsule_height_m'][3] = .3
    _, c = evaluate(s)
    assert c['handover_cost'][0] == 0.
    assert c['handover_cost'][1] < 0. and c['handover_cost'][2] < 0.
    assert c['handover_cost'][3] == 0.


def test_limits_self_clearance_and_nonfoot_contact_each_receive_separate_costs():
    s = state(4)
    s['substep_qd_ratio_peak'][:] = torch.tensor([1., 1.05, 1.1, 1.])
    s['shank_gap_m'][3] = -.01; s['substep_nonfoot_force_peak'][3] = 60.
    _, c = evaluate(s)
    assert c['joint_speed_cost'][0] == 0.
    assert c['joint_speed_cost'][2] < c['joint_speed_cost'][1] < 0.
    assert c['shank_clearance_cost'][3] < 0. and c['nonfoot_cost'][3] < 0.


def test_actual_residual_amplitude_and_direction_change_are_penalized_without_mutation():
    s = state(3)
    s['actual_delta'][1:] = .1; s['previous_delta'][1] = .1; s['previous_delta'][2] = -.1
    saved = {k:v.clone() for k,v in s.items()}
    _, c = evaluate(s)
    assert c['residual_magnitude_cost'][0] == 0.
    assert c['residual_magnitude_cost'][1] == c['residual_magnitude_cost'][2] < 0.
    assert c['residual_change_cost'][1] == 0. and c['residual_change_cost'][2] < 0.
    assert all(torch.equal(s[k], v) for k,v in saved.items())


def test_terminal_is_one_signed_event_with_explicit_policy_rate_conversion():
    s = state(2); s['terminated'][1] = True
    reward, c = evaluate(s)
    assert c['terminal_cost'].tolist() == [0., -20.]
    assert c['terminal_rate'].tolist() == [0., -1000.]
    assert reward[1]-reward[0] == pytest.approx(-1000.)


def test_horizontal_rotation_preserves_physical_costs_and_every_component_is_finite():
    s = state(); s['ground_force_N'][:, 1] = 0.; s['support_time_s'][:, 1] = 0.
    s['root_velocity_w'][0] = torch.tensor([.4, -.8, .1])
    s['activation_root_velocity_w'][0] = torch.tensor([.1, -2., 0.])
    s['capsule_lowest_point_velocity_w'][0, 1] = torch.tensor([1., -2., -.5])
    s['foot_angular_velocity_w'][0, 1] = torch.tensor([3., 2., 1.])
    turned = copy.deepcopy(s)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for key in ('root_velocity_w', 'activation_root_velocity_w', 'root_angular_velocity_w',
                'capsule_lowest_point_velocity_w', 'foot_angular_velocity_w'):
        turned[key] = turned[key] @ rotation.T
    ra, a = evaluate(s); rb, b = evaluate(turned)
    torch.testing.assert_close(ra, rb)
    for key in a:
        torch.testing.assert_close(a[key], b[key])
        assert torch.isfinite(a[key]).all()


@pytest.mark.parametrize('kind', ['nan_state', 'wrong_shape', 'zero_scale', 'negative_force'])
def test_invalid_physical_inputs_or_scales_are_rejected(kind):
    m = module(); s = state(); cfg = copy.deepcopy(m.DEFAULT_OBJECTIVE_CONFIG)
    if kind == 'nan_state': s['capsule_lowest_point_velocity_w'][0, 1, 0] = float('nan')
    if kind == 'wrong_shape': s['actual_delta'] = torch.zeros(1, 28)
    if kind == 'zero_scale': cfg['velocity_scale_mps'] = 0.
    if kind == 'negative_force': s['ground_force_N'][0, 0] = -1.
    with pytest.raises(ValueError): m.recovery_objective(s, cfg)
