"""Pure CPU reward contracts; no simulator or physical-success claim."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from prestrike_objective import prestrike_rate, shot_event


def state(n=3):
    return dict(eligible=torch.ones(n, dtype=torch.bool), pre_control=torch.ones(n, dtype=torch.bool),
        root_height_m=torch.full((n,), .78), reference_height_m=torch.full((n,), .78),
        upright_cos=torch.ones(n), root_velocity_w=torch.tensor([[0., -1.8, 0.]]*n),
        reference_velocity_w=torch.tensor([[0., -1.8, 0.]]*n),
        joint_position_error_rad=torch.zeros(n,29), qd_ratio=torch.full((n,), .5),
        actual_delta=torch.zeros(n,29), previous_delta=torch.zeros(n,29))


def test_prefix_is_zero_and_post_fade_keeps_only_eligible_speed_margin_cost():
    s = state()
    s['eligible'] = torch.tensor([False, True, True])
    s['pre_control'] = torch.tensor([True, True, False])
    s['qd_ratio'].fill_(1.01)
    before = {k:v.clone() for k,v in s.items()}
    reward, terms = prestrike_rate(s)
    torch.testing.assert_close(reward, torch.tensor([0., 4., -.5]), atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(reward, torch.stack(list(terms.values())).sum(0))
    assert all(torch.equal(s[k],v) for k,v in before.items())
    assert terms['velocity_reward'][1] == 1.  # Tracks running, never an early stop target.
    assert terms['height_reward'][2] == 0.


def test_reference_tracking_and_actual_reference_regularization_have_fixed_units():
    s = state(1)
    s['root_height_m'] += .1
    s['upright_cos'] -= .1
    s['root_velocity_w'][0,0] += .8
    s['joint_position_error_rad'].fill_(.25)
    s['actual_delta'].fill_(.02)
    s['previous_delta'].fill_(.012)
    reward, terms = prestrike_rate(s)
    expected = 4.5*torch.exp(torch.tensor(-1.))-.15*.25-.1
    torch.testing.assert_close(reward[0], expected)
    assert terms['residual_magnitude_cost'][0] == pytest.approx(-.0375)
    assert terms['residual_change_cost'][0] == pytest.approx(-.1)
    assert not any('terminal' in k or 'goal' in k or 'event' in k for k in terms)


def test_speed_margin_is_not_a_duplicate_overspeed_event():
    s = state(4)
    s['pre_control'].zero_()
    s['qd_ratio'] = torch.tensor([.98, .995, 1.13, 10.])
    rate, _ = prestrike_rate(s)
    torch.testing.assert_close(rate, torch.tensor([0., -.125, -12.5, -12.5]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('change', ['nonfinite', 'shape', 'negative_ratio', 'mask_dtype', 'mixed_dtype'])
def test_invalid_state_is_rejected_even_when_not_eligible(change):
    s = state(1); s['eligible'].zero_()
    if change == 'nonfinite': s['actual_delta'][0,0] = float('nan')
    if change == 'shape': s['joint_position_error_rad'] = torch.zeros(1,28)
    if change == 'negative_ratio': s['qd_ratio'][0] = -.1
    if change == 'mask_dtype': s['pre_control'] = torch.ones(1)
    if change == 'mixed_dtype': s['reference_velocity_w'] = s['reference_velocity_w'].double()
    with pytest.raises(ValueError): prestrike_rate(s)


def test_shot_bonus_is_a_single_event_uses_world_direction_and_requires_both_gates():
    velocity = torch.tensor([[4.5,0.,10.], [4.5,.5,0.], [3.5,0.,0.], [4.5,0.,0.], [4.5,0.,0.]])
    direction = torch.tensor([[2.,0.]]*5)
    eligible = torch.tensor([True,True,True,False,True])
    legal = torch.tensor([True,True,True,True,False])
    bonus, terms = shot_event(velocity,direction,eligible,legal)
    one_error = 10.*torch.exp(torch.tensor(-1.))
    torch.testing.assert_close(bonus, torch.tensor([10.,one_error,one_error,0.,0.]))
    torch.testing.assert_close(terms['forward_speed_mps'],velocity[:,0])
    torch.testing.assert_close(terms['lateral_speed_mps'],velocity[:,1])
    assert bonus.max() == 10.  # Not 10/.02; runtime consumes the event exactly once.
    rotated, _ = shot_event(torch.tensor([[0.,-4.5,0.]]),torch.tensor([[0.,-3.]]),
                           torch.tensor([True]),torch.tensor([True]))
    assert rotated[0] == 10.


@pytest.mark.parametrize('change', ['zero_direction', 'nonfinite', 'bad_mask'])
def test_invalid_shot_inputs_are_not_hidden_by_ineligible_mask(change):
    v, d = torch.zeros(1,3), torch.tensor([[1.,0.]])
    gate = torch.tensor([False])
    if change == 'zero_direction': d.zero_()
    if change == 'nonfinite': v[0,0] = float('inf')
    if change == 'bad_mask': gate = torch.zeros(1)
    with pytest.raises(ValueError): shot_event(v,d,gate,torch.tensor([False]))
