"""CPU contracts for a one-shot shared support reference; no simulator."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

CFG = dict(minimum_force_N=80., support_confirmation_s=.015, duration_s=.20)


def create(n=1):
    path = Path(__file__).resolve().parents[1] / 'tools/support_reference_frame.py'
    assert path.is_file(), 'The support reference frame is not implemented'
    spec = importlib.util.spec_from_file_location('support_reference_frame', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SupportReferenceFrame(n, 'cpu', CFG)


def inputs(n=1):
    return dict(now=torch.ones(n), ready_mask=torch.ones(n, dtype=torch.bool),
        support_N=torch.tensor([[120., 90.]]).repeat(n, 1),
        reference_feet=torch.tensor([[[1., 2., .7], [-1., -2., .4]]]).repeat(n, 1, 1),
        actual_feet=torch.tensor([[[4., -1., 9.], [2., 3., 8.]]]).repeat(n, 1, 1),
        reference_yaw=torch.zeros(n), actual_yaw=torch.full((n,), math.pi / 2))


def confirm(frame, support):
    for _ in range(3): frame.after_physics(support)


def test_endpoint_aligns_selected_pivot_and_rotates_all_positions_and_vectors_together():
    frame = create(); data = inputs(); confirm(frame, data['support_N'])
    frame.update(**data)
    data['now'] = torch.tensor([1.2]); result = frame.update(**data)
    positions = torch.tensor([[[1., 2., 3.], [2., 2., 4.], [1., 4., 5.]]])
    transformed = frame.transform_positions(positions)
    torch.testing.assert_close(transformed, torch.tensor([[[4., -1., 3.], [4., 0., 4.], [2., -1., 5.]]]))
    vectors = torch.tensor([[[1., 0., 7.], [0., 1., -2.]]])
    torch.testing.assert_close(frame.transform_vectors(vectors), torch.tensor([[[0., 1., 7.], [-1., 0., -2.]]]))
    assert torch.equal(transformed[..., 2], positions[..., 2])
    torch.testing.assert_close(torch.cdist(transformed, transformed), torch.cdist(positions, positions))
    assert result['selected_side'].tolist() == [0]
    assert result['alpha'].tolist() == [1.]
    assert result['yaw_rate'].tolist() == [0.]
    torch.testing.assert_close(result['rotation'] @ result['rotation'].transpose(1, 2), torch.eye(3)[None])


def test_untriggered_and_trigger_alpha_zero_are_bitwise_identity():
    frame = create(); data = inputs()
    values = torch.tensor([[[.123456789012345, -2.718281828459, 8.123456789],
                            [1e9 + .25, 1e-12, -3.75]]], dtype=torch.float64)
    frame.update(**data)
    assert torch.equal(frame.transform_positions(values), values)
    assert torch.equal(frame.transform_vectors(values[:, 0]), values[:, 0])
    confirm(frame, data['support_N']); state = frame.update(**data)
    assert state['trigger_time'].tolist() == [1.]
    assert state['alpha'].tolist() == [0.]
    assert torch.equal(frame.transform_positions(values), values)
    assert torch.equal(frame.transform_vectors(values), values)


def test_shortest_yaw_and_quintic_midpoint_have_consistent_yaw_rate():
    frame = create(); data = inputs()
    data['reference_yaw'] = torch.tensor([math.radians(170.)])
    data['actual_yaw'] = torch.tensor([math.radians(-170.)])
    confirm(frame, data['support_N']); frame.update(**data)
    data['now'] = torch.tensor([1.1]); state = frame.update(**data)
    assert float(state['yaw_delta'][0]) == pytest.approx(math.radians(20.), abs=1e-6)
    assert float(state['alpha'][0]) == pytest.approx(.5, abs=1e-6)
    # C2 derivative at u=.5 is 1.875 / duration.
    assert float(state['yaw_rate'][0]) == pytest.approx(math.radians(20.) * 1.875 / .2, abs=2e-5)
    direction = frame.transform_vectors(torch.tensor([[1., 0., 2.]]))
    torch.testing.assert_close(direction, torch.tensor([[math.cos(math.radians(10.)), math.sin(math.radians(10.)), 2.]]))
    pivot = frame.transform_positions(data['reference_feet'][:, 0])
    torch.testing.assert_close(pivot, torch.tensor([[2.5, .5, .7]]))


def test_gate_requires_continuous_support_selects_eligible_side_and_never_relatches():
    frame = create(2); data = inputs(2)
    data['support_N'] = torch.tensor([[0., 100.], [80., 79.]])
    frame.after_physics(data['support_N']); frame.after_physics(data['support_N'])
    assert frame.update(**data)['selected_side'].tolist() == [-1, -1]
    frame.after_physics(data['support_N'])
    data['ready_mask'] = torch.tensor([False, True])
    assert frame.update(**data)['selected_side'].tolist() == [-1, 0]
    # A stronger left force is not eligible after only one 5 ms sample.
    data['support_N'][0] = torch.tensor([200., 100.]); frame.after_physics(data['support_N'])
    data['ready_mask'][:] = True; data['now'][:] = 1.02
    assert frame.update(**data)['selected_side'].tolist() == [1, 0]
    frozen = {k: getattr(frame, k).clone() for k in ('reference_pivot', 'actual_pivot', 'yaw_delta', 'trigger_time', 'selected_side')}
    data['reference_feet'] += 10.; data['actual_feet'] -= 20.; data['actual_yaw'] *= -1
    data['now'][:] = 1.3; data['support_N'][:] = 0.
    frame.after_physics(data['support_N']); frame.update(**data)
    for key, value in frozen.items(): assert torch.equal(getattr(frame, key), value)
    frame.reset(torch.tensor([1]))
    assert frame.selected_side.tolist() == [1, -1]
    assert frame.alpha.tolist() == [1., 0.]
    assert frame.support_time[1].tolist() == [0., 0.]
    assert torch.equal(frame.rotation[1], torch.eye(3))
    assert frame.update(**data)['selected_side'].tolist() == [1, -1]


@pytest.mark.parametrize('field', ['now', 'support_N', 'reference_feet', 'actual_feet', 'reference_yaw', 'actual_yaw'])
def test_nonfinite_candidate_inputs_do_not_latch_or_contaminate_another_environment(field):
    frame = create(2); data = inputs(2); confirm(frame, data['support_N'])
    data[field][0] = float('-inf') if field in ('support_N', 'actual_yaw') else float('nan')
    state = frame.update(**data)
    assert state['trigger_time'].tolist() == [-1., 1.]
    assert state['selected_side'].tolist() == [-1, 0]
    assert torch.isfinite(state['rotation']).all()
    assert torch.isfinite(state['alpha']).all()
    assert torch.isfinite(state['yaw_rate']).all()


def test_nonfinite_force_breaks_its_confirmation_and_invalid_dt_is_rejected():
    frame = create(); data = inputs(); confirm(frame, data['support_N'])
    frame.after_physics(torch.tensor([[float('inf'), 0.]]))
    assert frame.support_time.tolist() == [[0., 0.]]
    assert frame.update(**data)['selected_side'].tolist() == [-1]
    with pytest.raises(ValueError): frame.after_physics(data['support_N'], dt=.02)
