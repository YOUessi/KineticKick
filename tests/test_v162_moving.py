import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import v162_moving
from v162_moving_batch import geometry


def test_rolling_waypoint_uses_world_velocity_before_contact_and_true_ball_after(monkeypatch):
    from contextlib import contextmanager
    inputs = []

    def record(q, dq, root, omega, ball, goal, prev, ref, frame, anchor, offsets):
        inputs.append((np.array(ball), prev.copy(),np.array(goal)))
        return np.zeros(160)

    @contextmanager
    def service(assets,protocol):
        meta={'anchor_body_name':'torso_link','action_scale':','.join(['1']*29)}
        yield meta, lambda frame,obs: np.ones(29) * 2

    monkeypatch.setattr(v162_moving.physical, 'digest', lambda path:json.loads(v162_moving.physical.PROTOCOL.read_text())['asset_hashes']['.pretrained_checkpoints/paid_official/policy_30000.onnx'])
    monkeypatch.setattr(v162_moving.physical, 'cpu_policy', service)
    monkeypatch.setattr(v162_moving.physical, 'observation', record)
    planner=v162_moving.stance_planner({}, {'body_names':['torso_link']},
                                      {'nominal_contact_s':2., 'rolling_decay_per_s':.015})
    ball=np.array([.1,-1.2,.11]);goal=np.array([0.,-3.7,.11]);initial=ball.copy()
    ctx=dict(t=1.,ball_pos=ball,ball_velocity=np.array([.3,-.2,0.]),first_contact_s=None,
             root_pos=np.zeros(3),root_quat=np.array([1,0,0,0]),root_velocity=np.zeros(6),
             q=np.zeros(29),dq=np.zeros(29),goal=goal,frame=0,default_q=np.zeros(29))
    planner(ctx)
    assert np.allclose(inputs[0][0][:2],initial[:2]+ctx['ball_velocity'][:2]*(-np.expm1(-.015)/.015))
    assert np.array_equal(ball,initial) and np.array_equal(inputs[0][2],goal)
    ctx.update(t=1.02,first_contact_s=1.01)
    planner(ctx)
    assert np.array_equal(inputs[1][0],ball) and np.array_equal(inputs[1][1],np.full(29,2.))


def test_preregistered_pair_geometry_preserves_time_and_independent_condition():
    protocol=json.loads((ROOT/'configs/v162_moving_protocol.json').read_text())
    for case in protocol['development']+protocol['holdout']:
        offset, velocity=geometry(case,protocol)
        assert np.allclose(offset+protocol['arrival_time_s']*velocity,case['crossing_offset_xy_m'])
