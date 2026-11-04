"""CPU conversion of the official PAiD ONNX actor into trainable RSL-RL weights.

Runner integration (before constructing OnPolicyRunner):
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.PaidActorCriticRecurrent = PaidActorCriticRecurrent
    cfg['empirical_normalization'] = False
    cfg['policy'] = dict(class_name='PaidActorCriticRecurrent',
                        onnx_path=path, init_noise_std=0.15)

Normalization is in the actor, so raw environment observations are accepted on
both the first rollout step and all subsequent steps. Critic starts fresh and
receives raw privileged observations. Actor weights and std remain trainable.
The normalization buffers are frozen and included in checkpoints.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import numpy_helper
import torch
from rsl_rl.modules import ActorCriticRecurrent

DEFAULT_ONNX = '/home/you/projects/kinetickick/.pretrained_checkpoints/paid_official/policy_30000.onnx'


def extract_actor(onnx_path=DEFAULT_ONNX):
    """Return RSL actor state and the exact ONNX normalization arrays."""
    model = onnx.load(str(onnx_path))
    tensors = {t.name: np.array(numpy_helper.to_array(t), copy=True) for t in model.graph.initializer}
    state = {}
    for i in (0, 2, 4, 6):
        for kind in ('weight', 'bias'):
            key = f'actor.{i}.{kind}'
            state[key] = torch.from_numpy(tensors[f'parent.{key}'])
    lstms = [n for n in model.graph.node if n.op_type == 'LSTM']
    assert len(lstms) == 2, 'Expected the official two-layer LSTM'

    # ONNX i,o,f,c -> PyTorch i,f,g,o; c and g denote the same candidate gate.
    def reorder(value):
        return np.concatenate([np.split(value, 4, axis=0)[j] for j in (0, 2, 3, 1)], axis=0)

    for layer, node in enumerate(lstms):
        w, r, b = [tensors[node.input[j]][0] for j in (1, 2, 3)]
        assert w.shape == (512, 160 if layer == 0 else 128)
        assert r.shape == (512, 128) and b.shape == (1024,)
        bias_input, bias_hidden = np.split(b, 2)
        for name, value in (('weight_ih', w), ('weight_hh', r),
                            ('bias_ih', bias_input), ('bias_hh', bias_hidden)):
            state[f'memory_a.rnn.{name}_l{layer}'] = torch.from_numpy(reorder(value))
    sub = next(n for n in model.graph.node if n.op_type == 'Sub' and n.input[0] == 'obs')
    div = next(n for n in model.graph.node if n.op_type == 'Div' and n.input[0] == sub.output[0])
    mean, denominator = tensors[sub.input[1]], tensors[div.input[1]]
    assert mean.shape == denominator.shape == (1, 160)
    assert np.isfinite(mean).all() and np.isfinite(denominator).all() and (denominator > 0).all()
    return state, torch.from_numpy(mean), torch.from_numpy(denominator)


class PaidActorCriticRecurrent(ActorCriticRecurrent):
    """RSL recurrent PPO actor with exact, fixed ONNX observation normalization."""
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, onnx_path=DEFAULT_ONNX,
                 init_noise_std=0.15, **kwargs):
        if (num_actor_obs, num_actions) != (160, 29):
            raise ValueError('The PAiD actor requires 160 observations and 29 actions')
        required = dict(actor_hidden_dims=[128, 64, 32], critic_hidden_dims=[128, 64, 32],
                        activation='elu', rnn_type='lstm', rnn_hidden_dim=128, rnn_num_layers=2)
        for key, value in required.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f'{key} must be {value!r}')
            kwargs[key] = value
        super().__init__(num_actor_obs, num_critic_obs, num_actions,
                         init_noise_std=init_noise_std, **kwargs)
        state, mean, denominator = extract_actor(onnx_path)
        own = self.state_dict()
        for key, value in state.items():
            if own[key].shape != value.shape:
                raise ValueError(f'Unexpected shape for {key}')
            own[key] = value
        self.load_state_dict(own)
        self.register_buffer('paid_obs_mean', mean)
        self.register_buffer('paid_obs_denominator', denominator)
        assert self.std.requires_grad

    def normalize_actor_obs(self, observations):
        return (observations - self.paid_obs_mean) / self.paid_obs_denominator

    def act(self, observations, masks=None, hidden_states=None):
        return super().act(self.normalize_actor_obs(observations), masks, hidden_states)

    def act_inference(self, observations):
        return super().act_inference(self.normalize_actor_obs(observations))


def load_matching_critic(policy, checkpoint_path):
    """Optional all-or-nothing critic restore; reject incompatible architectures."""
    saved = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    source = saved['model_state_dict']
    target = policy.state_dict()
    names = [k for k in target if k.startswith(('critic.', 'memory_c.'))]
    if any(k not in source or source[k].shape != target[k].shape for k in names):
        raise ValueError('Checkpoint critic is not architecture-compatible; use fresh critic')
    for k in names:
        target[k] = source[k]
    policy.load_state_dict(target)


def check_parity(onnx_path=DEFAULT_ONNX, sequence_length=12):
    """Bounded CPU deterministic check including nonzero state and repeated steps."""
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        sys.path.insert(0, '/home/you/projects/kinetickick/experiments/runtime/onnxruntime')
        import onnxruntime as ort
    torch.set_num_threads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    policy = PaidActorCriticRecurrent(160, 286, 29, onnx_path=onnx_path).cpu().eval()
    rng = np.random.default_rng(1010)
    errors = {'actions': 0.0, 'h': 0.0, 'c': 0.0}
    # Probe small raw observations, then near the real training mean. States
    # are carried across 12 steps; second sequence starts with nonzero h/c.
    with torch.inference_mode():
        for case in range(2):
            h = (rng.normal(size=(2, 1, 128)) * (0.1 if case else 0)).astype(np.float32)
            c = (rng.normal(size=(2, 1, 128)) * (0.1 if case else 0)).astype(np.float32)
            policy.memory_a.hidden_states = (torch.from_numpy(h.copy()), torch.from_numpy(c.copy()))
            for step in range(sequence_length):
                obs = rng.normal(0, 0.1, (1, 160)).astype(np.float32)
                if case:
                    obs = policy.paid_obs_mean.numpy() + obs * policy.paid_obs_denominator.numpy()
                expected, h, c = session.run(['actions', 'h_out', 'c_out'],
                    {'obs': obs, 'h_in': h, 'c_in': c, 'time_step': np.array([[step]], dtype=np.float32)})
                actual = policy.act_inference(torch.from_numpy(obs)).numpy()
                th, tc = (x.numpy() for x in policy.memory_a.hidden_states)
                for key, error in [('actions', np.max(abs(actual-expected))),
                                   ('h', np.max(abs(th-h))), ('c', np.max(abs(tc-c)))]:
                    errors[key] = max(errors[key], float(error))
    assert max(errors.values()) < 2e-5, errors
    result = dict(seed=1010, cpu=True, steps=sequence_length*2,
                  observation_source='seeded random small and exported-mean-relative',
                  max_abs_error=errors, actor_std_trainable=policy.std.requires_grad,
                  critic='fresh recurrent; local stage1 checkpoint incompatible')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-path', default=DEFAULT_ONNX)
    parser.add_argument('--report', default='/tmp/v010_actor_warmstart_parity.json')
    args = parser.parse_args()
    result = check_parity(args.onnx_path)
    Path(args.report).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
