#!/usr/bin/env python3
"""Export a PAiD warm-start PPO checkpoint into the official strict ONNX schema.

CPU only; no Isaac Sim, GPU, or training. Only actor/LSTM/normalization
initializers change: the official reference outputs, graph, schema, and metadata
are retained. Deterministic action/h/c parity is checked before writing output.

Example:
  python /tmp/v010_export_running_actor.py --checkpoint /path/last_model.pt \
    --output /path/running.onnx

Accepts OnPolicyRunner checkpoints containing model_state_dict or a plain policy
state dict. Requires the paid_obs_mean and paid_obs_denominator buffers from
PaidActorCriticRecurrent. Exploration std and critic are deliberately not exported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn

DEFAULT_SOURCE = '/home/you/projects/kinetickick/.pretrained_checkpoints/paid_official/policy_30000.onnx'
LOCAL_ORT = '/home/you/projects/kinetickick/experiments/runtime/onnxruntime'


class CheckpointActor(nn.Module):
    """Deterministic actor from the trained Torch state, independent of RSL/Sim."""
    def __init__(self, state):
        super().__init__()
        self.rnn = nn.LSTM(160, 128, 2)
        self.actor = nn.Sequential(nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 64),
                                   nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 29))
        self.rnn.load_state_dict({k.removeprefix('memory_a.rnn.'): v for k, v in state.items()
                                 if k.startswith('memory_a.rnn.')}, strict=True)
        self.actor.load_state_dict({k.removeprefix('actor.'): v for k, v in state.items()
                                   if k.startswith('actor.')}, strict=True)
        for key in ('paid_obs_mean', 'paid_obs_denominator'):
            value = state[key].detach().cpu().float()
            if value.shape != (1, 160) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid checkpoint normalization buffer: {key}')
            self.register_buffer(key, value)
        if not (self.paid_obs_denominator > 0).all():
            raise ValueError('Normalization denominator must be positive')
        self.eval()

    def forward(self, obs, h, c):
        normalized = (obs - self.paid_obs_mean) / self.paid_obs_denominator
        sequence, (new_h, new_c) = self.rnn(normalized.unsqueeze(0), (h, c))
        return self.actor(sequence.squeeze(0)), new_h, new_c


def torch_to_onnx_gates(value):
    """Torch i,f,g,o -> ONNX i,o,f,c; g and c are the same candidate gate."""
    array = value.detach().cpu().float().numpy()
    chunks = np.split(array, 4, axis=0)
    return np.concatenate([chunks[j] for j in (0, 3, 1, 2)], axis=0)


def replace_actor_initializers(model, state):
    initializers = {value.name: value for value in model.graph.initializer}
    changed = []

    def replace(name, array):
        array = np.asarray(array, dtype=np.float32)
        expected = tuple(initializers[name].dims)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f'Invalid initializer {name}: {array.shape}; expected {expected}')
        initializers[name].CopyFrom(numpy_helper.from_array(array, name=name))
        changed.append(name)

    for layer in (0, 2, 4, 6):
        for kind in ('weight', 'bias'):
            key = f'actor.{layer}.{kind}'
            replace(f'parent.{key}', state[key].detach().cpu().numpy())
    lstms = [node for node in model.graph.node if node.op_type == 'LSTM']
    if len(lstms) != 2:
        raise ValueError('Expected exactly two official LSTM graph nodes')
    for layer, node in enumerate(lstms):
        prefix = 'memory_a.rnn.'
        replace(node.input[1], torch_to_onnx_gates(state[f'{prefix}weight_ih_l{layer}'])[None])
        replace(node.input[2], torch_to_onnx_gates(state[f'{prefix}weight_hh_l{layer}'])[None])
        bias_input = torch_to_onnx_gates(state[f'{prefix}bias_ih_l{layer}'])
        bias_hidden = torch_to_onnx_gates(state[f'{prefix}bias_hh_l{layer}'])
        replace(node.input[3], np.concatenate((bias_input, bias_hidden))[None])
    sub = next(node for node in model.graph.node if node.op_type == 'Sub' and node.input[0] == 'obs')
    div = next(node for node in model.graph.node if node.op_type == 'Div' and node.input[0] == sub.output[0])
    replace(sub.input[1], state['paid_obs_mean'].detach().cpu().numpy())
    replace(div.input[1], state['paid_obs_denominator'].detach().cpu().numpy())
    return changed


def verify_parity(model_bytes, actor, steps=8):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        sys.path.insert(0, LOCAL_ORT)
        import onnxruntime as ort
    torch.set_num_threads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, sess_options=options, providers=['CPUExecutionProvider'])
    input_shapes = {entry.name: entry.shape for entry in session.get_inputs()}
    expected_shapes = {'obs': [1, 160], 'h_in': [2, 1, 128], 'c_in': [2, 1, 128], 'time_step': [1, 1]}
    if input_shapes != expected_shapes:
        raise ValueError(f'Unexpected exported input schema: {input_shapes}')
    rng = np.random.default_rng(1010)
    h = rng.normal(0, 0.05, (2, 1, 128)).astype(np.float32)
    c = rng.normal(0, 0.05, (2, 1, 128)).astype(np.float32)
    th, tc = torch.from_numpy(h.copy()), torch.from_numpy(c.copy())
    maximum = dict(actions=0.0, h=0.0, c=0.0)
    with torch.inference_mode():
        for step in range(steps):
            obs = rng.normal(0, 0.1, (1, 160)).astype(np.float32)
            if step % 2:
                obs = actor.paid_obs_mean.numpy() + obs * actor.paid_obs_denominator.numpy()
            expected, h, c = session.run(['actions', 'h_out', 'c_out'],
                {'obs': obs, 'h_in': h, 'c_in': c, 'time_step': np.array([[step]], np.float32)})
            actual, th, tc = actor(torch.from_numpy(obs), th, tc)
            for key, left, right in [('actions', actual.numpy(), expected),
                                     ('h', th.numpy(), h), ('c', tc.numpy(), c)]:
                maximum[key] = max(maximum[key], float(np.max(np.abs(left - right))))
                if not np.allclose(left, right, rtol=1e-5, atol=2e-5):
                    raise AssertionError(f'Export parity failed at step {step}, {key}: {maximum[key]}')
    return {'cpu_only': True, 'seed': 1010, 'steps': steps, 'max_abs_error': maximum,
            'input_schema': input_shapes, 'output_names': [entry.name for entry in session.get_outputs()]}


def export_checkpoint(checkpoint, output, source=DEFAULT_SOURCE, overwrite=False):
    checkpoint, output, source = map(Path, (checkpoint, output, source))
    if output.resolve() in (source.resolve(), checkpoint.resolve()):
        raise ValueError('Output must differ from the source ONNX and checkpoint')
    if output.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite {output}; pass --overwrite explicitly')
    if not output.parent.is_dir():
        raise FileNotFoundError(f'Output directory does not exist: {output.parent}')
    saved = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    state = saved.get('model_state_dict', saved)
    actor = CheckpointActor(state).cpu()
    model = onnx.load(str(source))
    original_metadata = [(entry.key, entry.value) for entry in model.metadata_props]
    original_graph = [node.SerializeToString() for node in model.graph.node]
    changed = replace_actor_initializers(model, state)
    if original_metadata != [(entry.key, entry.value) for entry in model.metadata_props]:
        raise AssertionError('Source metadata changed')
    if original_graph != [node.SerializeToString() for node in model.graph.node]:
        raise AssertionError('Source reference graph changed')
    onnx.checker.check_model(model)
    model_bytes = model.SerializeToString()
    result = verify_parity(model_bytes, actor)
    with output.open('wb' if overwrite else 'xb') as stream:
        stream.write(model_bytes)
    result.update(checkpoint=str(checkpoint.resolve()), source=str(source.resolve()), output=str(output.resolve()),
                  output_sha256=hashlib.sha256(model_bytes).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  initializers_replaced=changed, reference_graph_preserved=True, metadata_preserved=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default=DEFAULT_SOURCE)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--report', help='Optional JSON parity report path')
    args = parser.parse_args()
    if args.report and Path(args.report).exists() and not args.overwrite:
        raise FileExistsError(f'Refusing to overwrite report {args.report}')
    if args.report and Path(args.report).resolve() in {Path(p).resolve() for p in (args.checkpoint, args.output, args.source)}:
        raise ValueError('Report path must differ from checkpoint/source/output')
    result = export_checkpoint(args.checkpoint, args.output, args.source, args.overwrite)
    if args.report:
        with Path(args.report).open('w' if args.overwrite else 'x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
