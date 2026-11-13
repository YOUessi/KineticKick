"""Serve the released recurrent PAiD policy on CPU across Tang's Python ABIs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.assets / 'experiments/runtime/onnxruntime'))
    from soccer_onnx_policy import OnnxSoccerPolicy

    actor = OnnxSoccerPolicy(args.assets / '.pretrained_checkpoints/paid_official/policy_30000.onnx', 1)
    if actor.session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('A CPU-only ONNX session is required')
    print(json.dumps({'metadata': actor.metadata, 'providers': actor.session.get_providers()}), flush=True)
    for line in sys.stdin:
        frame, observation = json.loads(line)
        action = actor.step(np.asarray(observation, dtype=np.float32)[None], frame)
        print(json.dumps(action[0].tolist(), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
