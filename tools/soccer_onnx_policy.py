"""Strict adapter for the official PAiD recurrent soccer policy.

Weights and the optional isolated runtime remain local. The adapter never changes
the observation layout or silently pads/truncates model inputs.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


class OnnxSoccerPolicy:
    def __init__(self, path, num_envs):
        try:
            import onnxruntime as ort
        except ModuleNotFoundError:
            local = Path(__file__).resolve().parents[1] / "experiments/runtime/onnxruntime"
            if not (local / "onnxruntime").is_dir():
                raise ImportError("Install ONNX Runtime in the project-local runtime; see VERSION_CONTROL.md") from None
            sys.path.insert(0, str(local))
            import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.joint_names = self.metadata["joint_names"].split(",")
        self.body_names = self.metadata["body_names"].split(",")
        self.observation_names = self.metadata["observation_names"].split(",")
        inputs = {x.name: x.shape for x in self.session.get_inputs()}
        if inputs != {"obs": [1, 160], "h_in": [2, 1, 128], "c_in": [2, 1, 128], "time_step": [1, 1]}:
            raise ValueError(f"Unsupported policy input contract: {inputs}")
        if len(self.joint_names) != 29 or int(num_envs) < 1:
            raise ValueError("Expected the 29DoF policy and at least one environment")
        self.num_envs = int(num_envs)
        self.reset()

    def reset(self):
        self.h = np.zeros((self.num_envs, 2, 1, 128), np.float32)
        self.c = np.zeros_like(self.h)

    def validate_joints(self, names):
        if list(names) != self.joint_names:
            raise ValueError("Robot joint ordering differs from the policy metadata")

    def step(self, observations, frame):
        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape != (self.num_envs, 160) or not np.all(np.isfinite(obs)):
            raise ValueError("Expected finite observations with shape (num_envs, 160)")
        frames = np.broadcast_to(np.asarray(frame, np.float32), (self.num_envs,))
        if not np.all(np.isfinite(frames)) or np.any(frames < 0):
            raise ValueError("Reference frame must be finite and nonnegative")
        actions = []
        for i in range(self.num_envs):
            result = self.session.run(["actions", "h_out", "c_out"],
                {"obs": obs[i:i+1], "h_in": self.h[i], "c_in": self.c[i],
                 "time_step": np.array([[frames[i]]], np.float32)})
            action, self.h[i], self.c[i] = result
            if action.shape != (1, 29) or not np.all(np.isfinite(action)):
                raise ValueError("Policy returned invalid joint actions")
            actions.append(action[0])
        return np.stack(actions)
