"""Bounded joint-target corrections during the strike, measured in radians.

The recurrent policy still controls approach, balance and recovery. These
corrections only pass through the robot's normal joint position actuators.
"""
from __future__ import annotations

import numpy as np


class StrikeResidual:
    def __init__(self, entries, joint_names, scale, offset, limits):
        shape = (len(entries), len(joint_names))
        self.scale = np.broadcast_to(np.asarray(scale), shape)
        self.offset = np.broadcast_to(np.asarray(offset), shape)
        self.limits = np.broadcast_to(np.asarray(limits), (*shape, 2))
        self.windows = np.tile([80., 110., 110., 150.], (len(entries), 1))
        self.delta = np.zeros(shape)
        self.targets = np.full(shape, np.nan)
        for i, entry in enumerate(entries):
            window = entry.get('window', self.windows[i])
            if len(window) == 3:
                window = [window[0], window[1], window[1], window[2]]
            if len(window) != 4 or not np.all(np.isfinite(window)) or not (window[0] < window[1] <= window[2] < window[3]):
                raise ValueError('Strike window requires finite start < rise <= fall < end')
            self.windows[i] = window
            for name, value in entry.get('offsets', {}).items():
                if name not in joint_names or not np.isfinite(value):
                    raise ValueError(f'Invalid joint correction: {name}')
                self.delta[i, joint_names.index(name)] = float(value)
            for name, value in entry.get('targets', {}).items():
                if name not in joint_names or not np.isfinite(value):
                    raise ValueError(f'Invalid joint target: {name}')
                self.targets[i, joint_names.index(name)] = float(value)
        if np.any(self.scale <= 0):
            raise ValueError('Action scales must be positive')

    def envelope(self, frames, dt=1.):
        frames = np.broadcast_to(np.asarray(frames), (len(self.windows),))
        start, rise, fall, end = self.windows.T
        phase = np.where(frames <= rise, (frames-start)/(rise-start), (end-frames)/(end-fall))
        rate = np.where(frames <= rise, 1/(rise-start), -1/(end-fall)) / dt
        rate = np.where((phase > 0.) & (phase < 1.), rate, 0.)
        phase = np.clip(phase, 0., 1.)
        weight = .5 - .5 * np.cos(np.pi * phase)
        return weight[:, None], (.5 * np.pi * np.sin(np.pi * phase) * rate)[:, None]

    def trajectory(self, frames, dt=1.):
        weight, rate = self.envelope(frames, dt)
        correction = self.delta * weight
        velocity = self.delta * rate
        return correction, velocity

    def apply_reference(self, observations, frames, dt):
        result = np.array(observations, dtype=np.float32, copy=True)
        n = self.delta.shape[1]
        delta, velocity = self.trajectory(frames, dt)
        target = result[:, :n] + delta
        clipped = np.clip(target, self.limits[:, :, 0], self.limits[:, :, 1])
        mask = delta != 0.
        result[:, :n] = np.where(mask, clipped, result[:, :n])
        result[:, n:2*n] = np.where(mask, np.where(clipped == target, result[:, n:2*n] + velocity, 0.),
                                   result[:, n:2*n])
        return result

    def apply(self, actions, frames):
        correction, _ = self.trajectory(frames)
        weight, _ = self.envelope(frames)
        original = np.asarray(actions) * self.scale + self.offset
        targets = original + correction
        absolute = np.isfinite(self.targets) & (weight > 0.)
        targets = np.where(absolute, original * (1-weight) + self.targets * weight, targets)
        targets = np.clip(targets, self.limits[:, :, 0], self.limits[:, :, 1])
        modified = (targets-self.offset) / self.scale
        return np.where((correction != 0.) | absolute, modified, actions).astype(np.float32)
